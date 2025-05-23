"""Description: Cartpole n-link environment.
At upright position, all joint angles are at 0 radian. Positive angle is counter-clockwise.
"""

import torch
from torch.autograd import Function
import numpy as np
import ipdb
import time
import sys

import cartpole1l
import cartpole1l_v2
import cartpole2l


import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
deqmpc_dir = os.path.join(project_dir, 'deqmpc')
sys.path.insert(0, deqmpc_dir)
from utils import *
env_dir = os.path.join(deqmpc_dir, 'my_envs')
sys.path.insert(0, env_dir)
from dynamics import DynamicsFunction, Dynamics


class CartpoleDynamics(Dynamics):
    def __init__(self, nx=None, dt=0.01, kwargs=None):
        super().__init__(nx, dt, kwargs)
        if nx == 6:
            self.package = cartpole2l
            print("Using 2-link cartpole dynamics")
        elif nx == 4:
            self.package = cartpole1l
            # self.package = cartpole1l_v2
            print("Using 1-link cartpole dynamics")
        else:
            raise NotImplementedError
    
class CartpoleEnv(torch.nn.Module):
    def __init__(self, nx=None, dt=0.05, stabilization=False, kwargs=None):
        super().__init__()
        assert nx is not None
        self.dynamics = CartpoleDynamics(
            nx=nx, dt=dt, kwargs=kwargs
        )
        self.dynamics_derivatives = self.dynamics.dynamics_derivatives
        self.nx = nx
        self.spec_id = "Cartpole{}l-v0{}".format(
            nx // 2 - 1, "-stabilize" if stabilization else ""
        )
        self.nq = self.dynamics.nq
        self.nu = self.dynamics.nu
        self.dt = dt
        self.kwargs = kwargs
        self.stabilization = stabilization
        self.num_successes = 0
        
        self.bsz = 100
        if nx == 6:
            self.T = 300
            self.u_bounds = 250.0
        elif nx == 4:
            self.T = 200
            self.u_bounds = 100.0
        self.num_steps = 0
        # create observation space based on nx, position and velocity
        high = np.concatenate(
            (np.full(self.nq, np.pi), np.full(self.nq, np.pi * 5)))
        self.observation_space = Spaces(-high, high, (self.nx,))
        self.action_space = Spaces(
            np.full(self.nu, -self.u_bounds),
            np.full(self.nu, self.u_bounds),
            (self.nu,),
        )
        self.stabilization = stabilization
        # self.Qlqr = torch.ones(self.nx, **self.kwargs)
        self.Qlqr = torch.cat([torch.ones(self.nq, **self.kwargs)*1, torch.ones(self.nq, **self.kwargs)*1], dim=-1)
        self.Rlqr = torch.ones(self.nu, **self.kwargs)*0.0000000001
        if nx == 6:
            # self.saved_ckpt_name = "cgac_checkpoint_cartpole2link_swingupeplen300maxu100_initrew1finrew5"
            self.saved_ckpt_name = "cgac_checkpoint_cartpole2link_swingupeplen300dt005maxu250_initrew1finrew5_1"
        elif nx == 4:
            self.saved_ckpt_name = "cgac_checkpoint_cartpole1link_swingupeplen200maxu100_initrew20finrew08"
        

    def action_clip(self, action):
        return torch.clamp(action, -self.u_bounds, self.u_bounds)

    def state_clip(self, state):
        state[..., 1: self.nq] = angle_normalize_2pi(state[..., 1: self.nq])
        return state

    def seed(self, seed):
        """
        Seeds the environment to produce deterministic results.
        Args:
            seed (int): The seed to use.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self, bsz=None):
        """
        Resets the environment to an initial state, which is a random angle and angular velocity.
        Returns:
            numpy.ndarray: The initial state.
        """
        if self.stabilization:
            high = np.concatenate(
                (np.full(self.nq, 0.05), np.full(self.nq, 0.05)))
            high[0], high[1] = 0.1, 0.1  # cart
            offset = torch.tensor([np.pi, 0.0] * self.nq, **self.kwargs)
            offset[0], offset[1] = 0.0, 0.0  # cart
            self.state = (
                torch.tensor(
                    np.random.uniform(low=-high, high=high), **self.kwargs
                )
                + offset
            )
        else:
            high = np.concatenate(
                (np.full(self.nq, np.pi), np.full(self.nq, np.pi))
            )
            if bsz is None:
                bsz = self.bsz
            # high[0], high[1] = 1.0, 1.0  # cart
            high = high[None].repeat(bsz, 0)
            self.state = torch.tensor(
                np.random.uniform(low=-high, high=high), **self.kwargs
            )
            self.state = self.state_clip(self.state)
            # ipdb.set_trace()

        # self.state = torch.tensor(
        #     [0.0, np.pi / 2 + 0.01, 0.0, 0.0, 0.0, 0.0], **self.kwargs)  # fixed

        self.num_successes = 0
        self.num_steps = 0
        # return to_numpy(self.state)
        return self.state

    def step(self, action):
        """
        Applies an action to the environment and steps it forward by one timestep.
        Args:
            action (np array): The action to apply.
        Returns:
            tuple: A tuple containing the new state, reward, done flag, and info dict.
        """
        # action = torch.tensor([action], dtype=torch.float64)
        # print("action:", action[0].norm().item(), "state:", self.state[0].norm().item())
        action = torch.tensor(action, **self.kwargs)
        action = self.action_clip(action)
        # ipdb.set_trace()
        self.state = self.dynamics(self.state, action)
        self.state = self.state_clip(self.state)
        # ipdb.set_trace()
        self.num_steps += 1
        done = self.is_done()
        reward = self.get_reward(
            action
        )  # Define your reward function based on the state and action
        # print("action:", action[0].norm().item(), "state:", self.state[0].norm().item())
        # ipdb.set_trace()
        return self.state, reward, done, {}

    def is_done(self):
        """
        Determines whether the episode is done (e.g., if the pendulum is upright).
        Returns:
            bool: True if the episode is finished, otherwise False.
        """
        x = self.state[..., 0]
        theta = self.state[..., 1:2]
        desired_theta1 = torch.tensor([0.0], **self.kwargs)
        desired_theta2 = torch.tensor([2*np.pi], **self.kwargs) 
        delta_theta = torch.minimum(torch.abs(theta - desired_theta1), torch.abs(theta - desired_theta2))
        success = delta_theta < 0.05# and (torch.abs(x) < 0.05)
        # self.num_successes = 0 if not success else self.num_successes + 1
        return torch.tensor(self.num_successes >= 10 or self.num_steps >= self.T)

    def get_reward(self, action, state=None):
        """
        Calculates the reward for the current state and action.
        Args:
            action (float): The action taken.
        Returns:
            float: The calculated reward.
        """
        # Define your reward function; for simplicity, let's use the negative square of the angle
        # as a reward, so the closer to upright (0 rad), the higher the reward.
        # theta, _ = self.state.unbind()
        # theta, _ = self.state[0][0], self.state[0][1]
        if state is None:
            state = self.state
        x = state[..., :1]
        theta = state[..., 1:2]
        desired_theta1 = torch.tensor([0.0], **self.kwargs)
        desired_theta2 = torch.tensor([2*np.pi], **self.kwargs) 
        delta_theta = torch.minimum(torch.abs(theta - desired_theta1), torch.abs(theta - desired_theta2))
        rw = delta_theta + (torch.abs(x)) + (torch.abs(x) > 10).float() * 80
        return -rw

    def reward(self, state, action):
        x = state[..., :1]
        theta = state[..., 1:2]
        desired_theta1 = 0.0
        desired_theta2 = 2*np.pi
        delta_theta = np.minimum(np.abs(theta - desired_theta1), np.abs(theta - desired_theta2))
        rw = delta_theta + (np.abs(x)) + (np.abs(x) > 10) * 80
        return -rw

    def close(self):
        """
        Closes the environment.
        """
        pass

class Cartpole2linkEnv(CartpoleEnv):
    def __init__(self, dt=0.05, stabilization=False, kwargs=None):
        super().__init__(nx=6, dt=dt, stabilization=stabilization, kwargs=kwargs)
    
    def reset(self, bsz=None):
        """
        Resets the environment to an initial state, which is a random angle and angular velocity.
        Returns:
            numpy.ndarray: The initial state.
        """
        if bsz is None:
            bsz = self.bsz
        if self.stabilization:
            high = np.concatenate(
                (np.full(self.nq, 0.05), np.full(self.nq, 0.05)))
            high[0], high[1] = 0.1, 0.1
            high = high[None].repeat(bsz, 0)
            self.state = (
                torch.tensor(
                    np.random.uniform(low=-high, high=high), **self.kwargs
                )
            )
        else:
            high = np.concatenate(
                (np.full(self.nq, np.pi), np.full(self.nq, 0.05))
            )
            high = high[None].repeat(bsz, 0)
            self.state = self.state_clip(torch.tensor(
                np.random.uniform(low=-high, high=high), **self.kwargs
            ))

        self.state = torch.tensor(
                [0.0, np.pi / 2 + 0.01, 0.0, 0.0, 0.0, 0.0], **self.kwargs)[None].repeat(bsz, 1)
        self.num_successes = 0
        self.num_steps = 0
        return self.state

    def done(self, state):
        x = state[..., 0]
        theta = self.state[..., 1:3]
        desired_theta1 = torch.tensor([0, 0], **self.kwargs)[None]
        desired_theta2 = torch.tensor([2*np.pi]*2, **self.kwargs)[None]
        desired_theta3 = torch.tensor([0, 2*np.pi], **self.kwargs)[None]
        desired_theta4 = torch.tensor([2*np.pi, 0], **self.kwargs)[None]
        # delta_theta = torch.minimum((theta - desired_theta1).norm(dim=-1), (theta - desired_theta2).norm(dim=-1))
        delta_theta = torch.minimum(torch.minimum((theta - desired_theta1).norm(dim=-1), (theta - desired_theta2).norm(dim=-1)), torch.minimum((theta - desired_theta3).norm(dim=-1), (theta - desired_theta4).norm(dim=-1)))
        # success = torch.logical_and(torch.norm(
        #     theta - desired_theta) < 0.05 , (torch.abs(x) < 0.05))
        success = delta_theta < 0.15
        # self.num_successes = 0 if not success else self.num_successes + 1
        self.num_successes = (self.num_successes + 1)*success.item()
        # ipdb.set_trace()
        # if (self.num_successes >= 15).any():
        #     print(f'{(self.num_successes >= 15).sum().item()} successes')
        # return torch.logical_or(torch.abs(x) > 20, self.num_steps >= self.T)
        return self.num_successes >= 10 or self.num_steps >= self.T

    def get_reward(self, action):
        state = self.state
        x = state[..., 0]
        theta = state[..., 1:3]
        desired_theta1 = torch.tensor([0, 0], **self.kwargs)[None]
        desired_theta2 = torch.tensor([2*np.pi]*2, **self.kwargs)[None]
        desired_theta3 = torch.tensor([0, 2*np.pi], **self.kwargs)[None]
        desired_theta4 = torch.tensor([2*np.pi, 0], **self.kwargs)[None]
        # delta_theta = torch.minimum((theta - desired_theta1).norm(dim=-1), (theta - desired_theta2).norm(dim=-1))
        delta_theta = torch.minimum(torch.minimum((theta - desired_theta1).abs().sum(dim=-1), (theta - desired_theta2).abs().sum(dim=-1)), torch.minimum((theta - desired_theta3).abs().sum(dim=-1), (theta - desired_theta4).abs().sum(dim=-1)))
        rw = delta_theta + (torch.abs(x))
        return -rw

# if this is main then run the test
if __name__ == "__main__":
    # create the dynamics model

    kwargs = {
        "dtype": torch.float64,
        "device": torch.device("cuda"),
        "requires_grad": False,
    }
    nq = 3
    nx = nq * 2
    dt = 0.03
    dynamics = CartpoleDynamics(nx=nx, dt=dt, kwargs=kwargs)

    # create some random states and actions
    bsz = 1
    # state = torch.randn((bsz, nx), **kwargs)
    # action = torch.randn((bsz, 1), **kwargs)

    state = torch.tensor([[0.0, np.pi, 0, 0, 0, 0.0]], **kwargs)
    # state = torch.tensor([[0.0, 0.0, 0.0, 0.0]], **kwargs)
    # state = torch.tensor([[0.5, 0.5, 2.2, 1.0]], **kwargs)
    action = torch.tensor([[100.00000000012201]], **kwargs)

    # next_state = dynamics(state, action)
    # # jacobians = dynamics.derivatives(state, action)
    # jacobians_fd = dynamics.finite_diff_derivatives(state, action, eps=1e-5)
    # next_state, jacobians = dynamics.dynamics_derivatives(state, action)

    # print("next_state:", next_state)
    # print("jacobians[0]:", jacobians[0])
    # print("jacobians[1]:", jacobians[1])
    # print("jacobians_fd[0]:", jacobians_fd[0])
    # print("jacobians_fd[1]:", jacobians_fd[1])

    # # calculate the error between jacobians and jacobians_fd
    # error = np.zeros(2)
    # for i in range(len(jacobians)):
    #     error[i] = torch.norm(jacobians[i] - jacobians_fd[i]
    #                           ) / torch.norm(jacobians[i])
    # print("error:", error)

    # create the environment
    # env = CartpoleEnv(nx=nx, dt=dt, stabilization=False, kwargs=kwargs)
    # env.state = state
    # next_state2 = env.step(to_numpy(action))
    # print("next_state:", next_state2)

    #############################
    # Test vmap
    #############################
    # ls = 2
    # T = 3
    # bsz = 4
    # state = torch.randn((ls, bsz, T, nx), **kwargs)
    # action = torch.randn((ls, bsz, T, 1), **kwargs)
    # dx = dynamics

    # def merit(x, u): 
    #     x_next = dx(x.reshape(-1, nx), u.reshape(-1, 1)).view(bsz, T, nx)
    #     return x_next

    # # print("state:", merit(state[0], action[0]).shape)
    # my_vmap = torch.vmap(merit)
    # next_state = my_vmap(state, action)
    # print("next_state:", next_state)
    # ipdb.set_trace()

    #############################
    # Test saved trajectory
    #############################
    # traj = np.load('traj.npz')
    # x_traj = traj["X_np"]
    # u_traj = traj["U_np"]
    # x_traj = torch.tensor(x_traj, **kwargs)
    # u_traj = torch.tensor(u_traj, **kwargs)
    # ipdb.set_trace()
    # x_out = dynamics(x_traj[:-1], u_traj.reshape(-1,1))
    
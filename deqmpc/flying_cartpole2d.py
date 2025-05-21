import numpy as np
import torch
from rexquad_utils import rk4, deg2rad, Spaces, Spaces_np, w2pdotkinematics_mrp, quat2mrp, euler_to_quaternion, mrp2quat, quatrot, mrp2rot
from torch.func import hessian, vmap, jacrev
import ipdb
import time

def angle_normalize_2pi(x):
    return (((x) % (2*np.pi)))

class FlyingCartpole_dynamics(torch.nn.Module):
    # think about mrp vs quat for various things - play with Q cost 
    def __init__(self, bsz=1,  mass_q=2.0, mass_p=0.2, J=[[0.01566089, 0.00000318037, 0.0],[0.00000318037, 0.01562078, 0.0], [0.0, 0.0, 0.02226868]], L=0.5, gravity=[0,0,-9.81], motor_dist=0.28, kf=0.025, bf=-30.48576, km=0.00029958, bm=-0.367697, quad_min_throttle = 1148.0, quad_max_throttle = 1832.0, ned=False, cross_A_x=0.25, cross_A_y=0.25, cross_A_z=0.5, cd=[0.0, 0.0, 0.0], max_steps=100, dt=0.05, device=torch.device('cpu'), jacobian=False):
        super(FlyingCartpole_dynamics, self).__init__()
        self.m = mass_q + mass_p
        self.mq = mass_q
        self.mp = mass_p
        J = np.array(J)
        if len(J.shape) == 1:
            self.J = torch.diag(torch.FloatTensor(J)).unsqueeze(0).to(device)
        else:
            self.J = torch.FloatTensor(J).unsqueeze(0).to(device)
        self.Jinv = torch.linalg.inv(self.J).to(device)
        self.g = torch.FloatTensor(gravity).to(device).unsqueeze(0)
        self.motor_dist = motor_dist
        self.kf = kf
        self.km = km
        self.bf = bf
        self.bm = bm
        self.L = torch.tensor(L)
        self.bsz = bsz
        self.quad_min_throttle = quad_min_throttle
        self.quad_max_throttle = quad_max_throttle
        self.ned = ned
        self.nx = self.state_dim = 3 + 3*3 + 2
        self.nu = self.control_dim = 4
        self._max_episode_steps = max_steps
        self.bsz = bsz
        self.dt = dt
        self.act_scale = 10.0
        self.u_hover = torch.tensor([(-self.m*gravity[2])/self.act_scale/self.kf/4]*4).to(device)
        self.cd = torch.tensor(cd).unsqueeze(0).to(device)
        self.ss = torch.tensor([[1.,1,0], [1.,-1,0], [-1.,-1,0], [-1.,1,0]]).to(device).unsqueeze(0)
        self.ss = self.ss/self.ss.norm(dim=-1).unsqueeze(-1)

        self.device = device
        self.jacobian = jacobian
        self.identity = torch.eye(self.state_dim).to(device)

    def forces(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        m = x[..., 3:6]
        q = mrp2quat(-m)
        F = torch.sum(self.kf*u, dim=-1)
        # g = torch.tensor([0,0,-9.81]).to(x)#self.g
        F = torch.stack([torch.zeros_like(F), torch.zeros_like(F), F], dim=-1)
        f = F + quatrot(q, self.m * self.g)
        return f

    def moments(self, x, u):
        L = self.motor_dist        
        zeros = torch.zeros_like(u)
        F = self.kf*u#torch.maximum(self.kf*u, zeros)
        M = self.km*u
        tau1 = zeros[...,0]
        tau2 = zeros[...,0]
        tau3 = M[...,0]-M[...,1]+M[...,2]-M[...,3]
        torque = torch.stack([tau1, tau2, tau3], dim=-1)
        ss = self.ss
        if len(x.shape) == 3:
            ss = ss.unsqueeze(0)
        if ss.dtype != x.dtype:
            ss = ss.to(x.dtype)
        torque += torch.cross(self.motor_dist * ss, torch.stack([zeros, zeros, self.kf * u], dim=-1), dim=-1).sum(dim=-2)
        return torque

    def wrenches(self, x, u):
        F = self.forces(x, u)
        M = self.moments(x, u)
        return F, M

    def rk4_dynamics(self, x, u):
        # x, u = x.unsqueeze(0), u.unsqueeze(0)
        dt = self.dt
        dt2 = dt / 2.0
        y0 = x
        k1 = self.dynamics_(y0, u)
        k2 = self.dynamics_(y0 + dt2 * k1, u)
        k3 = self.dynamics_(y0 + dt2 * k2, u)
        k4 = self.dynamics_(y0 + dt * k3, u)
        yout = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return yout
    
    def forward(self, x, u):
        return self.rk4_dynamics(x, u)
        
    def get_quad_state(self, x):
        r = x[..., :3]
        m = x[..., 3:6] # mrp2quat?
        v = x[..., 7:10]
        w = x[..., 10:13]
        return r, m, v, w, torch.cat([r, m, v, w], dim=-1)

    def get_pend_state(self, x):
        theta = x[..., 6:7]  # pendulum angle
        theta_dot = x[..., 13:14]
        return theta, theta_dot

    def dynamics_(self, x, u):
        """
        state: [r, m, theta, v, w, theta_dot]
        control: [u1, u2, u3, u4]
        """
        u = self.act_scale * (u + self.u_hover)
        p, m, v, w, xq = self.get_quad_state(x) # possibly do mrp2quat
        q = mrp2quat(m)
        F, tau = self.wrenches(xq, u)
        mdot = w2pdotkinematics_mrp(m, w)
        pdot = quatrot(q, v)
        vdot = F/self.m - torch.cross(w, v, dim=-1)
        if len(w.shape) == 3:
            Jinv = self.Jinv.unsqueeze(0)
            J = self.J.unsqueeze(0)
        else:
            Jinv = self.Jinv
            J = self.J
        wdot = (Jinv*(tau - torch.cross(w, (J*(w.unsqueeze(-2))).sum(dim=-1), dim=-1)).unsqueeze(-2)).sum(dim=-1)

        # add the inverted pendulum
        theta, theta_dot = self.get_pend_state(x)
        x_ddot = quatrot(q, vdot)[...,0:1]
        theta_ddot = (self.g[0][2] * torch.sin(theta) + x_ddot*torch.cos(theta))/self.L
        # return full state
        return torch.cat([pdot, mdot, theta_dot, vdot, wdot, theta_ddot], dim=-1)

class FlyingCartpole_dynamics_jac(FlyingCartpole_dynamics):
    def __init__(self, bsz=1,  mass_q=2.0, mass_p=0.2, J=[[0.01566089, 0.00000318037, 0.0],[0.00000318037, 0.01562078, 0.0], [0.0, 0.0, 0.02226868]], L=0.5, gravity=[0,0,-9.81], motor_dist=0.28, kf=0.025, bf=-30.48576, km=0.00029958, bm=-0.367697, quad_min_throttle = 1148.0, quad_max_throttle = 1832.0, ned=False, cross_A_x=0.25, cross_A_y=0.25, cross_A_z=0.5, cd=[0.0, 0.0, 0.0], max_steps=100, dt=0.05, device=torch.device('cpu')):
        super(FlyingCartpole_dynamics_jac, self).__init__(bsz, mass_q, mass_p, J, L, gravity, motor_dist, kf, bf, km, bm, quad_min_throttle, quad_max_throttle, ned, cross_A_x, cross_A_y, cross_A_z, cd, max_steps, dt, device, True)
    
    def forward(self, x, u):
        ## use vmap to compute jacobian using autograd.grad
        x = x.unsqueeze(-2).repeat(1, self.state_dim, 1)
        u = u.unsqueeze(-2).repeat(1, self.state_dim, 1)
        # ipdb.set_trace()
        out_rk4 = self.rk4_dynamics(x, u)
        out = out_rk4*self.identity[None]
        jac_out = torch.autograd.grad([out.sum()], [x, u])
        
        return out_rk4[:, 0], jac_out

class FlyingCartpole(torch.nn.Module):
    def __init__(self, bsz=1, Qscale=1,  mass_q=2.0, mass_p=0.1, J=[[0.0023, 0.0, 0.0],[0.0, 0.0023, 0.0], [0.0, 0.0, 0.004]], L=0.5, gravity=[0,0,-9.81], motor_dist=0.175, kf=1.0, bf=0.0, km=0.025, bm=-0.367697, quad_min_throttle = 1148.0, quad_max_throttle = 1832.0, ned=False, cross_A_x=0.25, cross_A_y=0.25, cross_A_z=0.5, cd=[0.0, 0.0, 0.0], max_steps=100, dt=0.05, device=torch.device('cpu'), obstacles=False, obstacle_radius=0.25):
        super(FlyingCartpole, self).__init__()
        self.dynamics = FlyingCartpole_dynamics(bsz, mass_q, mass_p, J, L, gravity, motor_dist, kf, bf, km, bm, quad_min_throttle, quad_max_throttle, ned, cross_A_x, cross_A_y, cross_A_z, cd, max_steps, dt, device, False)
        self.dynamics = torch.jit.script(self.dynamics)
        self.dynamics_derivatives = FlyingCartpole_dynamics_jac(bsz, mass_q, mass_p, J, L, gravity, motor_dist, kf, bf, km, bm, quad_min_throttle, quad_max_throttle, ned, cross_A_x, cross_A_y, cross_A_z, cd, max_steps, dt, device)
        self.dynamics_derivatives = torch.jit.script(self.dynamics_derivatives)
        self.bsz = bsz
        self.nx = self.state_dim = self.dynamics.state_dim
        self.nu = self.control_dim = self.dynamics.control_dim
        self.nq = 7
        self._max_episode_steps = max_steps
        self.num_steps = torch.zeros(bsz).to(device)
        # self.x = self.reset()
        self.device = device
        self.dt = dt
        self.Qlqr = torch.tensor([10.0]*3 + [10.0]*3 + [80.0] + [1*Qscale]*6 + [1*Qscale]).to(device)#.unsqueeze(0)
        # self.Qlqr = torch.tensor([10.0]*3 + [0.01]*3 + [1.0]*3 + [0.01]*3).to(device)#.unsqueeze(0)
        self.Rlqr = torch.tensor([1e-8]*self.control_dim).to(device)#.unsqueeze(0)
        self.observation_space = Spaces_np((self.state_dim,))
        # self.max_torque = 18.3
        # self.action_space = Spaces_np((self.control_dim,), np.array([0.3*self.dynamics.u_hover.cpu()[0]]*self.control_dim), np.array([-0.3*self.dynamics.u_hover.cpu()[0]]*self.control_dim)) #12.0
        self.action_space = Spaces_np((self.control_dim,), np.array([0.3*self.dynamics.u_hover.cpu()[0]]*self.control_dim), np.array([-0.3*self.dynamics.u_hover.cpu()[0]]*self.control_dim)) #12.0
        self.x_window = torch.tensor([5.0,5.0,5.0,deg2rad(45.0),deg2rad(45.0),deg2rad(45.0),np.pi,1.0,1.0,1.0,1.0,1.0,1.0,1.0]).to(device)
        self.targ_pos = torch.zeros(self.state_dim).to(self.device)
        self.targ_pos[6] = np.pi # upright pendulum
        # self.goal = torch.tensor([ 7.4720e-02, -1.3457e-01,  2.4619e-01,  5.9315e-05,  8.8729e-05,
        #  9.2153e-02,  3.1420e+00,  6.5756e-04,  5.3479e-04,  1.1379e-03,
        # -2.6658e-04, -2.3303e-04,  3.0988e-04, -3.1224e-04])
        self.goal = torch.tensor([-4.8198e-01, -1.7375e-02,  2.6890e-01, -7.3554e-05,  2.2614e-04,
        -2.1837e-01,  3.1425e+00, -2.9245e-02,  1.3690e-02, -2.9136e-03,
         6.3533e-04,  1.2356e-03,  2.3545e-03,  2.1940e-04])
        self.obstacles = obstacles
        self.obstacle_positions = torch.tensor([[-0.81024811,  2.47097938, -1.46527507],
                                                [ 4.25182361,  2.61438853, -3.99336202],
                                                [-1.88037898,  0.56471661, -4.89251658],
                                                [-0.89694315, -1.17227775, -2.31071003],
                                                [ 0.31117451,  1.80695677, -1.32360805],
                                                [-0.83515347, -3.00524971, -0.37880997],
                                                [ 0.42646518, -2.55505666, -1.4580139 ],
                                                [-3.06794499, -3.75663134, -2.68146399],
                                                [-2.47485677, -3.2241943 ,  4.05956764],
                                                [ 4.90234242, -0.44360142,  3.00769067],
                                                [-0.4466156 ,  2.82236085, -3.70498888],
                                                [-3.30175667, -0.37915783,  3.76276084],
                                                [-1.55007116,  2.10577068,  1.29878497],
                                                [-3.18635985,  1.77041154,  2.70392382],
                                                [ 4.59851198,  1.34127491, -3.2508418 ],
                                                [-1.37583745, -2.16044885,  2.95391448],
                                                [-0.86607566,  1.90383346, -2.29595208],
                                                [-3.90222124,  3.93796524, -4.8222375 ],
                                                [ 1.83414656, -4.90150613, -3.97004479],
                                                [ 2.10004524,  4.32631374, -1.7743607 ],
                                                [-0.73089303,  0.71350811,  4.39234036],
                                                [-0.57377963, -1.27260017, -4.02256416],
                                                [ 4.17079007,  1.09192127,  3.44064471],
                                                [ 0.12940132,  2.81247143,  0.63458352],
                                                [ 0.64863905,  3.19743266,  2.347993  ],
                                                [-0.23519056, -3.91345489, -1.63826518],
                                                [ 3.18776744, -3.65825113,  3.33950477],
                                                [ 4.04449088,  1.13759982, -2.09952288],
                                                [ 1.34288453,  4.09059731,  4.26551507],
                                                [ 2.60116671,  3.27868729, -2.24891686],
                                                [ 2.45285101, -1.71007925,  3.67593524],
                                                [ 0.75708598,  0.37944072,  1.00573208],
                                                [-1.8836751 ,  3.45417808, -3.70553914],
                                                [-2.48427958, -2.98336577, -3.69115236],
                                                [ 2.53442509,  1.69021664,  4.56683953],
                                                [-3.40910921, -1.15168358, -3.11048661],
                                                [-1.85093449, -3.27903642,  4.40783757],
                                                [ 1.11234478, -2.84715317,  1.92532219],
                                                [-0.23470463, -4.51888895,  0.77568075],
                                                [-1.32348548,  1.48201938,  1.9124648 ]]).to(self.device)
        self.obstacle_radius = obstacle_radius
        if not obstacles:
            self.spec_id = "FlyingCartpole-v0"
            # self.saved_ckpt_name = "cgac_checkpoint_FlyingCartpole_swingup400maxu21_rew801_initrew1finrew5_0alphapk75_thres1000_massp04_fixmaskmem_best"
            self.saved_ckpt_name = "cgac_checkpoint_FlyingCartpole_swingup300_ub0.3x0.3x_L0.5_ent7.5_memfix_seed7_ok"

        ################################################ Obstacle avoidance ################################################
        
        elif obstacles:
            if obstacle_radius == 0.2:
                self.saved_ckpt_name = "cgac_checkpoint_FlyingCartpole_swingup200_ub03_rew808_initrew1finrew5_0alphapk75_obsr02_seed2"
                self.spec_id = "FlyingCartpole-v1-obsr02"
            elif obstacle_radius == 0.4:
                self.saved_ckpt_name = "cgac_checkpoint_FlyingCartpole_swingup200_ub03_rew808_initrew1finrew5_0alphapk75_obsr04"
                self.spec_id = "FlyingCartpole-v1-obsr04"
            elif obstacle_radius == 0.5:
                self.saved_ckpt_name = "cgac_checkpoint_FlyingCartpole_swingup200_ub03_rew808_initrew1finrew5_0alphapk75_obsr05"
                self.spec_id = "FlyingCartpole-v1-obsr05"
        
    def forward(self, x, u, jacobian=False):
        if jacobian:
            return self.dynamics_derivatives(x, u)
        else:
            return self.dynamics(x, u)

    def action_clip(self, action):
        return torch.clamp(action, torch.Tensor(self.action_space.low).to(self.device), torch.Tensor(self.action_space.high).to(self.device))

    def state_clip(self, state):
        state[..., 6] = angle_normalize_2pi(state[..., 6])
        return state
    
    def step(self, u):
        self.num_steps += 1
        u = self.action_clip(u)  # clip the action
        done_inf = torch.zeros(self.bsz).to(self.device, dtype=torch.bool)
        if u.dtype == np.float64 or u.dtype == np.float32:
            u = torch.tensor(u).to(self.x)
            if len(u.shape)==1:
                u = u.unsqueeze(0)
                self.x = self.dynamics(self.x, u)
                self.x = self.state_clip(self.x) # clip the state
                reward = self.reward(self.x, u).cpu().numpy().squeeze()
                if torch.isnan(self.x).sum() or torch.isinf(self.x).sum() or np.isinf(reward) or np.isnan(reward) or reward < -1000:
                    x = self.reset()
                    done_inf = True
                    reward = 0
                x_out = self.x.squeeze().detach().cpu().numpy()
            else:
                self.x = self.dynamics(self.x, u)
                self.x = self.state_clip(self.x) # clip the state
                reward = self.reward(self.x, u).cpu().numpy()
                if torch.isnan(self.x).sum() or torch.isinf(self.x).sum() or np.isinf(reward.sum()) or np.isnan(reward.sum()) or reward.sum() < -1000:
                    x = self.reset()
                    done_inf = True
                    reward = np.array([0.0])
                x_out = self.x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            self.x = self.dynamics(self.x, u)
            self.x = self.state_clip(self.x) # clip the state
            reward = self.reward(self.x, u)
            collision_mask = self.check_collisions(self.x)
            ifcond = torch.logical_or(torch.isnan(self.x).any(dim=-1), torch.logical_or(torch.isinf(self.x).any(dim=-1), torch.logical_or(torch.isinf(reward), torch.logical_or(torch.isnan(reward), (reward < -1000)))))
            ifcond = torch.logical_or(ifcond, collision_mask)
            if ifcond.any():
                self.reset(torch.where(ifcond))
                done_inf[ifcond] = 1
                reward[ifcond] = 0.0
            x_out = self.x
        done = torch.logical_or(self.num_steps >= self._max_episode_steps, done_inf)
        return x_out, reward, done, {'done_inf':done_inf}

    def stepx(self, x, u):
        done_inf = False
        u = self.action_clip(u)  # clip the action
        if u.dtype == np.float64 or u.dtype == np.float32:
            u = torch.tensor(u).to(self.x)
            x = torch.tensor(x).to(self.x)
            if len(u.shape)==1:
                u = u.unsqueeze(0)
                x = x.unsqueeze(0)
                x = self.dynamics(x, u)
                self.x = self.state_clip(self.x) # clip the state
                reward = self.reward(x, u).cpu().numpy().squeeze()
                if torch.isnan(x).sum() or torch.isinf(x).sum():
                    self.reset()
                    done_inf = True
                    reward = -1000
                    x = self.x
                x_out = x.squeeze().detach().cpu().numpy()
            else:
                x = self.dynamics(x, u)
                self.x = self.state_clip(self.x) # clip the state
                reward = self.reward(x, u).cpu().numpy()
                if torch.isnan(x).sum() or torch.isinf(x).sum():
                    self.reset()
                    done_inf = True
                    reward = np.array([-1000])
                    x = self.x
                x_out = x.detach().cpu().numpy()
        elif u.dtype == torch.float32 or u.dtype==torch.float64:
            self.x = self.state_clip(self.x) # clip the state
            reward = self.reward(x, u)
            if torch.isnan(x).sum() or torch.isinf(x).sum():
                self.reset()
                done_inf = True
                reward = torch.tensor([-1000]).to(self.x)
                x = self.x
            x_out = x
        done = self.num_steps >= self._max_episode_steps or done_inf
        return x_out, reward, done, {'done_inf':done_inf}

    def check_collisions(self, x):
        # check if the quadrotor is colliding with any obstacles
        if not self.obstacles:
            return torch.zeros_like(x[..., 0], dtype=torch.bool)
        r = x[..., :3]
        num_dims = len(r.shape) - 1
        obstacles = self.obstacle_positions
        for i in range(num_dims):
            obstacles = obstacles.unsqueeze(0).repeat(r.shape[num_dims-1-i], 1, 1)
        dist = torch.norm(r.unsqueeze(-2) - obstacles, dim=-1)
        return torch.any(dist < self.obstacle_radius, dim=-1)

    def reward(self, x, u):
        cost = (((x - self.targ_pos.cpu())**2)*self.Qlqr.cpu()/2).sum(dim=-1)/100 + ((u**2)*self.Rlqr.cpu()/2).sum(dim=-1)/10
        mask = (cost > 1000)
        rew = torch.exp(-cost/2+2)
        return rew

    def reset_torch(self, reset_ids=None, bsz=None, x_window=None):
        if bsz is None and reset_ids is None:
            bsz = self.bsz
            self.num_steps[:] = 0
        elif bsz is None:
            bsz = len(reset_ids)
        if x_window is None:
            x_window = self.x_window
        elif len(x_window.shape) == 1:
            x_window = x_window.unsqueeze(0)
        x = (torch.rand((bsz, self.state_dim))*2-1).to(self.x_window)*self.x_window 
        x = torch.cat([x[:,:3], quat2mrp(euler_to_quaternion(x[:, 3:6])), np.pi+x[:,6:7], x[:, 7:]], dim=-1) #quat2mrp
        if reset_ids is not None:
            self.x[reset_ids] = x
        else:
            self.x = x
        return self.x

    def reset(self, reset_ids=None, bsz=None, x_window=None):
        x = self.reset_torch(reset_ids, bsz, x_window)
        collision_mask = self.check_collisions(x)
        if collision_mask.any():
            x = self.reset(torch.where(collision_mask), None, x_window)
        return x
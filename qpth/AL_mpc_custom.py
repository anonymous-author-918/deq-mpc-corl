import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.func import hessian, vmap, jacrev

import numpy as np
import numpy.random as npr

from enum import Enum

import sys, time
from qpth.AL_mpc import MPC

from . import qp
import ipdb
from . import al_utils
from . import al_utils_se
from . import al_utils_lin
from . import util

class Obstacle_MPC(MPC):
    def __init__(
            self, n_state, n_ctrl, T,
            u_lower=None, u_upper=None,
            u_init=None,
            x_init=None,
            al_iter=2,
            verbose=0,
            eps=1e-7,
            back_eps=1e-7,
            n_batch=None,
            linesearch_decay=0.2,
            max_linesearch_iter=10,
            exit_unconverged=True,
            detach_unconverged=True,
            backprop=True,
            slew_rate_penalty=None,
            solver_type='dense',
            add_goal_constraint=False,
            x_goal=None,
            diag_cost=True, 
            ineqG=None,
            ineqh=None,
            state_estimator=False,
            dtype=torch.float64,
            env = None,
    ):
        super().__init__(n_state, n_ctrl, T, u_lower, u_upper, u_init, x_init, al_iter, verbose, eps, back_eps,
        n_batch, linesearch_decay, max_linesearch_iter, exit_unconverged, detach_unconverged, backprop, 
        slew_rate_penalty, solver_type, add_goal_constraint, x_goal, diag_cost, ineqG, ineqh, state_estimator, dtype)
        self.n_obstacles = 40
        self.n_obstacle_constraints = 4
        self.nineq += self.n_obstacle_constraints*T
        self.dyn_res_prev = 1000000
        self.mask = torch.ones(self.n_batch, self.T, 1).to(self.u_upper)
        self.obstacle_radius = 0.2 if env is None else env.obstacle_radius
        self.obstacle_radius = torch.tensor(self.obstacle_radius).to(self.u_upper)
        self.obstacle_positions = None if env is None else env.obstacle_positions
        self.lamda_prev = torch.zeros(self.n_batch, self.neq+self.nineq).to(self.u_upper)

    
    def merit_function(self, xu, Q, q, dx, x0, lamda, rho, grad=False, linearize_once=False):
        if self.state_estimator:
            return al_utils_se.merit_function(xu, Q, q, dx, x0, lamda, rho, self.x_lower, self.x_upper, self.diag_cost)
        elif linearize_once:
            return al_utils_lin.merit_function(xu, Q, q, dx, x0, lamda, rho, self.x_lower, self.x_upper, self.u_lower, self.u_upper, self.diag_cost)
        else:
            return al_utils.merit_function(xu, Q, q, dx, x0, lamda, rho, self.x_lower, self.x_upper, self.u_lower, self.u_upper, self.diag_cost, self.obstacles)
    def merit_hessian(self, xu, Q, q, dx, dx_jac, x0, lamda, rho, linearize_once=False):
        if self.state_estimator:
            return al_utils_se.merit_hessian(xu, Q, q, dx, dx_jac, x0, lamda, rho, self.x_lower, self.x_upper, self.diag_cost)
        elif linearize_once:
            return al_utils_lin.merit_hessian(xu, Q, q, dx, dx_jac, x0, lamda, rho, self.x_lower, self.x_upper, self.u_lower, self.u_upper, self.diag_cost)
        else:
            return al_utils.merit_hessian(xu, Q, q, dx, dx_jac, x0, lamda, rho, self.x_lower, self.x_upper, self.u_lower, self.u_upper, self.diag_cost, self.obstacles)

    def merit_grad_hess(self, xu, Q, q, dx, dx_jac, x0, lamda, rho, linearize_once=False):
        if self.state_estimator:
            return al_utils_se.merit_grad_hessian(xu, Q, q, dx, dx_jac, x0, lamda, rho, self.x_lower, self.x_upper, self.diag_cost)
        elif linearize_once:
            return al_utils_lin.merit_grad_hessian(xu, Q, q, dx, dx_jac, x0, lamda, rho, self.x_lower, self.x_upper, self.u_lower, self.u_upper, self.diag_cost)
        else:
            return al_utils.merit_grad_hessian(xu, Q, q, dx, dx_jac, x0, lamda, rho, self.x_lower, self.x_upper, self.u_lower, self.u_upper, self.diag_cost, self.obstacles)
        
    def dyn_res(self, xu, dx, x0, res_type='clamp', linearize_once=False):
        if self.state_estimator:
            res, res_clamp = al_utils_se.dyn_res(xu, dx, x0, self.x_lower, self.x_upper)
        elif linearize_once:
            res, res_clamp = al_utils_lin.dyn_res(xu, dx, x0, self.x_lower, self.x_upper, self.u_lower, self.u_upper)
        else:
            res, res_clamp = al_utils.dyn_res(xu, dx, x0, self.x_lower, self.x_upper, self.u_lower, self.u_upper, self.obstacles)
        if res_type == 'noclamp':
            return res
        elif res_type == 'clamp':
            return res_clamp
        else:
            return res, res_clamp

    def reinitialize(self, x, mask):
        self.u_init = None
        self.x_init = None
        self.n_batch = x.size(0)
        self.rho_prev = torch.ones((self.n_batch,1), device=x.device, dtype=x.dtype)
        self.lamda_prev = torch.zeros(self.n_batch, self.neq+self.nineq, device=x.device, dtype=x.dtype)
        self.cost_hist_stream = [[], []]
        self.dyn_res_prev = 1000000
        self.just_initialized = True
        self.warm_starting = False
        self.mask = mask
        # for each x in x(bsz,T,xdim), find the self.n_obstacle_constraints closest obstacles
        obstacle_dists = self.obstacle_positions.unsqueeze(0).unsqueeze(0).expand(self.n_batch, self.T, -1, -1) - x[:, :, :3].unsqueeze(2).expand(-1, -1, self.n_obstacles, -1)
        obstacle_ids = torch.argsort(obstacle_dists.norm(dim=-1), dim=-1)[..., :self.n_obstacle_constraints]
        obstacle_positions = self.obstacle_positions[obstacle_ids]
        self.obstacles = (obstacle_positions, self.obstacle_radius)

    
    def warm_start_initialize(self, x, u, args):
        self.u_init = u
        self.x_init = x
        bsz = x.size(0)
        lamda_eq = self.lamda_prev[:, :self.neq].reshape(self.n_batch, self.T, -1)
        lamda_ineq = self.lamda_prev[:, self.neq:].reshape(self.n_batch, self.T, -1)
        lamda_eq = torch.cat([lamda_eq[:,1:-1], lamda_eq[:, -2:]*0], dim=1).reshape(self.n_batch, -1)
        lamda_ineq = torch.cat([lamda_ineq[:,1:], lamda_ineq[:, -1:]*0], dim=1).reshape(self.n_batch, -1)
        self.lamda_prev = torch.cat([lamda_eq, lamda_ineq], dim=1)*0
        self.rho_prev = torch.clamp(self.rho_prev, max=args.rho_init_max)
        self.just_initialized = True
        self.warm_starting = True
        obstacle_positions = self.obstacles[0][:, 1:]
        obstacle_dists = self.obstacle_positions.unsqueeze(0).unsqueeze(0).expand(self.n_batch, 1, -1, -1) - x[:, -1:, :3].unsqueeze(2).expand(-1, -1, self.n_obstacles, -1)
        obstacle_ids = torch.argsort(obstacle_dists.norm(dim=-1), dim=-1)[..., :self.n_obstacle_constraints]
        obstacle_positions_last = self.obstacle_positions[obstacle_ids]
        obstacle_positions = torch.cat([obstacle_positions, obstacle_positions_last], dim=1)
        self.obstacles = (obstacle_positions, self.obstacle_radius)
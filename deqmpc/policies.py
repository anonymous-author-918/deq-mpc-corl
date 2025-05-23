import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import qpth.qp_wrapper as ip_mpc
import qpth.AL_mpc as al_mpc
from qpth.AL_mpc_custom import Obstacle_MPC
import qpth.al_utils as al_utils
import ipdb
# from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
# from policy_utils import SinusoidalPosEmb
import time
from deq_layer import *


# POSSIBLE OUTPUT TYPES OF THE POLICY
# 0: horizon action
# 1: horizon state
# 2: horizon state + action
# 3: horizon config (state no vel)



# Questions:
# 1. What is a good input : As in a 'good' qualitative estimate of the error.
# - error between the trajectory spit out and the optimized trajectory
# - just x - x0
# - maybe also give velocities as input?
# - maybe also give the previous xref or (xref - x0) prediction as input? - especially important if we want to correct xref instead of just x (which is probably useful when there are other regularizers in the cost)
# - also should be compatible with recurrence on the latent z
# - ideally would like to merge it with the features or state (perhaps 3D) we extract from the image - to get a better sense of the error as input
# 2. What type of recurrence to add? beyond just the fixed point iterate.
# 3. How many QP solves to perform?
# 4. How to do the backward pass? - implicit differentiation of the fp network's fixed point? or just regular backprop? don't have any jacobians to compute fixed points, could compute the jacobians though
# 5. Unclear if deltas naively are the best output - regression is a hard problem especially at that precision.
# 6. Architecture - need to probably do some sort of GNN type thing. Using FFN to crunch entire trajectory seems suboptimal.
# 7. Also need to spit out weights corresponding to the steps the network is not confident about.
# 8. Diffusion gives a nice way to represent course signals when the network isn't confident about the output. To prevent overpenalizing the network for being uncertain.
#    Question is how to do that without explicitly introducing diffusion/stochasticity into the network.
#    (i) Perhaps just use the weights as a way to do that? This would imply we have to use the weights on the output loss as well.
#    (ii) This is however a broadly interesting question - If the weights or other mechanisms can be used to represent uncertainty well, then maybe diffusion is redundant?
#           - Question is what are those other mechanisms?
#           - CVAE type thing represents it in the latent space? (for the network to represent uncertainty aleotorically) => khai: CVAE handles epistemic uncertainty
#           - Dropout type thing represents it in the network parameters? (for the network to represent uncertainty epistemically) -- Similar to ensembles
#           - Predicting explicit weights represents it for the optimization problem? (for the optimization problem to represent uncertainty aleotorically)
#           - Using the weights in the loss as well - does something similar to CVAE? - but CVAE is probably more robust to overfitting and more principled
#           - Using the weights however is more interpretable and probably nicer optimization wise.
#           - Plus, the weights also aid the decoder training more explicitly - CVAE needs to be implicit thanks to the averaging over samples. - robustness v/s precision
#           - But come to think of it, CVAE and diffusion probably have quite a bit in common and there should be some obvious middle ground if we think harder.
#                 - In fact, feels like DEQ + CVAE is the natural way to go! - CVAE is a natural way to represent uncertainty in the latent space and DEQ is a natural way to
#    (iii) Diffusion v/s CVAE
#           - Diffusion is iterative and CVAE is not - makes diffusion representationally more powerful
#           - Diffusion reasons about uncertainy/noise more explicitly - CVAE is more implicit : Both have pros and cons
#                 - The explicitness makes each iteration more stable but also less powerful. Hence diffusion models are also huge in size and slow to train.
#           - The decoder objective in CVAE doesn't account for the uncertainty - diffusion does. This can sometimes lead to the CVAE decoder confidently spitting out blurry outputs.
#    (iv) Diffusion as a way to represent uncertainty in the optimization problem - can we use the normalized error in the estimates through the iterations as a way to represent uncertainty?
#    (v) In the diffusion case, should we just look at the overall setup as a way to add dynamics within diffusion models?
#           - or should we still look at it as an optimization problem with stochasticity added to aid with exploration.
#           - or instead the noise as just a means of incorporating uncertainty in the optimization steps at each iteration.
#           - The specific philosophy maybe doesn't matter too much - but it will probably guide the thinking. Some clarity on this would be nice!
# More important questions are probably still 1, 5, 6, 7
# TODOs:
# 1. Make input flexible - x, xref, xref - x0, xref - xref_prev, xref - xref_prev - x0, xref - xref_prev - x0 + v, etc.
# 2. Make outputs flexible - deltas, xref, xref - x0, etc. and also weights corresponding to the steps the network is not confident about.
# 3. Make architecture flexible - GNN or FFN or whatever
#       - Note : don't do parameter sharing between the nodes in gnn - the sequential order is important and maybe keeping the parameters somewhat separate is a good idea.
#       - but with limited data - parameter sharing might be a good idea - so maybe some sort of hybrid?
# 4. Make recurrence flexible - fixed point or whatever
# 5. Options for diffing through only fixed point, computing fixed point and other losses.
#       - Only penalize the fixed point but train the network for the rest as well.
#       - Anderson/Broyden based fixed point solves
# 5. Figure out the collocation stuff
#       - Is there a simpler/cleaner way to handle the initial deq iterations? - maybe we should try to satisfy the constraints exactly only if the network outputs are actually close enough to the constraint manifold
# 6. Complexity analysis
# 7. Write other solvers - Autmented lagrangian or ADMM or whatever
# 8. Confidences for each knot point - for the Q cost coefficient. - There should again probably be some TD component to the cost coefficient too.

class DEQMPCPolicy(torch.nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.nq = args.nq
        self.T = args.T
        self.dt = env.dt
        self.bsz = args.bsz
        self.deq_reg = args.deq_reg
        self.device = args.device
        self.deq_iter = args.deq_iter
        # self.model = DEQLayerDelta(args, env)
        self.addmem = args.addmem
        if args.addmem:
            self.model = DEQLayerMem(args, env)
        elif args.deq_type == 'nn':
            self.model = FFDNetwork(args, env)
        else:            
            self.model = DEQLayer(args, env)  #TODO different types
        self.model.to(self.device)
        self.out_type = args.policy_out_type  # output type of policy
        self.loss_type = args.loss_type  # loss type for policy
        self.tracking_mpc = Tracking_MPC(args, env)
        self.mpc_time = []
        self.network_time = []

    def forward(self, x, x_gt, u_gt, mask, out_iter=0, qp_solve=True, lastqp_solve=False, warm_start=False):
        """
        Run the DEQLayer and then run the MPC iteratively in a for loop until convergence
        Args:
            x (tensor 0 x bsz x nx): input state
        Returns:
            trajs (list of tuples): list of tuples of nominal states and actions for each deq iteration
            trajs[k][0] (tensor T x bsz x nx): nominal states for the kth deq iteration
            trajs[k][1] (tensor T x bsz x nu): nominal actions for the kth deq iteration
        """
        if warm_start:
            policy_out = self.deqmpc_iter_warm_start(x, x_gt, u_gt, mask, out_iter, qp_solve, lastqp_solve)
            return policy_out
        # initialize trajectory with current state
        x_ref = torch.cat([x]*self.T, dim=-1).detach().clone()
        x_ref = x_ref.view(-1, self.T, self.nx)
        self.x_init = x_ref
        nominal_actions = torch.zeros((x.shape[0], self.T, self.nu), device=self.device)

        z = self.model.init_z(x.shape[0])

        out_aux_dict = {"z": z, "x": x_ref, 'u': nominal_actions}

        if self.addmem:
            out_aux_dict["mem"] = self.model.init_mem(x.shape[0])
            out_aux_dict["old_mem"] = out_aux_dict["mem"].clone().detach()
    
        # run the DEQ layer for deq_iter iterations
        policy_out = self.deqmpc_iter(x, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve, out_iter)        
        policy_out["init_states"] = x_ref
        return policy_out

    def deqmpc_iter(self, obs, out_aux_dict, x_gt, u_gt, mask, qp_solve=False, lastqp_solve=False, out_iter=0): 
        deq_iter = self.deq_iter   
        opt_from_iter = 0
        deq_stats = {"fwd_steps": [], "fwd_err": [], "jac_loss": []}
        # ipdb.set_trace()
        trajs = []
        scales = []
        status = False
        for i in range(deq_iter):
            in_obs_dict = {"o": obs}
            out_aux_dict["iter"] = i
            out_mpc_dict, out_aux_dict = self.model(in_obs_dict, out_aux_dict)
            x_t, x_ref, u_ref = out_mpc_dict["x_t"], out_mpc_dict["x_ref"], out_mpc_dict["u_ref"]
            try:
                if self.args.solver_type == "al" and i == 0 and (qp_solve or lastqp_solve):
                    self.tracking_mpc.reinitialize(x_ref, mask[:, :, None])
            except:
                ipdb.set_trace()
            # if (out_iter == 4900 or out_iter == 5000 or out_iter == 5500):
            #     ipdb.set_trace()
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            nominal_states_net = x_ref
            nominal_states = nominal_states_net
            nominal_actions = u_ref

            # Only run MPC after a few iterations, don't flow MPC gradients through the DEQ
            model_call = lambda xu: self.model_call(in_obs_dict, out_aux_dict, xu)
            if qp_solve and i >= opt_from_iter:
                nominal_states, nominal_actions, status = self.tracking_mpc(
                        x_t, xu_ref, x_ref, u_ref, model_call=model_call, al_iters=2
                        )
                out_aux_dict["x"] = nominal_states#.detach().clone()
                out_aux_dict["u"] = nominal_actions#.detach().clone()
                # out_aux_dict["xn"] = out_aux_dict["xn"].detach().clone()
                # if (out_iter == 5000 or out_iter == 5500):
                #     ipdb.set_trace()
            # if not lastqp_solve:
            #     out_aux_dict["x"] = out_aux_dict["x"].detach().clone()
            #     out_aux_dict["u"] = out_aux_dict["u"].detach().clone()
            
            if (qp_solve and i < opt_from_iter) or lastqp_solve:
                trajs.append((nominal_states_net, nominal_states.detach().clone(), 
                              nominal_actions.detach().clone()))
            else:
                # scales.append(out_mpc_dict["s"].detach().clone().mean().item())
                # Only supervise DEQ training or joint iterations for DEQMPC
                trajs.append((nominal_states_net, nominal_states, nominal_actions))
            
            if "deq_fwd_err" in out_aux_dict and out_aux_dict["deq_fwd_err"] is not None:
                deq_stats["fwd_err"].append(out_aux_dict["deq_fwd_err"])
                deq_stats["fwd_steps"].append(out_aux_dict["deq_fwd_steps"])
                deq_stats["jac_loss"].append(out_aux_dict["jac_loss"])

        # dyn_res = ((self.tracking_mpc.dyn(x_gt[:, :-1].reshape(-1, self.nx).double(
        # ), u_gt[:, :-1].reshape(-1, self.nu).double()) - x_gt[:,1:].reshape(-1, self.nx)).reshape(self.bsz, self.T-1, -1).norm(dim=-1)*mask[:, :-1]).norm(dim=-1).mean().item()
        # ipdb.set_trace()
        if lastqp_solve:
            nominal_states, nominal_actions, status = self.tracking_mpc(
                x_t, xu_ref, x_ref, u_ref, al_iters=10)
            trajs[-1] = (nominal_states_net, nominal_states, nominal_actions)        
        self.save_out_aux_dict(out_aux_dict)
        policy_out = {"trajs": trajs, "scales": scales}
        if len(deq_stats["fwd_err"]) > 0:
            policy_out["deq_stats"] = deq_stats
        return policy_out

    def deqmpc_iter_warm_start(self, obs, x_gt, u_gt, mask, out_iter=0, qp_solve=True, lastqp_solve=False):
        deq_iter = self.deq_iter
        opt_from_iter = 0
        trajs = []
        scales = []
        deq_stats = {"fwd_steps": [], "fwd_err": [], "jac_loss": []}
        if self.args.model_type == 'deq-1-mpc' or self.args.model_type == 'diff-mpc-nn':
            x_ref = torch.cat([obs]*self.T, dim=-1).detach().clone()
            x_ref = x_ref.view(-1, self.T, self.nx)
            self.x_init = x_ref
            nominal_actions = torch.zeros((obs.shape[0], self.T, self.nu), device=self.device)
            z = self.model.init_z(obs.shape[0])
            out_aux_dict = {"z": z, "x": x_ref, 'u': nominal_actions}
        else:
            out_aux_dict = self.out_aux_dict
        status = False
        # print(deq_iter)
        for i in range(deq_iter):
            in_obs_dict = {"o": obs}
            out_aux_dict["iter"] = i + 2
            out_mpc_dict, out_aux_dict = self.model(in_obs_dict, out_aux_dict)
            if i == 0:
                # if self.args.model_type == 'deq-1' or self.args.model_type == 'diff-mpc-nn':
                #     self.tracking_mpc.reinitialize(x_ref, mask[:, :, None])
                # else:
                self.tracking_mpc.warm_start_initialize(out_mpc_dict["x_ref"], out_mpc_dict["u_ref"])
            x_t, x_ref, u_ref = out_mpc_dict["x_t"], out_mpc_dict["x_ref"], out_mpc_dict["u_ref"]
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            nominal_states_net = x_ref
            nominal_states = nominal_states_net
            nominal_actions = u_ref
            if qp_solve and i >= opt_from_iter:
                nominal_states, nominal_actions, status = self.tracking_mpc(x_t, xu_ref, x_ref, u_ref, al_iters=2)
                out_aux_dict["x"] = nominal_states
                out_aux_dict["u"] = nominal_actions
            if (qp_solve and i < opt_from_iter) or lastqp_solve:
                trajs.append((nominal_states_net, nominal_states.detach().clone(), nominal_actions.detach().clone()))
            else:
                trajs.append((nominal_states_net, nominal_states, nominal_actions))
            
            if "deq_fwd_err" in out_aux_dict and out_aux_dict["deq_fwd_err"] is not None:
                deq_stats["fwd_err"].append(out_aux_dict["deq_fwd_err"])
                deq_stats["fwd_steps"].append(out_aux_dict["deq_fwd_steps"])
                deq_stats["jac_loss"].append(out_aux_dict["jac_loss"])
            if status:
                break
        if lastqp_solve:
            nominal_states, nominal_actions, status = self.tracking_mpc(
                x_t, xu_ref, x_ref, u_ref, al_iters=10)
            trajs[-1] = (nominal_states_net, nominal_states, nominal_actions)
        self.save_out_aux_dict(out_aux_dict)
        policy_out = {"trajs": trajs, "scales": scales}
        if len(deq_stats["fwd_err"]) > 0:
            policy_out["deq_stats"] = deq_stats
        return policy_out

    def model_call(self, in_obs_dict, out_aux_dict, xu_ref):
        with torch.no_grad():
            xsize = out_aux_dict["x"].shape[-1]
            x_ref, u_ref = xu_ref[:, :, :xsize], xu_ref[:, :, xsize:]
            out_aux_dict["x"] = x_ref.float()#.detach().clone()
            out_aux_dict["u"] = u_ref.float()
            out_mpc_dict, out_aux_dict = self.model(in_obs_dict, out_aux_dict, mem_update=False)
            x_ref_out, u_ref_out = out_mpc_dict["x_ref"], out_mpc_dict["u_ref"]
            xu_ref_out = torch.cat([x_ref_out, u_ref_out], dim=-1)
        return xu_ref_out.double()

    def save_out_aux_dict(self, out_aux_dict):
        last_x = self.tracking_mpc.dyn(out_aux_dict["x"][:, -1].double(), out_aux_dict["u"][:, -1].double()).float()
        self.out_aux_dict = {
            "z": torch.cat([out_aux_dict["z"][:, 1:], out_aux_dict["z"][:, -1:]], dim=1).detach().clone(),
            "x": torch.cat([out_aux_dict["x"][:, 1:], out_aux_dict["x"][:, -1:]], dim=1).detach().clone(),
            "u": torch.cat([out_aux_dict["u"][:, 1:], out_aux_dict["u"][:, -1:]], dim=1).detach().clone()
            }
        if self.addmem:
            self.out_aux_dict["mem"] = torch.cat([out_aux_dict["mem"][:, 1:], out_aux_dict["mem"][:, -1:]], dim=1).detach().clone()
            self.out_aux_dict["old_mem"] = torch.cat([out_aux_dict["old_mem"][:, 1:], out_aux_dict["old_mem"][:, -1:]], dim=1).detach().clone()

class DEQMPCPolicyHistory(DEQMPCPolicy):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.H = args.H
        if args.deq_out_type == 1:  # deq outputs only state predictions
            self.model = DEQLayerHistoryState(args, env).to(self.device)  
        elif args.deq_out_type == 2:  # deq outputs both state and action predictions
            self.model = DEQLayerHistory(args, env).to(self.device)
        else:
            raise NotImplementedError

    def forward(self, obs_hist, x_gt, u_gt, mask, out_iter=0, qp_solve=True, lastqp_solve=False):
        """
        Args:
            x_hist (tensor H x bsz x nx): input observation history, including current observation
        """
        if (self.H == 1):
            x_t = obs_hist.reshape(self.bsz, self.nx)
        else:
            x_t = obs_hist[:,-1].reshape(self.bsz, self.nx)
        x_ref = torch.cat([x_t]*self.T, dim=-1).detach().clone()
        x_ref = x_ref.view(-1, self.T, self.nx)
        self.x_init = x_ref
        nominal_actions = torch.zeros((self.bsz, self.T, self.nu), device=self.device)
        z = self.model.init_z(self.bsz)
        out_aux_dict = {"z": z, "x": x_ref, "u": nominal_actions}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(x_t, mask[:, :, None])

        # run the DEQ layer for deq_iter iterations
        policy_out = self.deqmpc_iter(obs_hist, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve)        
        return policy_out

class DEQMPCPolicyHistoryEstPred(DEQMPCPolicy):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.H = args.H
        self.state_estimator = Tracking_MPC(args, env, state_estimator=True)
        if args.deq_out_type == 1:  # deq outputs only state predictions
            self.model = DEQLayerHistoryStateEstPred(args, env).to(self.device)  
        elif args.deq_out_type == 2:  # deq outputs both state and action predictions
            self.model = DEQLayerHistory(args, env).to(self.device)
        else:
            raise NotImplementedError

    def forward(self, obs_hist, x_gt, u_gt, u_gt_est, mask, out_iter=0, qp_solve=True, lastqp_solve=False):
        """
        Args:
            x_hist (tensor H x bsz x nx): input observation history, including current observation
        """
        if (self.H == 1):
            x_t = obs_hist.reshape(self.bsz, self.nx)
        else:
            x_t = obs_hist[:,-1].reshape(self.bsz, self.nx)
        x_ref = torch.cat([x_t]*self.T, dim=-1).detach().clone()
        x_ref = x_ref.view(-1, self.T, self.nx)
        self.x_init = x_ref
        nominal_actions = torch.zeros((self.bsz, self.T, self.nu), device=self.device)
        z = self.model.init_z(self.bsz)
        out_aux_dict = {"z": z, "x": x_ref, "u": nominal_actions, "x_est": obs_hist.reshape(self.bsz, self.H, self.nx)}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(x_t, mask[:, :, None])
            self.state_estimator.reinitialize(x_t, mask[:, :, None])

        # run the DEQ layer for deq_iter iterations
        policy_out = self.deqmpc_iter(obs_hist, out_aux_dict, x_gt, u_gt, u_gt_est, mask, qp_solve, lastqp_solve)        
        return policy_out

    def deqmpc_iter(self, obs, out_aux_dict, x_gt, u_gt, u_gt_est, mask, qp_solve=False, lastqp_solve=False, out_iter=0): 
        deq_iter = self.deq_iter   
        opt_from_iter = 0

        trajs = []
        scales = []
        nominal_x_ests = []
        for i in range(deq_iter):
            in_obs_dict = {"o": obs}
            out_aux_dict["iter"] = i
            out_mpc_dict, out_aux_dict = self.model(in_obs_dict, out_aux_dict)
            x_t, x_ref, u_ref = out_mpc_dict["x_t"], out_mpc_dict["x_ref"], out_mpc_dict["u_ref"]
            x_est = out_aux_dict["x_est"]
            # if (out_iter == 4900 or out_iter == 5000 or out_iter == 5500):
            #     ipdb.set_trace()
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            nominal_states_net = x_ref
            nominal_states = nominal_states_net
            nominal_actions = u_ref
            nominal_x_est = x_est
            xu_est = torch.cat([x_est, u_gt_est], dim=-1)
            x_t_est = x_est[:, 0]

            # Only run MPC after a few iterations, don't flow MPC gradients through the DEQ
            if qp_solve and i >= opt_from_iter:
                nominal_states_est, _ = self.state_estimator(x_t_est, xu_est, x_est, u_gt_est, al_iters=2)
                nominal_states, nominal_actions = self.tracking_mpc(x_t, xu_ref, x_ref, u_ref, al_iters=2)
                out_aux_dict["x"] = nominal_states#.detach().clone()
                out_aux_dict["u"] = nominal_actions#.detach().clone()
                out_aux_dict["x_est"] = nominal_states_est#.detach().clone()
                # out_aux_dict["xn"] = out_aux_dict["xn"].detach().clone()
                # if (out_iter == 5000 or out_iter == 5500):
                #     ipdb.set_trace()
            # if not lastqp_solve:
            #     out_aux_dict["x"] = out_aux_dict["x"].detach().clone()
            #     out_aux_dict["u"] = out_aux_dict["u"].detach().clone()
            nominal_x_ests.append((nominal_x_est, x_est))
            if (qp_solve and i < opt_from_iter) or lastqp_solve:
                trajs.append((nominal_states_net, nominal_states.detach().clone(), nominal_actions.detach().clone()))
            else:
                # scales.append(out_mpc_dict["s"].detach().clone().mean().item())
                # Only supervise DEQ training or joint iterations for DEQMPC
                trajs.append((nominal_states_net, nominal_states, nominal_actions))

        dyn_res = (self.tracking_mpc.dyn(x_gt[:, :-1].reshape(-1, self.nx).double(
        ), u_gt[:, :-1].reshape(-1, self.nu).double()) - x_gt[:,1:].reshape(-1, self.nx)).reshape(self.bsz, -1).norm(dim=1).mean().item()
        self.network_time = []
        self.mpc_time = []

        if lastqp_solve:
            nominal_states, nominal_actions = self.tracking_mpc(
                x_t, xu_ref, x_ref, u_ref, al_iters=10)
            trajs[-1] = (nominal_states_net, nominal_states, nominal_actions)        
        policy_out = {"trajs": trajs, "dyn_res": dyn_res, "scales": scales, "nominal_x_ests": nominal_x_ests}    
        return policy_out

class DEQMPCPolicyFeedback(DEQMPCPolicy):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.model = DEQLayerFeedback(args, env).to(self.device)

    def forward(self, obs, x_gt, u_gt, mask, out_iter=0, qp_solve=True, lastqp_solve=False):
        x_ref = torch.cat([obs]*self.T, dim=-1).detach().clone()
        x_ref = x_ref.view(-1, self.T, self.nx)
        self.x_init = x_ref
        nominal_actions = torch.zeros((obs.shape[0], self.T, self.nu), device=self.device)
        z = self.model.init_z(self.bsz)
        # ipdb.set_trace()
        out_aux_dict = {"z": z, "xn": x_ref, "x": x_ref, 'u': nominal_actions}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(obs, mask[:, :, None])
    
        # run the DEQ layer for deq_iter iterations
        trajs, dyn_res, scales = self.deqmpc_iter(obs, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve, out_iter)        
        return trajs, dyn_res, scales, x_ref

class DEQMPCPolicyQ(DEQMPCPolicy):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.model = DEQLayerQ(args, env).to(self.device)

    def forward(self, obs, x_gt, u_gt, mask, out_iter=0, qp_solve=True, lastqp_solve=False):
        # obs is x0
        x_ref = torch.cat([obs]*self.T, dim=-1).detach().clone()
        x_ref = x_ref.view(-1, self.T, self.nx)
        self.x_init = x_ref
        nominal_actions = torch.zeros((obs.shape[0], self.T, self.nu), device=self.device)
        z = self.model.init_z(self.bsz)
        q = torch.ones_like(x_ref[:,:,0])
        # ipdb.set_trace()
        out_aux_dict = {"z": z, "x": x_ref, 'u': nominal_actions, 'q': q}

        if self.args.solver_type == "al":
            self.tracking_mpc.reinitialize(obs, mask[:, :, None])
    
        # run the DEQ layer for deq_iter iterations
        policy_out = self.deqmpc_iter(obs, out_aux_dict, x_gt, u_gt, mask, qp_solve, lastqp_solve, out_iter)        
        return policy_out

    def deqmpc_iter(self, obs, out_aux_dict, x_gt, u_gt, mask, qp_solve=False, lastqp_solve=False, out_iter=0): 
        deq_iter = self.deq_iter   
        opt_from_iter = 0

        trajs = []
        q_scalings = []
        for i in range(deq_iter):
            in_obs_dict = {"o": obs}
            out_aux_dict["iter"] = i
            out_mpc_dict, out_aux_dict = self.model(in_obs_dict, out_aux_dict)
            x_t, x_ref, u_ref, q_scaling = out_mpc_dict["x_t"], out_mpc_dict["x_ref"], out_mpc_dict["u_ref"], out_mpc_dict["q"]
            # if (out_iter == 4900 or out_iter == 5000 or out_iter == 5500):
            #     ipdb.set_trace()
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            nominal_states_net = x_ref
            nominal_states = nominal_states_net
            nominal_actions = u_ref

            # Only run MPC after a few iterations, don't flow MPC gradients through the DEQ
            if qp_solve and i >= opt_from_iter:
                # ipdb.set_trace()
                nominal_states, nominal_actions = self.tracking_mpc(x_t, xu_ref, x_ref, u_ref, q_scaling, al_iters=2)
                # out_aux_dict["x"] = nominal_states.detach().clone()
                # out_aux_dict["u"] = nominal_actions.detach().clone()

                # if (out_iter == 5000 or out_iter == 5500):
                #     ipdb.set_trace()
                
            # if not lastqp_solve:
            #     out_aux_dict["x"] = out_aux_dict["x"].detach().clone()
            #     out_aux_dict["u"] = out_aux_dict["u"].detach().clone()
            #     out_aux_dict["q"] = out_aux_dict["q"].detach().clone()
            
            q_scalings.append(q_scaling)
            if (qp_solve and i < opt_from_iter) or lastqp_solve:
                trajs.append((nominal_states_net, nominal_states.detach().clone(), nominal_actions.detach().clone()))
            else:
                # Only supervise DEQ training or joint iterations for DEQMPC
                trajs.append((nominal_states_net, nominal_states, nominal_actions))

        dyn_res = (self.tracking_mpc.dyn(x_gt[:, :-1].reshape(-1, self.nx).double(
        ), u_gt[:, :-1].reshape(-1, self.nu).double()) - x_gt[:,1:].reshape(-1, self.nx)).reshape(self.bsz, -1).norm(dim=1).mean().item()
        self.network_time = []
        self.mpc_time = []

        if lastqp_solve:
            nominal_states, nominal_actions = self.tracking_mpc(
                x_t, xu_ref, x_ref, u_ref, al_iters=10)
            trajs[-1] = (nominal_states_net, nominal_states, nominal_actions)    

        policy_out = {"trajs": trajs, "dyn_res": dyn_res, "q_scaling": q_scalings}    
        return policy_out


######################
# Loss computation
######################

def compute_loss_deqmpc_invres_l2_old(policy, gt_states, gt_actions, gt_mask, trajs, coeffs=None):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter": []}
    loss = 0.0
    losses = []
    residuals = []
    lossjs = []
    loss_proxies = []
    if coeffs is None:
        coeffs_pos = coeffs_vel = coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    else:
        coeffs = coeffs.view(len(trajs), -1)
        coeffs_pos = coeffs[:, 0]
        if coeffs.shape[1] > 1:
            coeffs_vel = coeffs[:, 1]
        else:
            coeffs_vel = torch.ones((len(trajs)), device=gt_states.device)
        if coeffs.shape[1] > 2:
            coeffs_act = coeffs[:, 2]
        else:
            coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    # supervise each DEQMPC iteration
    for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
        loss_j, res = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_proxy_j, res_proxy = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states_net, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        losses += [loss_j + policy.deq_reg * loss_proxy_j]
        loss_proxies += [loss_proxy_j]
        lossjs += [loss_j]
        # return_dict["losses_var"].append(loss_proxy_j.item())
        return_dict["losses_iter"].append(loss_proxy_j.mean().item())
        residuals.append(res)
    # ipdb.set_trace()
    residuals = torch.stack(residuals, dim=1)
    inv_residuals = 1/(residuals + 1e-8)
    inv_residuals = inv_residuals / inv_residuals.mean(dim=1, keepdim=True)
    losses = torch.stack(losses, dim=1)*(inv_residuals.detach().clone())
    loss = losses.mean(dim=0).sum()
    loss_end = compute_cost_coeff(
        policy, policy.out_type, policy.loss_type, gt_states, gt_actions, gt_mask,
        nominal_states, nominal_actions, coeffs_pos[-1], coeffs_vel[-1], coeffs_act[-1])[0].mean()
    return_dict["loss"] = loss
    return_dict["loss_end"] = loss_end
    # ipdb.set_trace()
    return return_dict

def compute_loss_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=None):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter_opt": [], "losses_iter_nn": [], "losses_iter_base": [], "losses_iter": []}
    trajs = policy_out["trajs"]
    loss = 0.0
    losses = []
    residuals = []
    loss_opts = []
    loss_nns = []
    if coeffs is None:
        coeffs_pos = coeffs_vel = coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    else:
        # ipdb.set_trace()
        coeffs = coeffs.view(len(trajs), -1)
        coeffs_pos = coeffs[:, 0]
        if coeffs.shape[1] > 1:
            coeffs_vel = coeffs[:, 1]
        else:
            coeffs_vel = torch.ones((len(trajs)), device=gt_states.device)
        if coeffs.shape[1] > 2:
            coeffs_act = coeffs[:, 2]
        else:
            coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    # supervise each DEQMPC iteration # replace x_init with gt[0]
    loss_init, res_init = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, policy.x_init, trajs[0][-1]*0,
                                    coeffs_pos[0], coeffs_vel[0], coeffs_act[0])
    residuals.append(res_init)
    for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
        loss_opt_j, res = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_nn_j, res_nn = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states_net, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        losses += [loss_opt_j + policy.deq_reg * loss_nn_j]
        loss_nns += [loss_nn_j]
        loss_opts += [loss_opt_j]
        # return_dict["losses_var"].append(loss_proxy_j.item())
        return_dict["losses_iter_opt"].append(loss_opt_j.mean().item())
        return_dict["losses_iter_nn"].append(loss_nn_j.mean().item())
        return_dict["losses_iter_base"].append(losses[-1].mean().item())
        return_dict["losses_iter"].append(losses[-1].mean().item())
        residuals.append(res)
    ### compute iteration weights based on previous losses and compute example weights based on net residuals
    ### iteration weights
    # ipdb.set_trace()
    residuals = torch.stack(residuals, dim=1)
    weight_mask = gt_mask.sum(dim=1) == 1
    iter_weights = 5**(torch.log(residuals[:,:1]/(10*residuals[:,:-1])))
    iter_weights[weight_mask] = 1
    iter_weights = iter_weights / iter_weights.sum(dim=1, keepdim=True)
    ex_weights = residuals.mean(dim=1, keepdim=True)#**2
    ex_weights = ex_weights / ex_weights.mean()
    losses = torch.stack(losses, dim=1)#*(ex_weights.detach().clone())#*(iter_weights.detach().clone())
    loss = losses.mean(dim=0).sum()
    loss_end = compute_cost_coeff(
        policy, policy.out_type, policy.loss_type, gt_states, gt_actions, gt_mask,
        nominal_states, nominal_actions, coeffs_pos[-1], coeffs_vel[-1], coeffs_act[-1])[0].mean()
    return_dict["loss"] = loss
    return_dict["loss_end"] = loss_end
    return_dict["residuals"] = residuals[:, -1]
    # return_dict["ex_weights"] = ex_weights
    # return_dict["iter_weights"] = iter_weights
    return return_dict

def compute_loss_deqmpc_hist(policy, gt_states, gt_actions, gt_obs, gt_mask, policy_out, coeffs=None):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter_opt": [], "losses_iter_nn": [], "losses_iter_base": [], "losses_iter": [], "losses_x_ests": []}
    trajs = policy_out["trajs"]
    loss = 0.0
    losses = []
    residuals = []
    loss_opts = []
    loss_nns = []
    if coeffs is None:
        coeffs_pos = coeffs_vel = coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    else:
        coeffs = coeffs.view(len(trajs), -1)
        coeffs_pos = coeffs[:, 0]
        if coeffs.shape[1] > 1:
            coeffs_vel = coeffs[:, 1]
        else:
            coeffs_vel = torch.ones((len(trajs)), device=gt_states.device)
        if coeffs.shape[1] > 2:
            coeffs_act = coeffs[:, 2]
        else:
            coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    # supervise each DEQMPC iteration # replace x_init with gt[0]
    loss_init, res_init = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, policy.x_init, trajs[0][-1]*0,
                                    coeffs_pos[0], coeffs_vel[0], coeffs_act[0])
    residuals.append(res_init)
    for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
        loss_opt_j, res = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_nn_j, res_nn = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states_net, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_hist_j, res_hist = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_obs,
                                    gt_actions, gt_mask*0+1, policy_out["nominal_x_ests"][j][0], nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_hist_nn_j, res_hist = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_obs,
                                    gt_actions, gt_mask*0+1, policy_out["nominal_x_ests"][j][1], nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        losses += [loss_opt_j + policy.deq_reg * loss_nn_j]# + loss_hist_j + policy.deq_reg * loss_hist_nn_j]
        loss_nns += [loss_nn_j]
        loss_opts += [loss_opt_j]
        # return_dict["losses_var"].append(loss_proxy_j.item())
        return_dict["losses_iter_opt"].append(loss_opt_j.mean().item())
        return_dict["losses_iter_nn"].append(loss_nn_j.mean().item())
        return_dict["losses_iter_base"].append(losses[-1].mean().item())
        return_dict["losses_iter"].append(losses[-1].mean().item())
        return_dict["losses_x_ests"].append(loss_hist_j.mean().item())
        residuals.append(res)
    ### compute iteration weights based on previous losses and compute example weights based on net residuals
    ### iteration weights
    residuals = torch.stack(residuals, dim=1)
    weight_mask = gt_mask.sum(dim=1) == 1
    iter_weights = 5**(torch.log(residuals[:,:1]/(10*residuals[:,:-1])))
    iter_weights[weight_mask] = 1
    iter_weights = iter_weights / iter_weights.sum(dim=1, keepdim=True)
    ex_weights = residuals.mean(dim=1, keepdim=True)#**2
    ex_weights = ex_weights / ex_weights.mean()
    losses = torch.stack(losses, dim=1)#*(ex_weights.detach().clone())#*(iter_weights.detach().clone())
    loss = losses.mean(dim=0).sum()
    loss_end = compute_cost_coeff(
        policy, policy.out_type, policy.loss_type, gt_states, gt_actions, gt_mask,
        nominal_states, nominal_actions, coeffs_pos[-1], coeffs_vel[-1], coeffs_act[-1])[0].mean()
    return_dict["loss"] = loss
    return_dict["loss_end"] = loss_end
    return_dict["ex_weights"] = ex_weights
    return_dict["iter_weights"] = iter_weights
    return return_dict


def compute_gradratios_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out):
    trajs = policy_out["trajs"]
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter": []}
    losses = []
    loss_proxies = []
    # supervise each DEQMPC iteration
    for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
        loss_j = compute_decomposed_loss(policy, policy.out_type, policy.loss_type, gt_states,
                                           gt_actions, gt_mask, nominal_states, nominal_actions)
        loss_proxy_j = compute_decomposed_loss(policy, policy.out_type, policy.loss_type, gt_states,
                                           gt_actions, gt_mask, nominal_states_net, nominal_actions)
        loss_proxies += loss_proxy_j
        losses += loss_j #+ policy.deq_reg * loss_proxy_j
        # return_dict["losses_var"].append(loss_proxy_j.item())
        # return_dict["losses_iter"].append(loss_proxy_j.item())
    losses = torch.stack(losses, dim=0)
    loss_proxies = torch.stack(loss_proxies, dim=0)
    
    grads = torch.stack([torch.autograd.grad(losses[i] + policy.deq_reg*loss_proxies[i], policy.model.out_layer[0].weight, retain_graph=True)[0].view(-1) for i in range(len(losses))], dim=0).norm(dim=-1) 
    # ipdb.set_trace()
    # if (grads < 1e-8).all():
    #     ipdb.set_trace()
    # if not (grads > 1e-8).all():
    #     ipdb.set_trace()
    # try:
    grad_idx = torch.where(grads>1e-8)[0][0]
    # except:
    #     ipdb.set_trace()
    grad_ratios = grads[grad_idx] / grads
    grad_ratios = torch.where(grad_ratios > 1e6, torch.ones_like(grad_ratios), grad_ratios)
    # compute moving averages
    return grad_ratios, losses.reshape((len(trajs), len(loss_j))).mean(dim=-1), loss_proxies.reshape((len(trajs), len(loss_j))).mean(dim=-1)

def compute_loss_deqmpc_qscaling(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=None):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter_opt": [], "losses_iter_nn": [], "losses_iter_base": [], "losses_iter": [], "q_scaling": []}
    trajs = policy_out["trajs"]
    q_scaling = policy_out["q_scaling"]
    loss = 0.0
    # supervise each DEQMPC iteration
    losses = []
    residuals = []
    loss_opts = []
    loss_nns = []
    if coeffs is None:
        coeffs_pos = coeffs_vel = coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    else:
        coeffs = coeffs.view(len(trajs), -1)
        coeffs_pos = coeffs[:, 0]
        if coeffs.shape[1] > 1:
            coeffs_vel = coeffs[:, 1]
        else:
            coeffs_vel = torch.ones((len(trajs)), device=gt_states.device)
        if coeffs.shape[1] > 2:
            coeffs_act = coeffs[:, 2]
        else:
            coeffs_act = torch.ones((len(trajs)), device=gt_states.device)
    # supervise each DEQMPC iteration
    loss_init, res_init = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, policy.x_init, trajs[0][-1]*0,
                                    coeffs_pos[0], coeffs_vel[0], coeffs_act[0])
    residuals.append(res_init)
    for j, (nominal_states_net, nominal_states, nominal_actions) in enumerate(trajs):
        loss_opt_j, res = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        loss_nn_j, res_nn = compute_cost_coeff(policy, policy.out_type, policy.loss_type, gt_states,
                                    gt_actions, gt_mask, nominal_states_net, nominal_actions,
                                    coeffs_pos[j], coeffs_vel[j], coeffs_act[j])
        q_scaling_j = q_scaling[j]
        loss_q_scaling_j = torch.abs(q_scaling_j - 1.0).sum(dim=1)
        losses += [loss_opt_j + policy.deq_reg * loss_nn_j + 0.02 * loss_q_scaling_j]
        loss_nns += [loss_nn_j]
        loss_opts += [loss_opt_j]
        # return_dict["losses_var"].append(loss_proxy_j.item())
        return_dict["losses_iter_opt"].append(loss_opt_j.mean().item())
        return_dict["losses_iter_nn"].append(loss_nn_j.mean().item())
        return_dict["losses_iter_base"].append((loss_opt_j + policy.deq_reg * loss_nn_j).mean().item())
        return_dict["losses_iter"].append(losses[-1].mean().item())
        return_dict["q_scaling"].append(loss_q_scaling_j.mean().item())
        residuals.append(res)
    ### compute iteration weights based on previous losses and compute example weights based on net residuals
    ### iteration weights
    residuals = torch.stack(residuals, dim=1)
    weight_mask = gt_mask.sum(dim=1) == 1
    iter_weights = 5**(torch.log(residuals[:,:1]/(10*residuals[:,:-1])))
    iter_weights[weight_mask] = 1
    iter_weights = iter_weights / iter_weights.sum(dim=1, keepdim=True)
    ex_weights = residuals.mean(dim=1, keepdim=True)#**2
    ex_weights = ex_weights / ex_weights.mean()
    losses = torch.stack(losses, dim=1)#*(ex_weights.detach().clone())#*(iter_weights.detach().clone())
    loss = losses.mean(dim=0).sum()
    loss_end = compute_cost_coeff(
        policy, policy.out_type, policy.loss_type, gt_states, gt_actions, gt_mask,
        nominal_states, nominal_actions, coeffs_pos[-1], coeffs_vel[-1], coeffs_act[-1])[0].mean()
    return_dict["loss"] = loss
    return_dict["loss_end"] = loss_end
    return_dict["ex_weights"] = ex_weights
    return_dict["iter_weights"] = iter_weights
    return return_dict

def compute_loss_bc(policy, gt_states, gt_actions, gt_mask, policy_out):
    return_dict = {"loss": 0.0, "loss_end": 0.0, "losses_var": [], "losses_iter": []}
    trajs = policy_out["trajs"]
    loss = 0.0
    nominal_states, nominal_actions = trajs
    loss = add_loss_based_on_out_type(
        policy, policy.out_type, policy.loss_type, gt_states, gt_actions, gt_mask, nominal_states, nominal_actions)
    loss_end = torch.Tensor([0.0])
    return_dict["loss"] = loss
    return_dict["loss_end"] = loss_end
    return return_dict

def add_loss_based_on_out_type(policy, out_type, loss_type, gt_states, gt_actions, gt_mask, nominal_states, nominal_actions):
    loss = 0.0
    if out_type == 0 or out_type == 2:
        # supervise action
        if loss_type == "l2":
            loss += torch.norm((nominal_actions - gt_actions) *
                               gt_mask[:, :, None]).pow(2).mean()
        elif loss_type == "l1":
            loss += torch.abs((nominal_actions - gt_actions) *
                              gt_mask[:, :, None])[:,:policy.T-1].sum(dim=-1).mean()
        # loss += torch.abs((nominal_actions - gt_actions) *
        #                   gt_mask[:, :, None])[:,:policy.T-1].sum(dim=-1).mean()
        # ipdb.set_trace()
    if out_type == 1 or out_type == 2:
        # supervise state
        if loss_type == "l2":
            loss += torch.norm((nominal_states - gt_states) *
                               gt_mask[:, :, None]).pow(2).mean()
        elif loss_type == "l1":
            loss += torch.abs((nominal_states - gt_states) *
                            gt_mask[:, :, None]).sum(dim=-1).mean()
    if out_type == 3:
        # supervise configuration
        if loss_type == "l2":
            loss += torch.norm((nominal_states[..., :policy.nq] - gt_states[..., :policy.nq]) *
                               gt_mask[:, :, None]).pow(2).mean()
        elif loss_type == "l1":
            loss += torch.abs((nominal_states[..., :policy.nq] - gt_states[..., :policy.nq]) *
                          gt_mask[:, :, None]).sum(dim=-1).mean()
        
    return loss

def compute_cost_coeff(policy, out_type, loss_type, gt_states, gt_actions, gt_mask, nominal_states, nominal_actions, coeffs_act, coeffs_pos, coeffs_vel):
    loss = 0.0
    resi = resj = resk = 0.0
    if out_type == 0 or out_type == 2:
        # supervise action
        _, lossk, resk = loss_type_conditioned_compute_loss(nominal_actions[:, :policy.T-1], gt_actions[:, :policy.T-1], gt_mask[:, :policy.T-1], loss_type)
        loss = lossk*coeffs_act
    if out_type == 1 or out_type == 2:
        # supervise state
        _, lossi, resi = loss_type_conditioned_compute_loss(nominal_states[:, :, :policy.nq], gt_states[:, :, :policy.nq], gt_mask, loss_type)
        _, lossj, resj = loss_type_conditioned_compute_loss(nominal_states[:, :, policy.nq:], gt_states[:, :, policy.nq:], gt_mask, loss_type)
        loss += lossi*coeffs_pos + lossj*coeffs_vel
    if out_type == 3:
        # supervise configuration
        _, lossi, resi = loss_type_conditioned_compute_loss(nominal_states[:, :, :policy.nq], gt_states[:, :, :policy.nq], gt_mask, loss_type)
        loss += lossi*coeffs_pos
    return loss, resi + resj + resk

def compute_decomposed_loss(policy, out_type, loss_type, gt_states, gt_actions, gt_mask, nominal_states, nominal_actions):
    loss = []
    if out_type == 0 or out_type == 2:
        # supervise action
        loss += [loss_type_conditioned_compute_loss(nominal_actions[:, :policy.T-1], gt_actions[:, :policy.T-1], gt_mask[:, :policy.T-1], loss_type)[0]]
    if out_type == 1 or out_type == 2:
        # supervise state
        loss += [loss_type_conditioned_compute_loss(nominal_states[:, :, :policy.nq], gt_states[:, :, :policy.nq], gt_mask, loss_type)[0]]
        loss += [loss_type_conditioned_compute_loss(nominal_states[:, :, policy.nq:], gt_states[:, :, policy.nq:], gt_mask, loss_type)[0]]
    if out_type == 3:
        # supervise configuration
        loss += [loss_type_conditioned_compute_loss(nominal_states[:, :, :policy.nq], gt_states[:, :, :policy.nq], gt_mask, loss_type)[0]]
    return loss

def loss_type_conditioned_compute_loss(pred, targ, mask, loss_type):
    res = torch.abs((pred - targ) * mask[:, :, None]).sum(dim=-1)
    if loss_type == "l2":        
        l2 = torch.norm((pred - targ) * mask[:, :, None], dim=-1).pow(2)
        return l2.mean(), l2.mean(dim=1), res.mean(dim=1)
    elif loss_type == "l1":
        l1 = torch.abs((pred - targ) * mask[:, :, None]).sum(dim=-1)
        return l1.mean(), l1.mean(dim=1), res.mean(dim=1)
    elif loss_type == "hinge":
        l1 = torch.abs((pred - targ) * mask[:, :, None])
        l2 = ((pred - targ) * mask[:, :, None]).pow(2)
        hingel = torch.min(l1, l2).sum(dim=-1)
        return hingel.mean(), hingel.mean(dim=1), res.mean(dim=1)

def compute_loss(policy, gt_states, gt_actions, gt_obs, gt_mask, policy_out, deq, deqmpc, coeffs=None):
    if True:#deq:
        # deq or deqmpc
        if deqmpc:
            # full deqmpc
            if "nominal_x_ests" in policy_out.keys():
                return compute_loss_deqmpc_hist(policy, gt_states, gt_actions, gt_obs, gt_mask, policy_out, coeffs=coeffs)
            elif "q_scaling" in policy_out.keys():
                return compute_loss_deqmpc_qscaling(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=coeffs)
            else:
                return compute_loss_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=coeffs)
        else:
            # deq -- pretrain
            return compute_loss_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=coeffs)
    # else:
    #     # vanilla behavior cloning
    #     return compute_loss_bc(policy, gt_states, gt_actions, gt_mask, policy_out, coeffs=coeffs)
    
def compute_grad_coeff(policy, gt_states, gt_actions, gt_mask, policy_out, deq, deqmpc):
    if True:#deq:
        # deq or deqmpc
        if deqmpc:
            # full deqmpc
            # if "q_scaling" in policy_out.keys():
            #     return compute_gradratios_deqmpc_qscaling(policy, gt_states, gt_actions, gt_mask, policy_out)
            # else:
            return compute_gradratios_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out)
        else:
            # deq -- pretrain
            return compute_gradratios_deqmpc(policy, gt_states, gt_actions, gt_mask, policy_out)
    # else:
    #     # vanilla behavior cloning
    #     return compute_loss_bc(policy, gt_states, gt_actions, gt_mask, policy_out)


######################
# Other policies
######################

class FFDNetwork(torch.nn.Module):
    """
    Feedforward network to generate reference trajectories of horizon T
    """

    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.nq = args.nq
        self.dt = env.dt
        self.T = args.T
        self.hdim = args.hdim
        self.layer_type = args.layer_type
        self.inp_type = ""
        self.out_type = args.deq_out_type
        self.loss_type = args.loss_type
        self.deq_reg = args.deq_reg
        self.kernel_width = args.kernel_width
        self.pooling = args.pooling
        self.expand = 4
        self.num_groups = 4
        self.pos_scale = args.pos_scale
        self.vel_scale = args.vel_scale
        self.expansion_type = args.expansion_type
        self.kernel_width_out = 1

        self.setup_layers()
        
        self.out_dim = self.nx + self.nu

    # TO BE OVERRIDEN
    def setup_layers(self):
        self.in_dim = self.nx + self.nx * (self.T - 1) # current state and state prediction
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hdim),
                torch.nn.LayerNorm(self.hdim),
                # torch.nn.ReLU()
            )
            expanddim = self.hdim*self.deq_expand
            self.fcdeq1 = torch.nn.Linear(self.hdim, expanddim)
            self.lndeq1 = torch.nn.LayerNorm(expanddim)
            self.reludeq1 = torch.nn.ReLU()
            self.fcdeq2 = torch.nn.Linear(expanddim, self.hdim)
            self.lndeq2 = torch.nn.LayerNorm(self.hdim)
            self.reludeq2 = torch.nn.ReLU()
            self.lndeq3 = torch.nn.LayerNorm(self.hdim)


            self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hdim, self.out_dim)
            )
            self.gradnorm = GradNormLayer(self.out_dim)

        elif self.layer_type == "gcn":
            # Get sinusoidal embeddings for the time steps
            # self.time_encoder = nn.Sequential(
            #     SinusoidalPosEmb(self.hdim),
            #     nn.Linear(self.hdim, self.hdim*4),
            #     nn.Mish(),
            #     nn.Linear(self.hdim*4, self.hdim),
            #     nn.LayerNorm(self.hdim)
            #     )
            self.time_emb = torch.nn.Parameter(torch.randn(self.T-1, self.hdim))
            # Get the node embeddings
            self.node_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.ReLU()
            )

            self.x0_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.ReLU()
            )

            self.input_encoder = nn.Sequential(
                nn.Conv1d(self.hdim*3, self.hdim*4, self.kernel_width, padding='same'),
                nn.ReLU(),
                nn.Conv1d(self.hdim*4, self.hdim, self.kernel_width, padding='same'),
                nn.GroupNorm(self.num_groups, self.hdim),
                # nn.Mish()
            )

            self.global_pooling = {
                "max": torch.max,
                "mean": torch.mean,
                "sum": torch.sum
            }[self.pooling]

            if self.expansion_type == "width":
                self.convdeq1 = torch.nn.Conv1d(
                    self.hdim, self.hdim*self.expand, self.kernel_width, padding='same')
                self.convdeq2 = torch.nn.Conv1d(
                    self.hdim*self.expand, self.hdim, self.kernel_width, padding='same')
                # self.mishdeq1 = torch.nn.Mish()#
                self.mishdeq1 = torch.nn.ReLU()
                # self.mishdeq2 = torch.nn.Mish() #
                self.mishdeq2 = torch.nn.ReLU()
                self.gndeq1 = torch.nn.GroupNorm(
                    self.num_groups, self.hdim*self.expand)
                self.gndeq2 = torch.nn.GroupNorm(self.num_groups, self.hdim)
                self.gndeq3 = torch.nn.GroupNorm(self.num_groups, self.hdim)
            elif self.expansion_type == "depth":
                # define a sequence of conv layers (self.expand) times
                self.res_layers = []
                self.post_res_relu = torch.nn.ReLU()
                for i in range(self.expand):
                    self.res_layers.append(torch.nn.Sequential(
                        torch.nn.Conv1d(self.hdim, self.hdim, self.kernel_width, padding='same'),
                        torch.nn.GroupNorm(self.num_groups, self.hdim),
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(self.hdim, self.hdim, self.kernel_width, padding='same'),
                        torch.nn.GroupNorm(self.num_groups, self.hdim),
                    ))
                self.res_layers = torch.nn.ModuleList(self.res_layers)

            self.out_layer = torch.nn.Sequential(
                torch.nn.Conv1d(self.hdim, self.hdim, self.kernel_width, padding='same'),
                torch.nn.GroupNorm(self.num_groups, self.hdim),
                torch.nn.ReLU(),
                torch.nn.Conv1d(self.hdim, self.nx, self.kernel_width_out, padding='same'),
            )

        elif self.layer_type == "gat":
            NotImplementedError

    def layers_forward(self, x, obs):
        if self.layer_type == "mlp":
            x = self.inp_layer(x)
            x = self.fcdeq1(x)
            x = self.lndeq1(x)
            x = self.reludeq1(x)
            x = self.fcdeq2(x)
            x = self.lndeq2(x)
            x = self.reludeq2(x)
            x = self.lndeq3(x)
            x = self.out_layer(x)
        elif self.layer_type == "gcn":
            t = self.time_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)#[:,1:]
            x = x.reshape(-1, self.T, self.nx)[:,1:]
            x_emb = self.node_encoder(x)
            x0_emb = self.x0_encoder(obs[:, 0]).unsqueeze(1).repeat(1, self.T-1, 1)  #TODO switch case for out_type
            inp = torch.cat([x_emb, x0_emb, t], dim=-1).transpose(1, 2).reshape(-1, self.hdim*3, self.T-1)
            inp = self.input_encoder(inp)
            
            # z = z.view(-1, self.T-1, self.hdim).permute(0, 2, 1)
            z = inp
            if self.expansion_type == "width":
                y = self.convdeq1(z)
                y = self.mishdeq1(y)
                y = self.gndeq1(y)
                z = self.mishdeq2(
                    z + self.gndeq2(self.convdeq2(y)))
                # out = out.permute(0, 2, 1)#.reshape(-1, self.hdim)
            elif self.expansion_type == "depth":
                for layer in self.res_layers:
                    z = layer(z)
                    z = self.post_res_relu(z)
                # out = z.permute(0, 2, 1)

            # bsz = z.shape[0]
            # z = z.view(-1, self.T-1, self.hdim).permute(0, 2, 1)
            out = self.out_layer(z).permute(0, 2, 1)
        elif self.layer_type == "gat":
            NotImplementedError
        return out
    
    def forward(self, in_obs_dict, in_aux_dict):
        """
        compute the policy output for the given state x
        """
        obs, x_prev, z = in_obs_dict["o"], in_aux_dict["x"], in_aux_dict["z"]
        bsz = obs.shape[0]
        obs = obs.view(bsz,1,-1)
        dx_ref = self.layers_forward(x_prev, obs)
        dx_ref = dx_ref.reshape(-1, self.T-1, self.nx)
        vel_ref = dx_ref[..., self.nq:]
        dx_ref = dx_ref[..., :self.nq]*self.dt
        x_ref = torch.cat([dx_ref + x_prev[..., :1, :self.nq], vel_ref], dim=-1)
        x_ref = torch.cat([obs, x_ref], dim=-2)
        u_ref = torch.zeros(bsz, self.T, self.nu, device=x_ref.device)

        out_mpc_dict = {"x_t" : obs[:,0], "x_ref": x_ref, "u_ref": u_ref}
        out_aux_dict = {"x": x_ref, "u": u_ref, "z": z}
        return out_mpc_dict, out_aux_dict
    def init_z(self, bsz):
        return torch.zeros(bsz, self.T-1, self.hdim)


class Tracking_MPC(torch.nn.Module):
    def __init__(self, args, env, state_estimator=False):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.nq = env.nq
        self.dt = env.dt
        self.T = args.T
        self.dyn = env.dynamics
        self.dyn_jac = env.dynamics_derivatives
        self.state_estimator = state_estimator

        # May comment out input constraints for now
        self.device = args.device
        self.u_upper = torch.tensor(env.action_space.high).to(self.device)
        self.u_lower = torch.tensor(env.action_space.low).to(self.device)
        self.qp_iter = args.qp_iter
        self.eps = args.eps
        self.warm_start = args.warm_start
        self.bsz = args.bsz

        self.Q = args.Q.to(self.device)
        self.R = args.R.to(self.device)
        self.dtype = torch.float64 if args.dtype == "double" else torch.float32
        # self.Qf = args.Qf
        if args.Q is None:
            self.Q = torch.ones(self.nx, dtype=self.dtype, device=self.device)
            # self.Qf = torch.ones(self.nx, dtype=torch.float32, device=self.device)
            self.R = torch.ones(self.nu, dtype=self.dtype, device=self.device)
        self.Q = torch.cat([self.Q, self.R], dim=0).to(self.dtype)
        self.aux_cost = ''#polepos'#quad_pos'
        if self.aux_cost == 'quad_pos':
            self.aux_Q = self.Q*0.2
            self.aux_Q[3:] = 0.0
            self.aux_x = torch.zeros_like(self.aux_Q)
            self.aux_x[:3] = torch.tensor([7.4720e-02, -1.3457e-01,  2.4619e-01]).to(self.aux_x)
            self.aux_Q = torch.diag(self.aux_Q).repeat(self.bsz, self.T, 1, 1)
            self.aux_p = -(self.aux_Q * self.aux_x.unsqueeze(-2)).sum(dim=-1)
            self.q_mask = torch.ones(self.bsz, dtype=self.dtype, device=self.device)
        elif self.aux_cost == 'polepos':
            self.aux_Q = self.Q*0.5
            self.aux_Q[:6] = 0.0
            self.aux_Q[7:] = 0.0
            self.aux_x = torch.zeros_like(self.aux_Q)
            self.aux_x[6] = np.pi
            self.aux_Q = torch.diag(self.aux_Q).repeat(self.bsz, self.T, 1, 1)
            self.aux_p = -(self.aux_Q * self.aux_x.unsqueeze(-2)).sum(dim=-1)
            self.q_mask = torch.ones(self.bsz, dtype=self.dtype, device=self.device)
        else:
            self.aux_Q = self.Q*0.2
            self.aux_p = torch.zeros_like(self.aux_Q)
            self.aux_Q = torch.diag(self.aux_Q).repeat(self.bsz, self.T, 1, 1)*0.0
            self.q_mask = torch.ones(self.bsz, dtype=self.dtype, device=self.device)

        self.Q = torch.diag(self.Q).repeat(self.bsz, self.T, 1, 1)
        self.u_init = torch.randn(
            self.bsz, self.T, self.nu, dtype=self.dtype, device=self.device
        )

        self.single_qp_solve = True if self.qp_iter == 1 else False

        if args.solver_type == "al":
            if "obstacles" in args.env:
                self.ctrl = Obstacle_MPC(
                    self.nx,
                    self.nu,
                    self.T,
                    u_lower=self.u_lower,
                    u_upper=self.u_upper,
                    # al_iter=self.qp_iter,
                    exit_unconverged=False,
                    eps=1e-2,
                    n_batch=self.bsz,
                    backprop=False,
                    verbose=0,
                    u_init=self.u_init,
                    solver_type="dense",
                    dtype=self.dtype,
                    state_estimator=self.state_estimator,
                    env = env,
                )
            else:
                self.ctrl = al_mpc.MPC(
                    self.nx,
                    self.nu,
                    self.T,
                    u_lower=self.u_lower,
                    u_upper=self.u_upper,
                    # al_iter=self.qp_iter,
                    exit_unconverged=False,
                    eps=1e-2,
                    n_batch=self.bsz,
                    backprop=False,
                    verbose=0,
                    u_init=self.u_init,
                    solver_type="dense",
                    dtype=self.dtype,
                    state_estimator=self.state_estimator,
                )
        else:
            self.ctrl = ip_mpc.MPC(
                self.nx,
                self.nu,
                self.T,
                u_lower=self.u_lower,
                u_upper=self.u_upper,
                qp_iter=self.qp_iter,
                exit_unconverged=False,
                eps=1e-5,
                n_batch=self.bsz,
                backprop=False,
                verbose=0,
                u_init=self.u_init.transpose(0, 1),
                grad_method=ip_mpc.GradMethods.ANALYTIC,
                solver_type="dense",
                single_qp_solve=self.single_qp_solve,
            )

    def forward(self, x0, xu_ref, x_ref, u_ref, q_scaling=None, model_call=None, al_iters=2):
        """
        compute the mpc output for the given state x and reference x_ref
        """
        if self.args.solver_type == "al":
            xu_ref = torch.cat([x_ref, u_ref], dim=-1)
            if self.x_init is None:
                self.x_init = self.ctrl.x_init = x_ref.detach().clone()
                self.u_init = self.ctrl.u_init = u_ref.detach().clone()
        if (q_scaling is not None):
            # ipdb.set_trace()
            q_scaling = q_scaling + torch.ones_like(q_scaling)
            Q = self.Q * q_scaling[:,:,None,None]
        else:
            Q = self.Q
        # self.q_mask = torch.logical_and((x0[:, 6] - np.pi).abs() < 0.1 , (x0[:, -1]).abs() < 0.3).float()
        aux_Q = self.aux_Q*self.q_mask[:, None, None, None]
        p, f = self.compute_pf(xu_ref, Q)
        # ipdb.set_trace()
        # xu = self.ctrl.get_xu()
        # cost_xu = (xu*(Q*xu.unsqueeze(-2)).sum(dim=-1)).sum(dim=-1) + (self.p*xu).sum(dim=-1) + self.f
        # print(cost_xu.sum(dim=0))
        # ipdb.set_trace()
        Q = Q + self.aux_Q
        # ipdb.set_trace()
        if self.args.solver_type == "al":
            self.ctrl.al_iter = al_iters
            cost = al_utils.QuadCost(Q, self.p, self.f)
        else:
            cost = ip_mpc.QuadCost(Q.transpose(
                0, 1), self.p.transpose(0, 1))
            self.ctrl.u_init = self.u_init.transpose(0, 1)
        # ipdb.set_trace()
        state = x0  # .unsqueeze(0).repeat(self.bsz, 1)
        # ipdb.set_trace()
        # cost_xu = self.ctrl.get_cost(cost)
        # print("Before : ", cost_xu.mean().item())
        compute_Qq = lambda xu : self.compute_Qq(xu, model_call)
        nominal_states, nominal_actions, status = self.ctrl(
            state, cost, self.dyn, self.dyn_jac, compute_Qq)
        if self.args.solver_type == "ip":
            nominal_states = nominal_states.transpose(0, 1)
            nominal_actions = nominal_actions.transpose(0, 1)
        # ipdb.set_trace()
        self.u_init = nominal_actions.clone().detach()
        # cost_xu = self.ctrl.get_cost(cost)
        # print("After : ", cost_xu.mean().item())
        return nominal_states, nominal_actions, status

    def compute_pf(self, x_ref, Q):
        """
        compute the p for the quadratic objective using self.Q as the diagonal matrix and the reference x_ref at each time without a for loop
        """
        # self.p = torch.zeros(
        #     self.T, self.bsz, self.nx + self.nu, dtype=torch.float32, device=self.device
        # )
        # self.p[:, :, : self.nx] = -(
        #     self.Q[:, :, : self.nx, : self.nx] * x_ref.unsqueeze(-2)
        # ).sum(dim=-1)
        self.p = -(Q * x_ref.unsqueeze(-2)).sum(dim=-1) + self.aux_p*self.q_mask[:, None, None]
        self.f = 0.5 * (x_ref * (Q * x_ref.unsqueeze(-2)).sum(dim=-1)).sum(dim=-1)
        return self.p, self.f

    def reinitialize(self, x, mask):
        self.u_init = torch.randn(
            self.bsz, self.T, self.nu, dtype=x.dtype, device=x.device)
        self.x_init = None
        self.ctrl.reinitialize(x, mask)
    
    def warm_start_initialize(self, x_ref, u_ref):
        self.u_init = self.ctrl.u_init
        self.x_init = self.ctrl.x_init
        self.u_init[:, -1:] = u_ref[:, -1:]
        self.x_init[:, -1:] = x_ref[:, -1:]
        self.ctrl.warm_start_initialize(self.x_init, self.u_init, self.args)
    
    def compute_Qq(self, xu, model_call):
        xu_ref = model_call(xu)
        p, f = self.compute_pf(xu_ref, self.Q)
        return self.Q.diagonal(dim1=-2, dim2=-1), p, f

class NNMPCPolicy(torch.nn.Module):
    """
    Feedforward Neural network based MPC policy
    """

    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.nq = args.nq
        self.T = args.T
        self.dt = env.dt
        self.device = args.device
        self.hdim = args.hdim
        self.out_type = args.policy_out_type
        self.loss_type = args.loss_type
        self.bsz = args.bsz
        self.deq_reg = args.deq_reg
        self.model = FFDNetwork(args, env)
        self.model.to(self.device)
        self.tracking_mpc = Tracking_MPC(args, env)
        self.mpc_time = []
        self.network_time = []

    def forward(self, x, x_gt, u_gt, mask, lastqp_solve=False, warm_start=False, **kwargs):
        """
        compute the policy output for the given state x
        """
        x_ref = torch.cat([x]*self.T, dim=1).detach().clone()
        x_ref = x_ref.view(-1, self.T, self.nx)
        self.x_init = x_ref
        nominal_actions = torch.zeros((x.shape[0], self.T, self.nu), device=self.device)
        z = torch.zeros((x.shape[0], self.T, self.nx), device=self.device)
        out_aux_dict = {"z": z, 'x' : x_ref, "u" : nominal_actions}
        in_obs_dict = {"o": x}
        out_mpc_dict, out_aux_dict = self.model(in_obs_dict, out_aux_dict)
        x_t, x_ref, u_ref = out_mpc_dict['x_t'], out_mpc_dict['x_ref'], out_mpc_dict['u_ref']
        xu_ref = torch.cat([x_ref, u_ref], dim=-1)
        nominal_actions = u_ref
        nominal_states_net = x_ref
        nominal_states = nominal_states_net
        if lastqp_solve:
            if warm_start:
                self.tracking_mpc.warm_start_initialize(x_ref, u_ref)
                nominal_states, nominal_actions, status = self.tracking_mpc(x_t, xu_ref, x_ref, u_ref, al_iters=6)
            else:
                self.tracking_mpc.reinitialize(x_ref, mask)
                nominal_states, nominal_actions, status = self.tracking_mpc(x_t, xu_ref, x_ref, u_ref ,al_iters=10)
        trajs = [(nominal_states_net, nominal_states, nominal_actions)]
        self.save_out_aux_dict(out_aux_dict)
        policy_out = {"trajs": trajs, "scales": []}
        # x_ref = x_ref.view(-1, self.nq)
        return policy_out

    def save_out_aux_dict(self, out_aux_dict):
        last_x = self.tracking_mpc.dyn(out_aux_dict["x"][:, -1].double(), out_aux_dict["u"][:, -1].double()).float()
        self.out_aux_dict = {
            "x": torch.cat([out_aux_dict["x"][:, 1:], out_aux_dict["x"][:, -1:]], dim=1).detach().clone(),
            "u": torch.cat([out_aux_dict["u"][:, 1:], out_aux_dict["u"][:, -1:]], dim=1).detach().clone()
            }

class NNPolicy(torch.nn.Module):
    """
    Some NN-based policy trained with behavioral cloning, outputting a trajectory (state or input) of horizon T
    """

    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.nq = args.nq
        self.T = args.T
        self.dt = env.dt
        self.bsz = args.bsz
        self.deq_reg = args.deq_reg
        self.device = args.device
        self.hdim = args.hdim
        self.out_type = args.policy_out_type
        self.loss_type = args.loss_type

        # define the network layers :
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.nx, self.hdim),
            torch.nn.LayerNorm(self.hdim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hdim, self.hdim),
            torch.nn.LayerNorm(self.hdim),
            torch.nn.ReLU(),
        )
        if self.out_type == 0:
            self.out_dim = self.nu * self.T
        elif self.out_type == 1:
            self.out_dim = self.nx * self.T
        elif self.out_type == 2:
            self.out_dim = (self.nx + self.nu) * self.T
        elif self.out_type == 3:
            self.out_dim = (self.nq) * self.T

        self.model.add_module("out", torch.nn.Linear(self.hdim, self.out_dim))

    def forward(self, x):
        """
        compute the trajectory given state x
        Args:
            x (tensor bsz x nx): input state
        Returns:
            states (tensor bsz x T x nx): nominal states or None
            actions (tensor bsz x T x nu): nominal actions or None
        """
        if self.out_type == 0:
            actions = self.model(x)
            actions = actions.view(-1, self.T, self.nu)
            states = None
        elif self.out_type == 1:
            states = self.model(x)
            states = states.view(-1, self.T, self.nx)
            actions = None
        elif self.out_type == 2:
            states = self.model(x)[:, : self.nx * self.T]
            states = states.view(-1, self.T, self.nx)
            actions = self.model(x)[:, self.nx * self.T:]
            actions = actions.view(-1, self.T, self.nu)
        elif self.out_type == 3:
            pos = self.model(x)
            vel = (pos[:, 1:] - pos[:, :-1]) / self.dt
            vel = torch.cat([vel, vel[:, -1:]], dim=1)
            states = torch.cat([pos, vel], dim=-1).view(-1, self.T, self.nx)
            actions = None
        return states, actions


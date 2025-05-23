import numpy as nq
import torch
import torch.nn as nn
import torch.autograd as autograd
import qpth.al_utils as al_utils
import ipdb
# from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
# from policy_utils import SinusoidalPosEmb
import time
from deq_layer_utils import GradNormLayer, ScaleMultiplyLayer, DEQFixedPointLayer, GatedResidual

# POSSIBLE OUTPUT TYPES OF DEQ LAYER
# 0: action prediction u[0]->u[T-1]
# 1: state prediction x[0]->x[T-1], would be x[1]->x[T-1] only if using DEQLayer
# 2: state prediction x[0]->x[T-1] and control prediction u[0]->u[T-1]


# Things to test architecturally
# 1. Dual inputs and why it isn't helping
# 2. No hidden state propagation
# 3. Delta outputs and why it isn't helping - maybe because of the way we are computing the loss, or the way we are computing the gradients, or inputs are not 'right'
#      3.1. Probably because of different scale requirements between 1st iteration and subsequent iterations.
#      3.2. diffusion models predict at a fixed scale and we scale it depending on the 'expected' noise. How to make sure the scales are similar. 
#           3.1.1 Let the network predict scale
#           3.1.2 Fix decreasing scale for initial few iterations and then let the network predict the scale or fix a small scale depending on the precision requirements.
#      3.3. This can help us think more deeply about the tradeoffs between diffusion models and DEQs. : Especially if you have constraints to think about. 
# 4. why the network worsens the mpc outputs - does using the same as the mpc outputs make it worse - does interpolating between the gt and network output worsen things - likewise for the mpc outputs and network outputs
# 5. Fundamental architectural changes - GNNs, GATs, etc. - UNet like architectures - skip connections, etc. - take inspiration from diffusion models stuff.
# 6. Learning rate schedules could probably also help.
# 7. Also need to test against the interior point and ilqr based differentiable mpc methods
class DEQLayer(torch.nn.Module):
    '''
    Base class for different DEQ architectures, child classes define `forward`, `setup_input_layer` and `setup_input_layer`
    '''
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.nq = args.nq
        self.dt = env.dt
        self.T = args.T
        self.hdim = args.hdim
        self.deq_iter = args.deq_iter
        self.layer_type = args.layer_type
        self.inp_type = ""  # args.inp_type
        self.out_type = args.deq_out_type
        self.loss_type = args.loss_type
        self.deq_reg = args.deq_reg
        self.kernel_width = args.kernel_width
        self.pooling = args.pooling
        self.deq_expand = 4
        self.kernel_width_out = 1
        self.num_groups = 4
        self.pos_scale = args.pos_scale
        self.vel_scale = args.vel_scale

        self.mulnogradlayer = ScaleMultiplyLayer()
        self.setup_input_layer()
        self.setup_deq_layer()
        self.setup_output_layer()

        self.DEQfplayer = DEQFixedPointLayer(self.deq_layer, **vars(args))

    # TO BE OVERRIDEN
    def forward(self, in_obs_dict, in_aux_dict):
        """
        compute the policy output for the given observation input and feedback input 
        """
        obs, x_prev, z, iter = in_obs_dict["o"], in_aux_dict["x"], in_aux_dict["z"], in_aux_dict["iter"]
        # if (x_prev.shape[1] != self.T - 1):  # handle the case of orginal DEQLayer not predicting current state
        #     x_prev = x_prev[:, 1:]
        bsz = obs.shape[0]
        _obs = obs.reshape(bsz,1,self.nx)
        # _input = torch.cat([_obs, x_prev], dim=-2).reshape(bsz, -1)
        _input = x_prev.reshape(bsz, -1)
        _input1 = self.input_layer(_input, _obs, self.embedding_params[0][None]*0)
        # z_out = self.deq_layer(_input1, z)
        z_out, logs = self.DEQfplayer(_input1, z, iter, build_graph=True)
        dx_ref = self.output_layer(z_out)
        # ipdb.set_trace()
        dx_ref = dx_ref.reshape(-1, self.T - 1, self.nx)
        vel_ref = dx_ref[..., self.nq:]#self.mulnogradlayer(dx_ref[..., self.nq:], self.vel_scale)
        dx_ref = dx_ref[..., :self.nq]*self.dt#self.mulnogradlayer(dx_ref[..., :self.nq], self.pos_scale)# self.dt
        x_ref = torch.cat([dx_ref + x_prev[..., :1, :self.nq], vel_ref], dim=-1)
        # x_ref = torch.cat([x_prev[..., 1:, :self.nq] + dx_ref, vel_ref + x_prev[..., 1:, self.nq:]], dim=-1)
        # x_ref = torch.cat([dx_ref + _obs[:, :, :self.nq], vel_ref], dim=-1)
        x_ref = torch.cat([_obs, x_ref], dim=-2)
        u_ref = torch.zeros_like(x_ref[..., :self.nu])
        
        out_mpc_dict = {"x_t": obs, "x_ref": x_ref, "u_ref": u_ref}
        out_aux_dict = {"x": x_ref[:,:], "u": u_ref, "z": z_out, "iter": iter,
                        "deq_fwd_err": logs["forward_rel_err"], "deq_fwd_steps": logs["forward_steps"], "jac_loss": logs["jac_loss"]}
        return out_mpc_dict, out_aux_dict

    def input_layer(self, x, obs, emb):
        if self.layer_type == "mlp":
            inp = self.inp_layer(x)
        elif self.layer_type == "gcn":
            # ipdb.set_trace()
            t = self.time_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)#[:,1:]
            emb = emb.repeat(x.shape[0], 1, 1)
            x = x.reshape(-1, self.T, self.nx)[:,1:]
            x_emb = self.node_encoder(x)
            x0_emb = self.x0_encoder(obs[:, 0]).unsqueeze(1).repeat(1, self.T-1, 1)  #TODO switch case for out_type
            inp = torch.cat([x_emb, x0_emb, t, emb], dim=-1).transpose(1, 2).reshape(-1, self.hdim*4, self.T-1)
            inp = self.input_encoder(inp)
        elif self.layer_type == "gat":
            NotImplementedError
        return inp

    def deq_layer(self, x, z):
        if self.layer_type == "mlp":
            y = self.fcdeq1(z)
            y = self.reludeq1(y)
            y = self.lndeq1(y)
            out = self.lndeq3(self.reludeq2(
                z + self.lndeq2(x + self.fcdeq2(y))))
        elif self.layer_type == "gcn":
            z = z.view(-1, self.T-1, self.hdim).permute(0, 2, 1)
            y = self.convdeq1(z)
            y = self.mishdeq1(y)
            y = self.gndeq1(y)
            out = self.gndeq3(self.mishdeq2(
                z + self.gndeq2(x + self.convdeq2(y))))
            out = out.permute(0, 2, 1)#.reshape(-1, self.hdim)
        elif self.layer_type == "gat":
            NotImplementedError
        return out

    def output_layer(self, z):
        if self.layer_type == "mlp":
            out = self.out_layer(z)
            # out = self.gradnorm(out)
            return out
        elif self.layer_type == "gcn":
            bsz = z.shape[0]
            z = z.view(-1, self.T-1, self.hdim).permute(0, 2, 1)
            out = self.out_layer(z).permute(0, 2, 1)
            # out = self.gradnorm(out)
            # out = self.gradnorm(out.reshape(bsz, -1)).view(bsz, self.T-1, -1)
            return out
            # z = self.convout(z)
            # z = self.mishout(z)
            # z = self.gnout(z)
            # return self.final_layer(z).permute(0, 2, 1)#[:, 1:]
        elif self.layer_type == "gat":
            NotImplementedError

    def init_z(self, bsz):
        if self.layer_type == "mlp":
            return torch.zeros(bsz, self.hdim, dtype=torch.float32, device=self.args.device)
        elif self.layer_type == "gcn":
            return torch.zeros(bsz, self.T-1, self.hdim, dtype=torch.float32, device=self.args.device)
        elif self.layer_type == "gat":
            NotImplementedError

    # TO BE OVERRIDEN
    def setup_input_layer(self):
        self.in_dim = self.nx + self.nx * (self.T - 1) # current state and state prediction
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hdim),
                torch.nn.LayerNorm(self.hdim),
                # torch.nn.ReLU()
            )
            # self.fc_inp = torch.nn.Linear(self.nx + self.nq*self.T, self.hdim)
            # self.ln_inp = torch.nn.LayerNorm(self.hdim)
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
                nn.Conv1d(self.hdim*4, self.hdim*4, self.kernel_width, padding='same'),
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
        elif self.layer_type == "gat":
            NotImplementedError

    def setup_deq_layer(
        self,
    ):
        if self.layer_type == "mlp":
            self.embedding_params = torch.nn.Parameter(torch.zeros(self.deq_iter, self.hdim))
            expanddim = self.hdim*self.deq_expand
            self.fcdeq1 = torch.nn.Linear(self.hdim, expanddim)
            self.lndeq1 = torch.nn.LayerNorm(expanddim)
            self.reludeq1 = torch.nn.ReLU()
            self.fcdeq2 = torch.nn.Linear(expanddim, self.hdim)
            self.lndeq2 = torch.nn.LayerNorm(self.hdim)
            self.reludeq2 = torch.nn.ReLU()
            self.lndeq3 = torch.nn.LayerNorm(self.hdim)
        elif self.layer_type == "gcn":
            self.embedding_params = torch.nn.Parameter(torch.zeros(self.deq_iter, self.T-1, self.hdim))
            self.convdeq1 = torch.nn.Conv1d(
                self.hdim, self.hdim*self.deq_expand, self.kernel_width, padding='same')
            self.convdeq2 = torch.nn.Conv1d(
                self.hdim*self.deq_expand, self.hdim, self.kernel_width, padding='same')
            # self.mishdeq1 = torch.nn.Mish()#
            self.mishdeq1 = torch.nn.ReLU()
            # self.mishdeq2 = torch.nn.Mish() #
            self.mishdeq2 = torch.nn.ReLU()
            self.gndeq1 = torch.nn.GroupNorm(
                self.num_groups, self.hdim*self.deq_expand)
            self.gndeq2 = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.gndeq3 = torch.nn.GroupNorm(self.num_groups, self.hdim)
        elif self.layer_type == "gat":
            NotImplementedError

    # TO BE OVERRIDEN
    def setup_output_layer(self):  
        self.out_dim = self.nx * (self.T-1)  # state prediction
        if self.layer_type == "mlp":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hdim, self.out_dim)
            )
            self.gradnorm = GradNormLayer(self.out_dim)
        elif self.layer_type == "gcn":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Conv1d(self.hdim, self.hdim, self.kernel_width, padding='same'),
                torch.nn.GroupNorm(self.num_groups, self.hdim),
                torch.nn.ReLU(),
                torch.nn.Conv1d(self.hdim, self.nx, self.kernel_width_out, padding='same'),
            )
            self.gradnorm = GradNormLayer(self.out_dim)
            ###################### Implement GradNormLayer for GCN ######################
            # self.convout = torch.nn.Conv1d(
            #     self.hdim, self.hdim, self.kernel_width, padding='same')
            # self.gnout = torch.nn.GroupNorm(self.num_groups, self.hdim)
            # self.mishout = torch.nn.Mish()
            # self.final_layer = torch.nn.Conv1d(
            #     self.hdim, self.nx, self.kernel_width_out, padding='same')
        elif self.layer_type == "gat":
            NotImplementedError

class DEQLayerMem(DEQLayer):
    '''
    Base class for different DEQ architectures, child classes define `forward`, `setup_input_layer` and `setup_input_layer`
    '''
    def __init__(self, args, env):
        super().__init__(args, env)
        self.setup_mem_layer()

    # TO BE OVERRIDEN
    def forward(self, in_obs_dict, in_aux_dict, mem_update=True):
        """
        compute the policy output for the given observation input and feedback input 
        """
        obs, x_prev, z, iter, mem, old_mem = in_obs_dict["o"], in_aux_dict["x"], in_aux_dict["z"], in_aux_dict["iter"], in_aux_dict["mem"], in_aux_dict["old_mem"]
        # if (x_prev.shape[1] != self.T - 1):  # handle the case of orginal DEQLayer not predicting current state
        #     x_prev = x_prev[:, 1:]
        bsz = obs.shape[0]
        _obs = obs.reshape(bsz,1,self.nx)
        # _input = torch.cat([_obs, x_prev], dim=-2).reshape(bsz, -1)
        _input = x_prev.reshape(bsz, -1)
        if mem_update:
            _input1 = self.input_layer(_input, _obs, self.embedding_params[iter][None]*0, mem)
        else:
            _input1 = self.input_layer(_input, _obs, self.embedding_params[iter][None]*0, old_mem)
        # z_out = self.deq_layer(_input1, z)
        z_out, logs = self.DEQfplayer(_input1, z, iter, build_graph=mem_update)
        dx_ref = self.output_layer(z_out)
        if mem_update:
            old_mem = mem
            mem = self.mem_layer(mem, z_out)
        # ipdb.set_trace()
        dx_ref = dx_ref.reshape(-1, self.T - 1, self.nx)
        vel_ref = dx_ref[..., self.nq:]#self.mulnogradlayer(dx_ref[..., self.nq:], self.vel_scale)
        dx_ref = dx_ref[..., :self.nq]*self.dt#self.mulnogradlayer(dx_ref[..., :self.nq], self.pos_scale)# self.dt
        x_ref = torch.cat([dx_ref + x_prev[..., :1, :self.nq], vel_ref], dim=-1)
        # x_ref = torch.cat([x_prev[..., 1:, :self.nq] + dx_ref, vel_ref + x_prev[..., 1:, self.nq:]], dim=-1)
        # x_ref = torch.cat([dx_ref + _obs[:, :, :self.nq], vel_ref], dim=-1)
        x_ref = torch.cat([_obs, x_ref], dim=-2)
        u_ref = torch.zeros_like(x_ref[..., :self.nu])
        
        out_mpc_dict = {"x_t": obs, "x_ref": x_ref, "u_ref": u_ref}
        out_aux_dict = {"x": x_ref[:,:], "u": u_ref, "z": z_out, "mem": mem, "iter": iter, "old_mem": old_mem,
                        "deq_fwd_err": logs["forward_rel_err"], "deq_fwd_steps": logs["forward_steps"], "jac_loss": logs["jac_loss"]}
        return out_mpc_dict, out_aux_dict

    def input_layer(self, x, obs, emb, mem):
        if self.layer_type == "mlp":
            inp = self.inp_layer(x)
        elif self.layer_type == "gcn":
            # ipdb.set_trace()
            bsz = x.shape[0]
            t = self.time_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)#[:,1:]
            mem = mem.reshape(-1, self.T-1, self.hdim)
            # ipdb.set_trace()
            emb = emb.repeat(x.shape[0], 1, 1)
            x = x.reshape(-1, self.T, self.nx)[:,1:]
            x_emb = self.node_encoder(x)
            x0_emb = self.x0_encoder(obs[:, 0]).unsqueeze(1).repeat(1, self.T-1, 1)  #TODO switch case for out_type
            inp = torch.cat([x_emb, x0_emb, t, emb, mem], dim=-1).transpose(1, 2).reshape(bsz, -1, self.T-1)
            inp = self.input_encoder(inp)
        elif self.layer_type == "gat":
            NotImplementedError
        return inp

    def init_mem(self, bsz):
        if self.layer_type == "mlp":
            return torch.zeros(bsz, self.hdim, dtype=torch.float32, device=self.args.device)
        elif self.layer_type == "gcn":
            return torch.zeros(bsz, self.T-1, self.hdim, dtype=torch.float32, device=self.args.device)
        elif self.layer_type == "gat":
            NotImplementedError

    def init_z(self, bsz):
        if self.layer_type == "mlp":
            return torch.zeros(bsz, self.hdim, dtype=torch.float32, device=self.args.device)
        elif self.layer_type == "gcn":
            return torch.zeros(bsz, self.T-1, self.hdim, dtype=torch.float32, device=self.args.device)
        elif self.layer_type == "gat":
            NotImplementedError

    # TO BE OVERRIDEN
    def setup_input_layer(self):
        self.in_dim = self.nx + self.nx * (self.T - 1) # current state and state prediction
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hdim),
                torch.nn.LayerNorm(self.hdim),
                # torch.nn.ReLU()
            )
            # self.fc_inp = torch.nn.Linear(self.nx + self.nq*self.T, self.hdim)
            # self.ln_inp = torch.nn.LayerNorm(self.hdim)
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
                # torch.nn.Mish(),
            )

            self.x0_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.ReLU()
                # torch.nn.Mish()
            )

            self.input_encoder = nn.Sequential(
                nn.Conv1d(self.hdim*5, self.hdim*4, self.kernel_width, padding='same'),
                nn.ReLU(),
                # torch.nn.Mish(),
                nn.Conv1d(self.hdim*4, self.hdim, self.kernel_width, padding='same'),
                nn.GroupNorm(self.num_groups, self.hdim),
                # nn.Mish()
            )

            self.global_pooling = {
                "max": torch.max,
                "mean": torch.mean,
                "sum": torch.sum
            }[self.pooling]
        elif self.layer_type == "gat":
            NotImplementedError

    def mem_layer(self, mem, z):
        if self.layer_type == "mlp":
            mem = self.gru(mem, z)
        elif self.layer_type == "gcn":
            # z = z.view(-1, self.T-1, self.hdim)
            # mem = self.mem_ln1(mem)
            mem = self.mem_gr1(mem, z)
            # mem = self.mem_ln2(mem)
            mem = self.mem_gr2(mem, z)
        elif self.layer_type == "gat":
            NotImplementedError
        return mem

    def setup_mem_layer(self,):
        # compute a new memory state gru style
        # self.mem_ln1 = nn.LayerNorm(self.hdim, eps=1e-3)
        self.mem_gr1 = GatedResidual(self.hdim)
        # self.mem_ln2 = nn.LayerNorm(self.hdim, eps=1e-3)
        self.mem_gr2 = GatedResidual(self.hdim)



class DEQLayerDelta(DEQLayer):
    '''
    Base class for different DEQ architectures, child classes define `forward`, `setup_input_layer` and `setup_input_layer`
    '''
    def __init__(self, args, env):
        super().__init__(args, env)
        self.max_scale = args.max_scale + 0.1
        # self.scales = [self.max_scale/(2**(i+1)) for i in range(3)] + [self.max_scale/10 for i in range(3, args.deq_iter)]
        # self.scales = [torch.ones(self.T-1, self.nx).to(self.max_scale) for i in range(args.deq_iter)]
        # Define scales as nn.Parameter instead filled with ones
        self.scales = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(self.T-1, self.nx).to(self.max_scale)) for i in range(args.deq_iter)])
        self.embedding_params = torch.nn.Parameter(torch.zeros(args.deq_iter, self.hdim))
    # TO BE OVERRIDEN
    def forward(self, in_obs_dict, in_aux_dict):
        """
        compute the policy output for the given observation input and feedback input 
        """
        obs, x_prev, z, iter = in_obs_dict["o"], in_aux_dict["x"], in_aux_dict["z"], in_aux_dict["iter"]
        # if (x_prev.shape[1] != self.T - 1):  # handle the case of orginal DEQLayer not predicting current state
        #     x_prev = x_prev[:, 1:]
        bsz = obs.shape[0]
        _obs = obs.reshape(bsz,1,self.nx)
        # _input = torch.cat([_obs, x_prev], dim=-2).reshape(bsz, -1)
        _input = x_prev.reshape(bsz, -1)
        _input1 = self.input_layer(_input)
        z_out = self.deq_layer(_input1, z + self.embedding_params[iter][None])
        dx_ref, s = self.output_layer(z_out, iter)
        # ipdb.set_trace()
        dx_ref = dx_ref.view(-1, self.T - 1, self.nx)
        vel_ref = dx_ref[..., self.nq:]
        dx_ref = dx_ref[..., :self.nq] * self.dt
        # x_ref = torch.cat([dx_ref + x_prev[..., :1, :self.nq], vel_ref], dim=-1)
        x_ref = torch.cat([x_prev[..., 1:, :self.nq] + dx_ref, vel_ref + x_prev[..., 1:, self.nq:]], dim=-1)
        # x_ref = torch.cat([dx_ref + _obs[:, :, :self.nq], vel_ref], dim=-1)
        x_ref = torch.cat([_obs, x_ref], dim=-2)
        u_ref = torch.zeros_like(x_ref[..., :self.nu])
        
        out_mpc_dict = {"x_t": obs, "x_ref": x_ref, "u_ref": u_ref, "s": s}
        out_aux_dict = {"x": x_ref[:,:], "u": u_ref, "z": z_out}
        return out_mpc_dict, out_aux_dict

    def output_layer(self, z, iter):
        if self.layer_type == "mlp":
            out = self.out_layer(z)
            out = self.gradnorm(out)
            out = out
            # scale = self.scale_layer(z)
            scale = self.scales[iter].clone()
            scale[:, :self.nq] = scale[:, :self.nq] / self.dt
            scale = scale.reshape(-1)[None].repeat(out.shape[0], 1)#, self.T-1)
            out = self.scale_multiply(out, scale)
            return out, scale
        elif self.layer_type == "gcn":
            z = z.view(-1, self.T, self.hdim)
            z = self.convout(z.permute(0, 2, 1))
            z = self.mishout(z)
            z = self.gnout(z)
            return self.final_layer(z).permute(0, 2, 1)[:, 1:]
        elif self.layer_type == "gat":
            NotImplementedError

    # TO BE OVERRIDEN
    def setup_output_layer(self):  
        self.out_dim = self.nx * (self.T-1)  # state prediction
        if self.layer_type == "mlp":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hdim, self.out_dim)
            )
            self.gradnorm = GradNormLayer(self.out_dim)
            # self.scale_layer = torch.nn.Sequential(
            #     torch.nn.Linear(self.hdim, self.hdim),
            #     torch.nn.LayerNorm(self.hdim),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(self.hdim, 1),
            #     torch.nn.Softplus())
            self.scale_multiply = ScaleMultiplyLayer()

        elif self.layer_type == "gcn":
            self.convout = torch.nn.Conv1d(
                self.hdim, self.hdim, self.kernel_width)
            self.gnout = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.mishout = torch.nn.Mish()
            self.final_layer = torch.nn.Conv1d(
                self.hdim, self.nq, self.kernel_width_out)
        elif self.layer_type == "gat":
            NotImplementedError

class DEQLayerHistoryState(DEQLayer):
    '''
    DEQ layer takes state history, outputs current state and state prediction
    '''
    def __init__(self, args, env):
        self.H = args.H  # number of history steps (including current state)
        super().__init__(args, env)
    
    def forward(self, in_obs_dict, in_aux_dict):
        """
        compute the policy output for the given observation input and feedback input 
        """
        obs, x, z, iter = in_obs_dict["o"], in_aux_dict["x"], in_aux_dict["z"], in_aux_dict["iter"]
        bsz = obs.shape[0]
        _obs = obs.reshape(bsz, -1)
        _x = x.reshape(bsz, -1)
        # _input = torch.cat([_obs, _x], dim=-1)
        _input = _x
        _input1 = self.input_layer(_input, obs, iter, z)
        z_out = self.deq_layer(_input1, z)
        _dx_ref = self.output_layer(z_out)

        dx_ref = _dx_ref.view(-1, self.T, self.nx)
        vel_ref = dx_ref[..., self.nq:]
        dx_ref = dx_ref[..., :self.nq] * self.dt
        x_ref = torch.cat([dx_ref + x[..., :self.nq], vel_ref], dim=-1)  # why predict delta_x and v instead of x and v or delta_x and delta_v or x and delta_v
        # x_ref = torch.cat([obs[:,None], x_ref[:, 1:]], dim=-2)
        u_ref = torch.zeros_like(x_ref[..., :self.nu])
        
        out_mpc_dict = {"x_t": x_ref[:,0], "x_ref": x_ref, "u_ref": u_ref}
        out_aux_dict = {"x": x_ref, "u": u_ref, "z": z_out}
        return out_mpc_dict, out_aux_dict
    
    def setup_input_layer(self):
        self.in_dim = self.nx * self.H + self.nx * self.T # external input and aux input
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hdim),
                torch.nn.LayerNorm(self.hdim),
            )
        elif self.layer_type == "gcn":
            # Get sinusoidal embeddings for the time steps
            # self.time_encoder = nn.Sequential(
            #     SinusoidalPosEmb(self.hdim),
            #     nn.Linear(self.hdim, self.hdim*4),
            #     nn.Mish(),
            #     nn.Linear(self.hdim*4, self.hdim),
            #     nn.LayerNorm(self.hdim)
            #     )
            self.time_emb = torch.nn.Parameter(torch.randn(self.T + self.H, self.hdim))
            # Get the node embeddings
            self.node_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
            )

            self.obs_encoder = nn.Sequential(
                nn.Conv1d(self.hdim*2, self.hdim*3, self.kernel_width, padding='same'),
                nn.Mish(),
                nn.Conv1d(self.hdim*3, self.hdim, self.kernel_width, padding='same'),
                nn.GroupNorm(self.num_groups, self.hdim),
                # nn.Mish()
            )
            self.z0_encoder = nn.Sequential(
                nn.Linear(self.hdim, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
            )
            self.x0_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
            )

            self.input_encoder = nn.Sequential(
                nn.Conv1d(self.hdim*3, self.hdim*3, self.kernel_width, padding='same'),
                nn.Mish(),
                nn.Conv1d(self.hdim*3, self.hdim, self.kernel_width, padding='same'),
                nn.GroupNorm(self.num_groups, self.hdim),
                # nn.Mish()
            )

            self.global_pooling = {
                "max": torch.max,
                "mean": torch.mean,
                "sum": torch.sum
            }[self.pooling]
        elif self.layer_type == "gat":
            NotImplementedError

    def input_layer(self, x, obs, iter, z):
        if self.layer_type == "mlp":
            bsz = x.shape[0]
            x = x.reshape(bsz, -1)
            obs = obs.reshape(bsz, -1)
            x = torch.cat([obs, x], dim=-1)
            inp = self.inp_layer(x)
        elif self.layer_type == "gcn":
            t = self.time_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
            tH = t[:, :self.H]
            tT = t[:, self.H:]
            x = x.reshape(-1, self.T, self.nx)
            obs = obs.reshape(-1, self.H, self.nx)
            obs_emb = self.node_encoder(obs)
            obs_inp = torch.cat([obs_emb, tH], dim=-1).transpose(1, 2).reshape(-1, self.hdim*2, self.H)
            obs_inp = self.obs_encoder(obs_inp)

            x_emb = self.node_encoder(x)
            if False:#iter > 0:
                zest = z[0]
                x0_emb = self.z0_encoder(zest[:, -1]).unsqueeze(1).repeat(1, self.T, 1)
            else:
                # x0_emb = self.x0_encoder(obs[:, -1]).unsqueeze(1).repeat(1, self.T, 1)
                x0_emb = obs_inp[:, None, :, -1].repeat(1, self.T, 1)
            inp = torch.cat([x_emb, x0_emb, tT], dim=-1).transpose(1, 2).reshape(-1, self.hdim*3, self.T)
            inp = self.input_encoder(inp)
            inp = (obs_inp, inp)
        elif self.layer_type == "gat":
            NotImplementedError
        return inp

    def output_layer(self, z):
        if self.layer_type == "mlp":
            out = self.out_layer(z)
            # out = self.gradnorm(out)
            return out
        elif self.layer_type == "gcn":
            z = z[1]
            bsz = z.shape[0]
            z = z.view(bsz, self.T, self.hdim).permute(0, 2, 1)
            out = self.out_layer(z).permute(0, 2, 1)
            # out = self.gradnorm(out)
            # out = self.gradnorm(out.reshape(bsz, -1)).view(bsz, self.T-1, -1)
            return out
        elif self.layer_type == "gat":
            NotImplementedError

    def setup_output_layer(self):  
        self.out_dim = self.nx * self.T
        if self.layer_type == "mlp":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hdim, self.out_dim)
            )
            self.gradnorm = GradNormLayer(self.out_dim)
        elif self.layer_type == "gcn":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Conv1d(self.hdim, self.hdim, self.kernel_width, padding='same'),
                torch.nn.GroupNorm(self.num_groups, self.hdim),
                torch.nn.Mish(),
                torch.nn.Conv1d(self.hdim, self.nx, self.kernel_width_out, padding='same'),
            )
            self.gradnorm = GradNormLayer(self.out_dim)
            ###################### Implement GradNormLayer for GCN ######################
            # self.convout = torch.nn.Conv1d(
            #     self.hdim, self.hdim, self.kernel_width, padding='same')
            # self.gnout = torch.nn.GroupNorm(self.num_groups, self.hdim)
            # self.mishout = torch.nn.Mish()
            # self.final_layer = torch.nn.Conv1d(
            #     self.hdim, self.nx, self.kernel_width_out, padding='same')
        elif self.layer_type == "gat":
            NotImplementedError

    def deq_layer(self, x, z):
        if self.layer_type == "mlp":
            y = self.fcdeq1(z)
            y = self.reludeq1(y)
            y = self.lndeq1(y)
            out = self.lndeq3(self.reludeq2(
                z + self.lndeq2(x + self.fcdeq2(y))))
        elif self.layer_type == "gcn":
            xest, xpred = x
            zest, zpred = z

            ### Estimation part
            zest = zest.view(-1, self.H, self.hdim).permute(0, 2, 1)
            yest = self.convdeq1_est(zest)
            yest = self.mishdeq1_est(yest)
            yest = self.gndeq1_est(yest)
            outest = self.gndeq3_est(self.mishdeq2_est(
                zest + self.gndeq2_est(xest + self.convdeq2_est(yest))))
            outest = outest.permute(0, 2, 1)
            zest_out = self.lin_zest_out(outest[:, -1]).unsqueeze(2).repeat(1, 1, self.T)

            ### Prediction part
            zpred = zpred.view(-1, self.T, self.hdim).permute(0, 2, 1)
            y = self.convdeq1(zpred)
            y = self.mishdeq1(y)
            y = self.gndeq1(y)
            outpred = self.gndeq3(self.mishdeq2(
                zpred + self.gndeq2(xpred + zest_out + self.convdeq2(y))))
            outpred = outpred.permute(0, 2, 1)#.reshape(-1, self.hdim)
            out = (outest, outpred)
        elif self.layer_type == "gat":
            NotImplementedError
        return out


    def init_z(self, bsz):
        if self.layer_type == "mlp":
            return torch.zeros(bsz, self.hdim, dtype=torch.float32, device=self.args.device)
        elif self.layer_type == "gcn":
            z1 = torch.zeros(bsz, self.H, self.hdim, dtype=torch.float32, device=self.args.device)
            z2 = torch.zeros(bsz, self.T, self.hdim, dtype=torch.float32, device=self.args.device)
            return (z1, z2)
        elif self.layer_type == "gat":
            NotImplementedError


    def setup_deq_layer(
        self,
    ):
        if self.layer_type == "mlp":
            self.embedding_params = torch.nn.Parameter(torch.zeros(self.deq_iter, self.hdim))
            expanddim = self.hdim*self.deq_expand
            self.fcdeq1 = torch.nn.Linear(self.hdim, expanddim)
            self.lndeq1 = torch.nn.LayerNorm(expanddim)
            self.reludeq1 = torch.nn.ReLU()
            self.fcdeq2 = torch.nn.Linear(expanddim, self.hdim)
            self.lndeq2 = torch.nn.LayerNorm(self.hdim)
            self.reludeq2 = torch.nn.ReLU()
            self.lndeq3 = torch.nn.LayerNorm(self.hdim)
        elif self.layer_type == "gcn":
            self.embedding_params = torch.nn.Parameter(torch.zeros(self.deq_iter, self.T-1, self.hdim))
            ### trajectory estimation
            self.convdeq1_est = torch.nn.Conv1d(
                self.hdim, self.hdim*self.deq_expand, self.kernel_width, padding='same')
            self.convdeq2_est = torch.nn.Conv1d(
                self.hdim*self.deq_expand, self.hdim, self.kernel_width, padding='same')
            self.mishdeq1_est = torch.nn.Mish()
            self.mishdeq2_est = torch.nn.Mish()
            self.gndeq1_est = torch.nn.GroupNorm(
                self.num_groups, self.hdim*self.deq_expand)
            self.gndeq2_est = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.gndeq3_est = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.lin_zest_out = torch.nn.Linear(self.hdim, self.hdim)

            ### trajectory prediction
            self.convdeq1 = torch.nn.Conv1d(
                self.hdim, self.hdim*self.deq_expand, self.kernel_width, padding='same')
            self.convdeq2 = torch.nn.Conv1d(
                self.hdim*self.deq_expand, self.hdim, self.kernel_width, padding='same')
            self.mishdeq1 = torch.nn.Mish()
            self.mishdeq2 = torch.nn.Mish()
            self.gndeq1 = torch.nn.GroupNorm(
                self.num_groups, self.hdim*self.deq_expand)
            self.gndeq2 = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.gndeq3 = torch.nn.GroupNorm(self.num_groups, self.hdim)
        elif self.layer_type == "gat":
            NotImplementedError

class DEQLayerHistoryStateEstPred(DEQLayer):
    '''
    DEQ layer takes state history, outputs current state and state prediction
    '''
    def __init__(self, args, env):
        self.H = args.H  # number of history steps (including current state)
        super().__init__(args, env)
    
    def forward(self, in_obs_dict, in_aux_dict):
        """
        compute the policy output for the given observation input and feedback input 
        """
        obs, x, z, iter = in_obs_dict["o"], in_aux_dict["x"], in_aux_dict["z"], in_aux_dict["iter"]
        x_est = in_aux_dict["x_est"]
        bsz = obs.shape[0]
        _obs = obs.reshape(bsz, -1)
        _x = x.reshape(bsz, -1)
        # _input = torch.cat([_obs, _x], dim=-1)
        _input = _x
        _input1 = self.input_layer(_input, x_est, obs, iter, z)
        z_out = self.deq_layer(_input1, z)
        _dx_ref = self.output_layer(z_out)
        _dx_refest, _dx_refpred = _dx_ref
        
        _dx_ref = _dx_refest
        dx_ref = _dx_ref.view(-1, self.H, self.nx)
        dvel_ref = dx_ref[..., self.nq:]
        dx_ref = dx_ref[..., :self.nq] * self.dt
        x_est = torch.cat([dx_ref+obs[..., :self.nq], dvel_ref+obs[..., self.nq:]], dim=-1)  # why predict delta_x and v instead of x and v or delta_x and delta_v or x and delta_v

        _dx_ref = _dx_refpred
        dx_ref = _dx_ref.view(-1, self.T, self.nx)
        vel_ref = dx_ref[..., self.nq:]
        dx_ref = dx_ref[..., :self.nq] * self.dt
        x_ref = torch.cat([dx_ref + x[..., :self.nq], vel_ref], dim=-1)  # why predict delta_x and v instead of x and v or delta_x and delta_v or x and delta_v
        # x_ref = torch.cat([obs[:,None], x_ref[:, 1:]], dim=-2)
        u_ref = torch.zeros_like(x_ref[..., :self.nu])
        
        out_mpc_dict = {"x_t": x_ref[:,0], "x_ref": x_ref, "u_ref": u_ref, "x_est": x_est}
        out_aux_dict = {"x": x_ref, "u": u_ref, "z": z_out, "x_est": x_est}
        return out_mpc_dict, out_aux_dict
    
    def setup_input_layer(self):
        self.in_dim = self.nx * self.H + self.nx * self.T # external input and aux input
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hdim),
                torch.nn.LayerNorm(self.hdim),
            )
        elif self.layer_type == "gcn":
            # Get sinusoidal embeddings for the time steps
            # self.time_encoder = nn.Sequential(
            #     SinusoidalPosEmb(self.hdim),
            #     nn.Linear(self.hdim, self.hdim*4),
            #     nn.Mish(),
            #     nn.Linear(self.hdim*4, self.hdim),
            #     nn.LayerNorm(self.hdim)
            #     )
            self.time_emb = torch.nn.Parameter(torch.randn(self.T + self.H, self.hdim))
            # Get the node embeddings
            self.node_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
            )

            self.obs_encoder = nn.Sequential(
                nn.Conv1d(self.hdim*3, self.hdim*3, self.kernel_width, padding='same'),
                nn.Mish(),
                nn.Conv1d(self.hdim*3, self.hdim, self.kernel_width, padding='same'),
                nn.GroupNorm(self.num_groups, self.hdim),
                # nn.Mish()
            )
            self.z0_encoder = nn.Sequential(
                nn.Linear(self.hdim, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
            )
            self.x0_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
            )

            self.input_encoder = nn.Sequential(
                nn.Conv1d(self.hdim*3, self.hdim*3, self.kernel_width, padding='same'),
                nn.Mish(),
                nn.Conv1d(self.hdim*3, self.hdim, self.kernel_width, padding='same'),
                nn.GroupNorm(self.num_groups, self.hdim),
                # nn.Mish()
            )

            self.global_pooling = {
                "max": torch.max,
                "mean": torch.mean,
                "sum": torch.sum
            }[self.pooling]
        elif self.layer_type == "gat":
            NotImplementedError

    def input_layer(self, x, x_est, obs, iter, z):
        if self.layer_type == "mlp":
            bsz = x.shape[0]
            x = x.reshape(bsz, -1)
            obs = obs.reshape(bsz, -1)
            x = torch.cat([obs, x], dim=-1)
            inp = self.inp_layer(x)
        elif self.layer_type == "gcn":
            t = self.time_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
            tH = t[:, :self.H]
            tT = t[:, self.H:]
            x = x.reshape(-1, self.T, self.nx)
            obs = obs.reshape(-1, self.H, self.nx)
            x_est = x_est.reshape(-1, self.H, self.nx)
            obs_emb = self.node_encoder(obs)
            x_est_emb = self.node_encoder(x_est)
            obs_inp = torch.cat([obs_emb, x_est_emb, tH], dim=-1).transpose(1, 2).reshape(-1, self.hdim*3, self.H)
            obs_inp = self.obs_encoder(obs_inp)

            x_emb = self.node_encoder(x)
            if True:#iter > 0:
                zest = z[0]
                x0_emb = self.z0_encoder(zest[:, -1]).unsqueeze(1).repeat(1, self.T, 1)
            else:
                # x0_emb = self.x0_encoder(obs[:, -1]).unsqueeze(1).repeat(1, self.T, 1)
                x0_emb = obs_inp[:, None, :, -1].repeat(1, self.T, 1)
            inp = torch.cat([x_emb, x0_emb, tT], dim=-1).transpose(1, 2).reshape(-1, self.hdim*3, self.T)
            inp = self.input_encoder(inp)
            inp = (obs_inp, inp)
        elif self.layer_type == "gat":
            NotImplementedError
        return inp

    def output_layer(self, z):
        if self.layer_type == "mlp":
            out = self.out_layer(z)
            # out = self.gradnorm(out)
            return out
        elif self.layer_type == "gcn":
            zest, zpred = z
            bsz = zpred.shape[0]
            zest = zest.view(bsz, self.H, self.hdim).permute(0, 2, 1)
            zpred = zpred.view(bsz, self.T, self.hdim).permute(0, 2, 1)

            outest = self.out_layer1(zest).permute(0, 2, 1)
            outpred = self.out_layer(zpred).permute(0, 2, 1)
            out = (outest, outpred)
            # out = self.gradnorm(out)
            # out = self.gradnorm(out.reshape(bsz, -1)).view(bsz, self.T-1, -1)
            return out
        elif self.layer_type == "gat":
            NotImplementedError

    def setup_output_layer(self):  
        self.out_dim = self.nx * self.T
        if self.layer_type == "mlp":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hdim, self.out_dim)
            )
            self.gradnorm = GradNormLayer(self.out_dim)
        elif self.layer_type == "gcn":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Conv1d(self.hdim, self.hdim, self.kernel_width, padding='same'),
                torch.nn.GroupNorm(self.num_groups, self.hdim),
                torch.nn.Mish(),
                torch.nn.Conv1d(self.hdim, self.nx, self.kernel_width_out, padding='same'),
            )
            self.gradnorm = GradNormLayer(self.out_dim)
            self.out_layer1 = torch.nn.Sequential(
                torch.nn.Conv1d(self.hdim, self.hdim, self.kernel_width, padding='same'),
                torch.nn.GroupNorm(self.num_groups, self.hdim),
                torch.nn.Mish(),
                torch.nn.Conv1d(self.hdim, self.nx, self.kernel_width_out, padding='same'),
            )
            ###################### Implement GradNormLayer for GCN ######################
            # self.convout = torch.nn.Conv1d(
            #     self.hdim, self.hdim, self.kernel_width, padding='same')
            # self.gnout = torch.nn.GroupNorm(self.num_groups, self.hdim)
            # self.mishout = torch.nn.Mish()
            # self.final_layer = torch.nn.Conv1d(
            #     self.hdim, self.nx, self.kernel_width_out, padding='same')
        elif self.layer_type == "gat":
            NotImplementedError

    def deq_layer(self, x, z):
        if self.layer_type == "mlp":
            y = self.fcdeq1(z)
            y = self.reludeq1(y)
            y = self.lndeq1(y)
            out = self.lndeq3(self.reludeq2(
                z + self.lndeq2(x + self.fcdeq2(y))))
        elif self.layer_type == "gcn":
            xest, xpred = x
            zest, zpred = z

            ### Estimation part
            zest = zest.view(-1, self.H, self.hdim).permute(0, 2, 1)
            yest = self.convdeq1_est(zest)
            yest = self.mishdeq1_est(yest)
            yest = self.gndeq1_est(yest)
            outest = self.gndeq3_est(self.mishdeq2_est(
                zest + self.gndeq2_est(xest + self.convdeq2_est(yest))))
            outest = outest.permute(0, 2, 1)
            zest_out = self.lin_zest_out(outest[:, -1]).unsqueeze(2).repeat(1, 1, self.T)

            ### Prediction part
            zpred = zpred.view(-1, self.T, self.hdim).permute(0, 2, 1)
            y = self.convdeq1(zpred)
            y = self.mishdeq1(y)
            y = self.gndeq1(y)
            outpred = self.gndeq3(self.mishdeq2(
                zpred + self.gndeq2(xpred + zest_out + self.convdeq2(y))))
            outpred = outpred.permute(0, 2, 1)#.reshape(-1, self.hdim)
            out = (outest, outpred)
        elif self.layer_type == "gat":
            NotImplementedError
        return out


    def init_z(self, bsz):
        if self.layer_type == "mlp":
            return torch.zeros(bsz, self.hdim, dtype=torch.float32, device=self.args.device)
        elif self.layer_type == "gcn":
            z1 = torch.zeros(bsz, self.H, self.hdim, dtype=torch.float32, device=self.args.device)
            z2 = torch.zeros(bsz, self.T, self.hdim, dtype=torch.float32, device=self.args.device)
            return (z1, z2)
        elif self.layer_type == "gat":
            NotImplementedError


    def setup_deq_layer(
        self,
    ):
        if self.layer_type == "mlp":
            self.embedding_params = torch.nn.Parameter(torch.zeros(self.deq_iter, self.hdim))
            expanddim = self.hdim*self.deq_expand
            self.fcdeq1 = torch.nn.Linear(self.hdim, expanddim)
            self.lndeq1 = torch.nn.LayerNorm(expanddim)
            self.reludeq1 = torch.nn.ReLU()
            self.fcdeq2 = torch.nn.Linear(expanddim, self.hdim)
            self.lndeq2 = torch.nn.LayerNorm(self.hdim)
            self.reludeq2 = torch.nn.ReLU()
            self.lndeq3 = torch.nn.LayerNorm(self.hdim)
        elif self.layer_type == "gcn":
            self.embedding_params = torch.nn.Parameter(torch.zeros(self.deq_iter, self.T-1, self.hdim))
            ### trajectory estimation
            self.convdeq1_est = torch.nn.Conv1d(
                self.hdim, self.hdim*self.deq_expand, self.kernel_width, padding='same')
            self.convdeq2_est = torch.nn.Conv1d(
                self.hdim*self.deq_expand, self.hdim, self.kernel_width, padding='same')
            self.mishdeq1_est = torch.nn.Mish()
            self.mishdeq2_est = torch.nn.Mish()
            self.gndeq1_est = torch.nn.GroupNorm(
                self.num_groups, self.hdim*self.deq_expand)
            self.gndeq2_est = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.gndeq3_est = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.lin_zest_out = torch.nn.Linear(self.hdim, self.hdim)

            ### trajectory prediction
            self.convdeq1 = torch.nn.Conv1d(
                self.hdim, self.hdim*self.deq_expand, self.kernel_width, padding='same')
            self.convdeq2 = torch.nn.Conv1d(
                self.hdim*self.deq_expand, self.hdim, self.kernel_width, padding='same')
            self.mishdeq1 = torch.nn.Mish()
            self.mishdeq2 = torch.nn.Mish()
            self.gndeq1 = torch.nn.GroupNorm(
                self.num_groups, self.hdim*self.deq_expand)
            self.gndeq2 = torch.nn.GroupNorm(self.num_groups, self.hdim)
            self.gndeq3 = torch.nn.GroupNorm(self.num_groups, self.hdim)
        elif self.layer_type == "gat":
            NotImplementedError

class DEQLayerHistory(DEQLayer):
    '''
    DEQ layer takes state history, outputs state and action predictions
    '''
    def __init__(self, args, env):
        self.H = args.H  # number of history steps (including current state)
        super().__init__(args, env)        
    
    def forward(self, in_obs_dict, in_aux_dict):
        """
        compute the policy output for the given observation input and feedback input 
        """
        obs, x, u, z, iter = in_obs_dict["o"], in_aux_dict["x"], in_aux_dict["u"], in_aux_dict["z"], in_aux_dict["iter"]
        bsz = obs.shape[0]
        _obs = obs.reshape(bsz, -1)
        _x = x.reshape(bsz, -1)
        _u = u[:,:self.T-1].reshape(bsz, -1)  # remove last action
        _input = torch.cat([_obs, _x, _u], dim=-1)

        _input1 = self.input_layer(_input, obs)
        z_out = self.deq_layer(_input1, z)
        _dxu_ref = self.output_layer(z_out)

        dx_ref = _dxu_ref[..., :self.nx*self.T].reshape(-1, self.T, self.nx)
        u_ref = _dxu_ref[..., self.nx*self.T:].reshape(-1, self.T-1, self.nu)
        u_ref = torch.cat([u_ref, torch.zeros_like(u_ref[:, -1:])], dim=1)  # append zero to last action
        vel_ref = dx_ref[..., self.nq:]
        dx_ref = dx_ref[..., :self.nq] * self.dt
        x_ref = torch.cat([dx_ref + x[..., :self.nq], vel_ref], dim=-1)
        
        out_mpc_dict = {"x_t": x_ref[:,0], "x_ref": x_ref, "u_ref": u_ref}
        out_aux_dict = {"x": x_ref, "u": u_ref, "z": z_out}
        return out_mpc_dict, out_aux_dict
    
    def setup_input_layer(self,):
        self.in_dim = self.nx * self.H + self.nx * self.T + self.nu * (self.T-1) # external input and aux input
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hdim),
                torch.nn.LayerNorm(self.hdim),
            )
        else:
            NotImplementedError

    def setup_output_layer(self):  
        self.out_dim = self.nx * self.T + self.nu * (self.T-1)
        if self.layer_type == "mlp":
            self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hdim, self.out_dim)
            )
        else:
            NotImplementedError

class DEQLayerFeedback(DEQLayer):
    """
    Input: current_state, nominal_states_net, nominal_states
    """
    def __init__(self, args, env):
        super().__init__(args, env)

    def forward(self, in_obs_dict, in_aux_dict):
        obs, xn, x, z, iter = in_obs_dict["o"], in_aux_dict["xn"], in_aux_dict["x"], in_aux_dict["z"], in_aux_dict["iter"]
        bsz = obs.shape[0]
        _obs = obs.reshape(bsz,1,self.nx)
        _x = x.reshape(bsz, -1)
        _xn = xn.reshape(bsz, -1)
        # _input = torch.cat([_xn, _x], dim=-1)

        _input1 = self.input_layer(_xn, _x, _obs)
        z_out = self.deq_layer(_input1, z + self.embedding_params[iter][None])
        dx_ref = self.output_layer(z_out)

        dx_ref = dx_ref.view(-1, self.T - 1, self.nx)
        vel_ref = dx_ref[..., self.nq:]
        dx_ref = dx_ref[..., :self.nq] * self.dt
        x_ref = torch.cat([dx_ref + x[..., :1, :self.nq], vel_ref], dim=-1)
        # x_ref = torch.cat([xn[..., 1:, :self.nq].detach().clone() + dx_ref, vel_ref.detach().clone() + xn[..., 1:, self.nq:]], dim=-1)
        # x_ref = torch.cat([dx_ref + _obs[:, :, :self.nq], vel_ref], dim=-1)
        x_ref = torch.cat([_obs, x_ref], dim=-2)
        u_ref = torch.zeros_like(x_ref[..., :self.nu])
        
        out_mpc_dict = {"x_t": obs, "x_ref": x_ref, "u_ref": u_ref}
        out_aux_dict = {"xn": x_ref, "x": x_ref, "u": u_ref, "z": z_out}
        return out_mpc_dict, out_aux_dict

    def setup_input_layer(self):
        
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.in_dim = self.nx * self.T * 2# external input and aux input
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hdim),
                torch.nn.LayerNorm(self.hdim),
            )
        elif self.layer_type == "gcn":
            self.time_emb = torch.nn.Parameter(torch.randn(self.T-1, self.hdim))
            # Get the node embeddings
            self.node_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
            )

            self.x0_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
            )

            self.input_encoder = nn.Sequential(
                nn.Conv1d(self.hdim*4, self.hdim*4, self.kernel_width, padding='same'),
                nn.Mish(),
                nn.Conv1d(self.hdim*4, self.hdim, self.kernel_width, padding='same'),
                nn.GroupNorm(self.num_groups, self.hdim),
                # nn.Mish()
            )

            self.global_pooling = {
                "max": torch.max,
                "mean": torch.mean,
                "sum": torch.sum
            }[self.pooling]
        else:
            NotImplementedError

    def input_layer(self, xn, x, obs):
        if self.layer_type == "mlp":
            inp = self.inp_layer(x)
        elif self.layer_type == "gcn":
            t = self.time_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)#[:,1:]
            x = x.reshape(-1, self.T, self.nx)[:,1:]
            xn = xn.reshape(-1, self.T, self.nx)[:,1:]
            x_emb = self.node_encoder(x)
            xn_emb = self.node_encoder(xn)
            x0_emb = self.x0_encoder(obs[:, 0]).unsqueeze(1).repeat(1, self.T-1, 1)  #TODO switch case for out_type
            inp = torch.cat([x_emb, xn_emb, x0_emb, t], dim=-1).transpose(1, 2).reshape(-1, self.hdim*4, self.T-1)
            inp = self.input_encoder(inp)
        elif self.layer_type == "gat":
            NotImplementedError
        return inp

class DEQLayerQ(DEQLayer):
    """
    Input: current_state, nominal_states_net, nominal_states
    Output: state prediction, Q scalars (bsz x T)
    """
    def __init__(self, args, env):
        super().__init__(args, env)

    def forward(self, in_obs_dict, in_aux_dict):
        """
        compute the policy output for the given observation input and feedback input 
        """
        obs, x_prev, z, q, iter = in_obs_dict["o"], in_aux_dict["x"], in_aux_dict["z"], in_aux_dict["q"], in_aux_dict["iter"]
        # if (x_prev.shape[1] != self.T - 1):  # handle the case of orginal DEQLayer not predicting current state
        #     x_prev = x_prev[:, 1:]
        bsz = obs.shape[0]
        _obs = obs.reshape(bsz,1,self.nx)
        # _input = torch.cat([_obs, x_prev], dim=-2).reshape(bsz, -1)
        _x_prev = x_prev#.reshape(bsz, -1)
        _input = torch.cat([_x_prev, q.reshape(bsz,self.T,1)], dim=-1).reshape(bsz, -1)
        _input1 = self.input_layer(_input, _obs)
        z_out = self.deq_layer(_input1, z + self.embedding_params[iter][None])
        output = self.output_layer(z_out)
        # dx_ref, q_out = output[..., :self.nx*(self.T-1)], output[..., self.nx*(self.T-1):]
        dx_ref, q_out = output[:, :, :self.nx], output[:, :, self.nx:]
        q_out = torch.nn.ReLU()(q_out).reshape(-1, self.T-1)
        q_out = torch.cat([torch.ones_like(q_out[:, -1:]), q_out], dim=1)  # append zero to last action
        dx_ref = dx_ref.view(-1, self.T - 1, self.nx)
        vel_ref = dx_ref[..., self.nq:]
        dx_ref = dx_ref[..., :self.nq] * self.dt
        x_ref = torch.cat([dx_ref + x_prev[..., :1, :self.nq], vel_ref], dim=-1)
        # x_ref = torch.cat([x_prev[..., 1:, :self.nq] + dx_ref, vel_ref + x_prev[..., 1:, self.nq:]], dim=-1)
        # x_ref = torch.cat([dx_ref + _obs[:, :, :self.nq], vel_ref], dim=-1)
        x_ref = torch.cat([_obs, x_ref], dim=-2)
        u_ref = torch.zeros_like(x_ref[..., :self.nu])
        
        out_mpc_dict = {"x_t": obs, "x_ref": x_ref, "u_ref": u_ref, "q": q_out}
        out_aux_dict = {"x": x_ref[:,:], "u": u_ref, "z": z_out, "q": q_out}
        return out_mpc_dict, out_aux_dict

    def setup_input_layer(self):
        self.in_dim = self.nx + self.nx * (self.T - 1) + 1*(self.T) # current state and state prediction
        if self.layer_type == "mlp":
            # ipdb.set_trace()
            self.inp_layer = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hdim),
                torch.nn.LayerNorm(self.hdim),
            )
        elif self.layer_type == "gcn":
            self.time_emb = torch.nn.Parameter(torch.randn(self.T-1, self.hdim))
            # Get the node embeddings
            self.node_encoder = nn.Sequential(
                nn.Linear(self.nx+1, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
            )
            self.x0_encoder = nn.Sequential(
                nn.Linear(self.nx, self.hdim),
                nn.LayerNorm(self.hdim),
                nn.Mish()
            )

            self.input_encoder = nn.Sequential(
                nn.Conv1d(self.hdim*3, self.hdim*4, self.kernel_width, padding='same'),
                nn.Mish(),
                nn.Conv1d(self.hdim*4, self.hdim, self.kernel_width, padding='same'),
                nn.GroupNorm(self.num_groups, self.hdim),
                # nn.Mish()
            )

            self.global_pooling = {
                "max": torch.max,
                "mean": torch.mean,
                "sum": torch.sum
            }[self.pooling]
        else:
            NotImplementedError

    def setup_output_layer(self):  
        
        if self.layer_type == "mlp":
            self.out_dim = self.nx * (self.T-1) + 1*(self.T) # state prediction and Q scalars
            self.out_layer = torch.nn.Sequential(
                torch.nn.Linear(self.hdim, self.out_dim)
            )
            self.gradnorm = GradNormLayer(self.out_dim)
        elif self.layer_type == "gcn":
            self.out_dim = self.nx + 1  # state prediction and 1 scalar (Q) for each T
            self.out_layer = torch.nn.Sequential(
                torch.nn.Conv1d(self.hdim, self.hdim, self.kernel_width, padding='same'),
                torch.nn.GroupNorm(self.num_groups, self.hdim),
                torch.nn.Mish(),
                torch.nn.Conv1d(self.hdim, self.out_dim, self.kernel_width_out, padding='same'),
            )
            self.gradnorm = GradNormLayer(self.out_dim)
            # Implement GradNormLayer for GCN
        else:
            NotImplementedError

    def input_layer(self, x, obs):
        if self.layer_type == "mlp":
            inp = self.inp_layer(x)
        elif self.layer_type == "gcn":
            t = self.time_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)#[:,1:]
            x = x.reshape(-1, self.T, self.nx+1)[:,1:]
            x_emb = self.node_encoder(x)
            x0_emb = self.x0_encoder(obs[:, 0]).unsqueeze(1).repeat(1, self.T-1, 1)  #TODO switch case for out_type
            inp = torch.cat([x_emb, x0_emb, t], dim=-1).transpose(1, 2).reshape(-1, self.hdim*3, self.T-1)
            inp = self.input_encoder(inp)
        elif self.layer_type == "gat":
            NotImplementedError
        return inp

####################
# End
####################

class DEQPolicy(torch.nn.Module):
    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.nu = env.nu
        self.nx = env.nx
        self.dt = env.dt
        self.T = args.T
        self.hdim = args.hdim

        self.fc_inp = torch.nn.Linear(self.nx, self.hdim)
        self.ln_inp = torch.nn.LayerNorm(self.hdim)

        self.fcdeq1 = torch.nn.Linear(self.hdim, self.hdim)
        self.lndeq1 = torch.nn.LayerNorm(self.hdim)
        self.reludeq1 = torch.nn.ReLU()
        self.fcdeq2 = torch.nn.Linear(self.hdim, self.hdim)
        self.lndeq2 = torch.nn.LayerNorm(self.hdim)
        self.reludeq2 = torch.nn.ReLU()
        self.lndeq3 = torch.nn.LayerNorm(self.hdim)

        self.fc_out = torch.nn.Linear(self.hdim, self.nx * self.T)

        self.solver = self.anderson

    def forward(self, x):
        """
        compute the policy output for the given state x
        """
        xinp = self.fc_inp(x)
        xinp = self.ln_inp(xinp)
        z_shape = list(xinp.shape[:-1]) + [
            self.hdim,
        ]
        z = torch.zeros(z_shape).to(xinp)
        z_out = self.deq_fixed_point(xinp, z)
        x_ref = self.fc_out(z_out)
        x_ref = x_ref.view(-1, self.T, self.nx)
        x_ref = x_ref + x[:, None, : self.nx] * 10
        return x_ref

    def deq_fixed_point(self, x, z):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(
                lambda z: self.f(z, x), z, **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(
                lambda y: autograd.grad(
                    f0, z0, y, retain_graph=True)[0] + grad,
                grad,
                **self.kwargs
            )
            return g

        z.register_hook(backward_hook)
        return z

    def f(self, z, x):
        z = self.fcdeq1(z)
        z = self.reludeq1(z)
        z = self.lndeq1(z)
        out = self.lndeq3(self.reludeq2(z + self.lndeq2(x + self.fcdeq2(z))))
        return out

    def anderson(f, x0, m=5, lam=1e-4, max_iter=15, tol=1e-2, beta=1.0):
        """Anderson acceleration for fixed point iteration."""
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
        X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
        X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

        H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
        H[:, 0, 1:] = H[:, 1:, 0] = 1
        y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
        y[:, 0] = 1

        res = []
        for k in range(2, max_iter):
            n = min(k, m)
            G = F[:, :n] - X[:, :n]
            H[:, 1: n + 1, 1: n + 1] = (
                torch.bmm(G, G.transpose(1, 2))
                + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
            )
            alpha = torch.solve(y[:, : n + 1], H[:, : n + 1, : n + 1])[0][
                :, 1: n + 1, 0
            ]  # (bsz x n)

            X[:, k % m] = (
                beta * (alpha[:, None] @ F[:, :n])[:, 0]
                + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
            )
            F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
            res.append(
                (F[:, k % m] - X[:, k % m]).norm().item()
                / (1e-5 + F[:, k % m].norm().item())
            )
            if res[-1] < tol:
                break
        return X[:, k % m].view_as(x0), res


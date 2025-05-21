import sys
import os
import time
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, project_dir)
from torch.utils.tensorboard import SummaryWriter
import policies
from policies import *
from datagen import *
from rex_quadrotor import RexQuadrotor
from my_envs.cartpole import CartpoleEnv
from envs import PendulumEnv, IntegratorEnv
import ipdb
import qpth.qp_wrapper as mpc
import utils
import math
import numpy as np
import torch
import torch.autograd as autograd
from eval import eval_policy, check_param_sensitivity
from fwd_funcs import global_fwd, streaming_fwd, validate_policy

torch.set_default_device('cuda')
np.set_printoptions(precision=4, suppress=True)

models_dir = "./model/"

def seeding(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="pendulum", choices=["pendulum", "cartpole-v0", "cartpole1link", "integrator", "pendulum_stabilize", "rexquadrotor", "FlyingCartpole", "FlyingCartpole_obstacles"])
    parser.add_argument("--nq", type=int, default=1)  # observation (configurations) for the policy
    parser.add_argument("--T", type=int, default=5)  # look-ahead horizon length (including current time step)
    parser.add_argument("--H", type=int, default=1)  # observation history length (including current time step)
    parser.add_argument("--qp_iter", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warm_start", type=bool, default=True)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--deq", action="store_true")
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--deq_iter", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--layer_type", type=str, default='mlp')
    parser.add_argument("--kernel_width", type=int, default=3)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--model_type", type=str, default='', choices=['', 'deq-mpc-deq', 'deq', 'nn', 'diff-mpc-deq', 'diff-mpc-nn', 'deq-mpc-nn'])
    parser.add_argument("--lastqp_solve", action="store_true")
    parser.add_argument("--qp_solve", action="store_true")
    parser.add_argument("--pooling", type=str, default="sum")
    parser.add_argument("--solver_type", type=str, default='al')
    parser.add_argument("--dtype", type=str, default="double")
    parser.add_argument("--deq_out_type", type=int, default=1) 
    parser.add_argument("--policy_out_type", type=int, default=1) 
    parser.add_argument("--loss_type", type=str, default='l1')
    parser.add_argument("--deq_reg", type=float, default=0.1) 
    # check noise_utils.py for noise_type
    parser.add_argument("--data_noise_type", type=int, default=0)
    parser.add_argument("--data_noise_std", type=float, default=0.05)
    parser.add_argument("--data_noise_mean", type=float, default=0.3)
    parser.add_argument("--grad_coeff", action="store_true")
    parser.add_argument("--scaled_output", action="store_true")
    parser.add_argument("--num_trajs_data", type=int, default=1000000)
    parser.add_argument("--num_trajs_frac", type=float, default=1.0)
    parser.add_argument("--check_param_sensitivity", action="store_true")
    parser.add_argument("--rho_init_max", type=int, default=1e4)
    parser.add_argument("--max_train_steps", type=int, default=20000)

    # DEQ/NN specific arguments
    parser.add_argument("--fp_type", type=str, default='anderson', choices=['single', 'multi', 'broyden', 'anderson'])
    parser.add_argument("--inner_deq_iters", type=int, default=4)
    parser.add_argument("--grad_type", type=str, default='fp_grad', choices=['fp_grad', 'last_step_grad', 'bptt'])
    parser.add_argument("--addmem", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--streaming_steps", type=int, default=3)
    parser.add_argument("--streaming_start_iter", type=int, default=0)
    parser.add_argument("--expansion_type", type=str, default='width', choices=['width', 'depth'])
    parser.add_argument("--deq_type", type=str, default='deq', choices=['deq', 'nn'])
    parser.add_argument("--Qscale", type=float, default=1)
    
    # Anderson/Broyden parameters
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--max_steps", type=float, default=10)
    parser.add_argument("--acc_type", type=str, default='good', choices=['good', 'bad'])

    # Environment parameters
    parser.add_argument("--obstacle_radius", type=float, default=0.5)

    # loading/eval params
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ckpt", type=str, default="bc_sac_pen")
    parser.add_argument("--start_iter", type=int, default=-1)


    args = parser.parse_args()

    if args.model_type == 'deq-mpc-deq':
        args.deq = True
        args.qp_solve = True
        args.lastqp_solve = False
    elif args.model_type == 'deq-mpc-nn':
        args.deq = True
        args.qp_solve = True
        args.lastqp_solve = False
        args.deq_type = 'nn'
    elif args.model_type == 'deq': 
        args.deq = True
        args.qp_solve = False
        args.lastqp_solve = False
        args.deq_iter = 1
    elif args.model_type == 'nn':
        args.deq = False
        args.qp_solve = False
        args.lastqp_solve = False
        args.deq_iter = 1
    elif args.model_type == 'diff-mpc-deq':
        args.deq = True
        args.qp_solve = False
        args.lastqp_solve = True
        args.deq_iter = 1
    elif args.model_type == 'diff-mpc-nn':
        args.deq = True
        args.qp_solve = False
        args.lastqp_solve = True
        args.deq_iter = 1
        args.deq_type = 'nn'
    else:
        raise NotImplementedError


    args.str_al_iter = min(int(np.log10(1e10/args.rho_init_max)/2), args.deq_iter)
    if args.streaming and args.streaming_start_iter == 0:
        start_streaming = 1
        args.total_deq_iter = total_deq_iter = args.deq_iter + args.str_al_iter*args.streaming_steps
    else:
        start_streaming = 0
        args.total_deq_iter = total_deq_iter = args.deq_iter
    if args.eval:
        args_file = "./logs/" + args.ckpt + "/args"
        args1 = torch.load(args_file)
        args1.load = True
        args1.ckpt = args.ckpt
        args1.bsz = 200
        args1.test = True
        args1.eval = True
        args1.save = False
        args1.qp_solve = args.qp_solve
        args1.lastqp_solve = args.lastqp_solve
        args = utils.merge_args(args, args1)
        ipdb.set_trace()
    seeding(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device if args.device is None else args.device
    if args.save:
        method_name = args.model_type
        args.name = f"{method_name}_{args.env}_{args.name}" + \
            f"_T{args.T}_bsz{args.bsz}_deq_iter{total_deq_iter}_hdim{args.hdim}_valfrac{args.num_trajs_frac}"
        if 'nn' in args.model_type and args.expansion_type == 'depth':
            args.name += f"_exptype{args.expansion_type}"
        writer = SummaryWriter("./logs/" + args.name)
        print("logging to: ", args.name)

    kwargs = {"dtype": torch.float64,
              "device": args.device, "requires_grad": False}

    if args.env == "pendulum":
        env = PendulumEnv(stabilization=False)
        gt_trajs = get_gt_data(args, env, "sac")
    elif args.env == "integrator":
        env = IntegratorEnv()
        gt_trajs = get_gt_data(args, env, "mpc")
    elif args.env == "rexquadrotor":
        env = RexQuadrotor(bsz=args.bsz, device=args.device)
        gt_trajs = get_gt_data(args, env, "cgac")
    elif args.env == "pendulum_stabilize":
        env = PendulumEnv(stabilization=True)
        gt_trajs = get_gt_data(args, env, "sac")
    elif args.env == "cartpole1link":
        env = CartpoleEnv(nx=4, dt=0.05, stabilization=False, kwargs=kwargs)
        gt_trajs = get_gt_data(args, env, "cgac")
    elif args.env == "cartpole2link":
        env = CartpoleEnv(nx=6, dt=0.03, stabilization=False, kwargs=kwargs)
        gt_trajs = get_gt_data(args, env, "cgac")
    elif args.env == "FlyingCartpole":
        env = FlyingCartpole(max_steps=args.T, Qscale=args.Qscale, bsz=args.bsz, device=args.device)
        gt_trajs = get_gt_data(args, env, "cgac")
    elif args.env == "FlyingCartpole_obstacles":
        env = FlyingCartpole(max_steps=args.T, Qscale=args.Qscale, bsz=args.bsz, device=args.device, obstacles=True, obstacle_radius=args.obstacle_radius)
        gt_trajs = get_gt_data(args, env, "cgac")
    else:
        raise NotImplementedError

    args.num_trajs_data = round(len(gt_trajs)*0.9*args.num_trajs_frac)
    val_gt_trajs = gt_trajs[round(-len(gt_trajs)*0.1):]
    gt_trajs = merge_gt_data(gt_trajs, num_trajs=args.num_trajs_data)
    val_gt_trajs = merge_gt_data(val_gt_trajs, num_trajs=100000000)

    val_samples = []
    num_val_steps = 100
    for i in range(num_val_steps):
        val_samples.append(sample_trajectory(val_gt_trajs, args.bsz, args.H, args.T + args.streaming_steps*int(args.streaming)))
    
    args.Q = env.Qlqr.to(args.device)
    args.R = env.Rlqr.to(args.device)

    traj_sample = sample_trajectory(gt_trajs, 2000, args.H, args.T)
    if args.env == "pendulum":
        traj_sample["state"] = utils.unnormalize_states_pendulum(
            traj_sample["state"])
    elif args.env == "cartpole1link" or args.env == "cartpole2link":
        traj_sample["state"] = utils.unnormalize_states_cartpole_nlink(
            traj_sample["state"])
    elif args.env == "FlyingCartpole":
        traj_sample["state"] = utils.unnormalize_states_flyingcartpole(
            traj_sample["state"])
    args.max_scale = ((traj_sample["state"] - traj_sample["state"][:, :1])*traj_sample["mask"][:, :, None]).reshape(2000*args.T,env.nx).abs().max(dim=0)[0].to(args.device)
    args.pos_scale = ((traj_sample["state"][:, 1:, :args.nq] - traj_sample["state"][:, :1, :args.nq])*traj_sample["mask"][:, 1:, None]).abs().sort(dim=0)[0][-100].to(args.device) # .sort(dim=0)[0][-100]
    args.vel_scale = (traj_sample["state"][:, 1:, args.nq:]*traj_sample["mask"][:, 1:, None]).abs().sort(dim=0)[0][-100].to(args.device) # .sort(dim=0)[0][-100]

    if args.deq:
        policy = DEQMPCPolicy(args, env).to(args.device)
        # save arguments
        if args.save:
            torch.save(args, "./logs/" + args.name + "/args")
            os.makedirs(f"./logs/{args.name}/code/", exist_ok=True)
            os.makedirs(f"./logs/{args.name}/code/deqmpc/", exist_ok=True)
            os.makedirs(f"./logs/{args.name}/code/qpth/", exist_ok=True)
            os.system(f"cp -r {project_dir}/deqmpc/*.py ./logs/{args.name}/code/deqmpc/")
            os.system(f"cp -r {project_dir}/qpth/*.py ./logs/{args.name}/code/qpth/")
            os.system(f"cp -r {project_dir}/deqmpc/run.sh ./logs/{args.name}/code/deqmpc/")
    else:
        policy = NNMPCPolicy(args, env).to(args.device)
        # save arguments
        if args.save:
            torch.save(args, "./logs/" + args.name + "/args")
            os.makedirs(f"./logs/{args.name}/code/", exist_ok=True)
            os.makedirs(f"./logs/{args.name}/code/deqmpc/", exist_ok=True)
            os.makedirs(f"./logs/{args.name}/code/qpth/", exist_ok=True)
            os.system(f"cp -r {project_dir}/deqmpc/*.py ./logs/{args.name}/code/deqmpc/")
            os.system(f"cp -r {project_dir}/qpth/*.py ./logs/{args.name}/code/qpth/")
            os.system(f"cp -r {project_dir}/deqmpc/run.sh ./logs/{args.name}/code/deqmpc/")

    if args.load:
        print("loading model : ", args.ckpt)
        policy.load_state_dict(torch.load(models_dir + args.ckpt))
    
    if args.eval:
        eval_policy(args, env, policy, gt_trajs)
        return None
    if args.check_param_sensitivity:
        check_param_sensitivity(args, env, policy, gt_trajs)
        return None
        
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=args.lr)
    losses = []
    losses_end = []
    time_diffs = []
    dyn_resids = []
    losses_var = []
    losses_iter = [[] for _ in range(total_deq_iter)]
    best_loss = 1e10
    if args.deq_out_type == 0:
        args.num_coeffs_per_iter = 1
    elif args.deq_out_type == 1:
        args.num_coeffs_per_iter = 2
    elif args.deq_out_type == 2:
        args.num_coeffs_per_iter = 3
    elif args.deq_out_type == 3:
        args.num_coeffs_per_iter = 1
    coeffs = torch.ones((total_deq_iter, args.num_coeffs_per_iter), device=args.device)
    losses_iter_opt = [[] for _ in range(total_deq_iter)]
    losses_iter_nn = [[] for _ in range(total_deq_iter)]
    losses_iter_base = [[] for _ in range(total_deq_iter)]
    loss_iter_q = [[] for _ in range(total_deq_iter)]
    losses_iter_nocoeff = [[] for _ in range(total_deq_iter)]
    losses_proxy_iter_nocoeff = [[] for _ in range(total_deq_iter)]
    losses_iter_hist = [[] for _ in range(total_deq_iter)]
    deq_stats = {"fwd_err": [[] for _ in range(total_deq_iter)], "fwd_steps": [[] for _ in range(total_deq_iter)]}

    # run imitation learning using gt_trajs
    start_train = time.time()
    for i in range(args.start_iter+1, args.max_train_steps):
        # sample bsz random trajectories from gt_trajs and a random time step for each
        traj_sample = sample_trajectory(gt_trajs, args.bsz, args.H, args.T + args.streaming_steps*start_streaming)
        traj_sample = {k: v.to(args.device) for k, v in traj_sample.items()}

        if args.env == "pendulum":
            traj_sample["state"] = utils.unnormalize_states_pendulum(
                traj_sample["state"])
            traj_sample["obs"] = utils.unnormalize_states_pendulum(traj_sample["obs"])
        elif args.env == "cartpole1link" or args.env == "cartpole2link":
            traj_sample["state"] = utils.unnormalize_states_cartpole_nlink(
                traj_sample["state"])
            traj_sample["obs"] = utils.unnormalize_states_pendulum(traj_sample["obs"])
        elif args.env == "FlyingCartpole":
            traj_sample["state"] = utils.unnormalize_states_flyingcartpole(
            traj_sample["state"])
            traj_sample["obs"] = utils.unnormalize_states_flyingcartpole(traj_sample["obs"])
        pretrain_done = False if (i < 5000 and args.pretrain) else True
        gt_obs = traj_sample["obs"]
        obs_in = gt_obs.squeeze(1)
        
        gt_actions = traj_sample["action"]
        gt_states = traj_sample["state"]
        gt_mask = traj_sample["mask"]
        gt_obs_actions = traj_sample["obs_action"]
        if not start_streaming:
            time_diff, loss_dict, policy_out, coeffs = global_fwd(args, obs_in, gt_obs, gt_states, gt_actions,
                                    gt_obs_actions, gt_mask, policy, coeffs, pretrain_done, i, 
                                    losses_iter_nocoeff, losses_proxy_iter_nocoeff)
        else:
            time_diff, loss_dict, policy_out, coeffs = streaming_fwd(args, obs_in, gt_obs, gt_states, gt_actions, 
                                    gt_obs_actions, gt_mask, policy, coeffs, pretrain_done, i, 
                                    losses_iter_nocoeff, losses_proxy_iter_nocoeff)
        loss = loss_dict["loss"]
        net_jac_loss = 0.0
        time_diffs.append(time_diff)
        optimizer.zero_grad()
        (loss).backward()
        if policy.model.node_encoder[0].weight.grad.isnan().any():
            ipdb.set_trace()
        losses.append(loss.item())
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 2)
        optimizer.step()

        loss_end = loss_dict["loss_end"]
        losses_end.append(loss_end.item())
        losses_var.append(loss_dict["losses_var"])
        [losses_iter[k].append(loss_dict["losses_iter"][k]) for k in range(total_deq_iter)]
        [losses_iter_opt[k].append(loss_dict["losses_iter_opt"][k]) for k in range(total_deq_iter)]
        [losses_iter_nn[k].append(loss_dict["losses_iter_nn"][k]) for k in range(total_deq_iter)]
        [losses_iter_base[k].append(loss_dict["losses_iter_base"][k]) for k in range(total_deq_iter)]
        if "deq_stats" in policy_out:
            [deq_stats["fwd_err"][k].append(policy_out["deq_stats"]["fwd_err"][k].item()) for k in range(total_deq_iter)]
            [deq_stats["fwd_steps"][k].append(policy_out["deq_stats"]["fwd_steps"][k].item()) for k in range(total_deq_iter)]
        if 'nominal_x_ests' in policy_out:
            [losses_iter_hist[k].append(loss_dict['losses_x_ests'][k]) for k in range(total_deq_iter)]
        if 'q_scaling' in policy_out:
            [loss_iter_q[k].append(loss_dict['q_scaling'][k]) for k in range(total_deq_iter)]

        # Printing
        if i % 100 == 0:
            end_train = time.time()
            start_val = time.time()
            with torch.no_grad():
                loss_end_val, loss_opt_val, loss_nn_val = validate_policy(args, env, policy, val_samples, coeffs, pretrain_done, total_deq_iter)
            end_val = time.time()
            print("train time: ", end_train - start_train, "val time: ", end_val - start_val)
            start_train = time.time()
            print("iter: ", i, "deqmpc" if pretrain_done else "deq")
            print(
                "grad norm: ",
                torch.nn.utils.clip_grad_norm_(
                    policy.model.parameters(), 1000),
            )
            print(
                "loss: ",
                np.mean(losses) / total_deq_iter,
                "loss_end: ",
                np.mean(losses_end)/(1 + args.streaming_steps*start_streaming),
                "loss_end_val: ", loss_end_val,
                "avg time: ",
                np.mean(time_diffs)
            )
            if args.save:
                if loss_end_val < best_loss:
                    best_loss = loss_end_val
                    torch.save(policy.state_dict(), "./model/" + args.name)
                    print("saved model")
                writer.add_scalar("losses/loss_avg",
                                  np.mean(losses) / total_deq_iter, i)
                writer.add_scalar("losses/loss_end", np.mean(losses_end)/(1 + args.streaming_steps*start_streaming), i)
                
                [writer.add_scalar(f"losses/loss{k}", np.mean(losses_iter[k]), i) for k in range(len(losses_iter))]
                [writer.add_scalar(f"coeffs/coeff{j}{k}", coeffs[j,k], i) for j in range(total_deq_iter) for k in range(args.num_coeffs_per_iter)]
                [writer.add_scalar(f"losses_nocoeff/loss_nocoeff{k}", np.mean(losses_iter_nocoeff[k]), i) for k in range(len(losses_iter_nocoeff))]
                [writer.add_scalar(f"losses_nocoeff/loss_proxy_nocoeff{k}", np.mean(losses_proxy_iter_nocoeff[k]), i) for k in range(len(losses_proxy_iter_nocoeff))]
                [writer.add_scalar(f"losses_opt/losses_iter_opt{k}", np.mean(losses_iter_opt[k]), i) for k in range(len(losses_iter_opt))]
                [writer.add_scalar(f"losses_nn/losses_iter_nn{k}", np.mean(losses_iter_nn[k]), i) for k in range(len(losses_iter_nn))]
                [writer.add_scalar(f"losses_base/losses_iter_base{k}", np.mean(losses_iter_base[k]), i) for k in range(len(losses_iter_base))]
                [writer.add_scalar(f"losses_var/losses_var{k}", np.mean(losses_var[k]), i) for k in range(len(losses_var))]
                [writer.add_scalar(f"losses_q/losses_q{k}", np.mean(loss_iter_q[k]), i) for k in range(len(loss_iter_q))]
                [writer.add_scalar(f"losses_hist/losses_hist{k}", np.mean(losses_iter_hist[k]), i) for k in range(len(losses_iter_hist))]
                [writer.add_scalar(f"deq_stats/fwd_err{k}", np.mean(deq_stats["fwd_err"][k]), i) for k in range(total_deq_iter)]
                [writer.add_scalar(f"deq_stats/fwd_steps{k}", np.mean(deq_stats["fwd_steps"][k]), i) for k in range(total_deq_iter)]
                
                # validation losses
                writer.add_scalar("val_losses/loss_end", loss_end_val, i)
                [writer.add_scalar(f"val_losses_opt/loss_opt_{k}", loss_opt_val[k], i) for k in range(total_deq_iter)]
                [writer.add_scalar(f"val_losses_nn/loss_nn_{k}", loss_nn_val[k], i) for k in range(total_deq_iter)]

            if i > args.streaming_start_iter and args.streaming and not start_streaming:
                start_streaming = 1
                args.total_deq_iter = total_deq_iter = args.deq_iter + args.str_al_iter*args.streaming_steps
                coeffs = torch.ones((total_deq_iter, args.num_coeffs_per_iter), device=args.device)
            losses = []
            losses_end = []
            time_diffs = []
            losses_iter = [[] for _ in range(total_deq_iter)]
            losses_iter_noQreg = [[] for _ in range(total_deq_iter)]
            losses_var = [[] for _ in range(total_deq_iter)]
            losses_iter_nocoeff = [[] for _ in range(total_deq_iter)]
            losses_proxy_iter_nocoeff = [[] for _ in range(total_deq_iter)]
            losses_iter_hist = [[] for _ in range(total_deq_iter)]
            losses_iter_opt = [[] for _ in range(total_deq_iter)]
            losses_iter_nn = [[] for _ in range(total_deq_iter)]
            losses_iter_base = [[] for _ in range(total_deq_iter)]
            loss_iter_q = [[] for _ in range(total_deq_iter)]
            
            deq_stats = {"fwd_err": [[] for _ in range(total_deq_iter)], "fwd_steps": [[] for _ in range(total_deq_iter)]}


if __name__ == "__main__":
    main()

import numpy as np
import torch
import pickle

def get_gt_data(args, env, type="mpc"):
    """
    Get ground truth data for imitation learning.
    Args:
        args: The arguments for the training script.
        env: The environment.
        type: The type of controller to use. Can be 'mpc' or 'ppo' or 'sac'.
    Returns:
        A list of trajectories, each trajectory is a list of (state, action) tuples.
    """
    if args.env == "pendulum":
        data_dir = 'expert_traj_sac-Pendulum-v0_new.pkl'
    elif args.env == "pendulum_stabilize":
        data_dir = 'expert_traj_mpc-Pendulum-v0-stabilize_new.pkl'
    elif args.env == "integrator":
        data_dir = 'expert_traj_mpc-Integrator-v0_new.pkl'
    elif args.env == "rexquadrotor":
        data_dir = 'expert_traj_cgac-RexQuadrotor-v0_new.pkl'
    elif args.env == "cartpole1link":
        data_dir = 'expert_traj_cgac-Cartpole1l-v0_new.pkl'
    elif args.env == "FlyingCartpole":
        data_dir = f"expert_traj_{type}-{env.spec_id}-ub03-clip-s_new.pkl"
    elif args.env == 'FlyingCartpole_obstacles':
        data_dir = f"expert_traj_{type}-{env.spec_id}-ub03-clip-s_new.pkl"
    with open('data/' + data_dir, 'rb') as f:
        gt_trajs = pickle.load(f)

    return gt_trajs

def merge_trajs_data(gt_trajs, num_trajs=2000000):
    """
    Merge ground truth data for imitation learning.
    Args:
        gt_trajs: A list of trajectories, each trajectory is a list of (state, action) tuples.
    Returns:
        A list of (state, action) tuples.
    """
    merged_states = []
    merged_actions = []
    for i, traj in enumerate(gt_trajs):
        states = []
        actions = []
        if i >= num_trajs:
            break
        for state, action in traj:
            states.append(state)
            actions.append(action)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        merged_states.append(states)
        merged_actions.append(actions)
    merged_states = torch.stack(merged_states, dim=0)
    merged_actions = torch.stack(merged_actions, dim=0) 
    return merged_states, merged_actions

def merge_gt_data(gt_trajs, num_trajs=2):
    """
    Merge ground truth data for imitation learning.
    Args:
        gt_trajs: A list of trajectories, each trajectory is a list of (state, action) tuples.
    Returns:
        A list of (state, action) tuples.
    """
    merged_gt_traj = {"state": [], "action": [], "mask": []}
    for i, traj in enumerate(gt_trajs):
        if i >= num_trajs:
            break
        for state, action in traj:
            merged_gt_traj["state"].append(state)
            merged_gt_traj["action"].append(action)
            merged_gt_traj["mask"].append(1)
        merged_gt_traj["mask"][-1] = 0  
    merged_gt_traj["state"] = torch.tensor(
        np.array(merged_gt_traj["state"]), dtype=torch.float32
    )
    merged_gt_traj["action"] = torch.tensor(
        np.array(merged_gt_traj["action"]), dtype=torch.float32
    )
    merged_gt_traj["mask"] = torch.tensor(
        np.array(merged_gt_traj["mask"]), dtype=torch.float32
    )
    return merged_gt_traj

def sample_trajectory(gt_trajs, bsz, H, T):
    """
    Sample a batch of trajectories from the ground truth data.
    Args:
        gt_trajs: A dictionary of "state", "action" and "mask" tensors with concatenated trajectories.
        bsz: Batch size.
        H: History length.
        T: Lookahead horizon length.
    Returns:
        A list of trajectories, each trajectory is a list of (obs, state, action) tuples (H, T, T).
    """
    idxs = np.random.randint(H-1, len(gt_trajs["state"]), bsz*4)
    trajs = {"obs": [], "obs_action": [], "state": [], "action": [], "mask": []}
    i = 0
    j = 0
    while j < bsz:
        if (gt_trajs["mask"][idxs[i]+1-H:idxs[i]+1] == 0).sum()>0:
            i += 1
            continue
        trajs["obs"].append(gt_trajs["state"][idxs[i]+1 - H : idxs[i]+1])
        trajs["obs_action"].append(gt_trajs["action"][idxs[i]+1 - H : idxs[i]+1])
        if idxs[i] + T <= len(gt_trajs["state"]):
            trajs["state"].append(gt_trajs["state"][idxs[i] : idxs[i] + T])
            trajs["action"].append(gt_trajs["action"][idxs[i] : idxs[i] + T])
            trajs["mask"].append(gt_trajs["mask"][idxs[i] : idxs[i] + T])
        else:
            padding = idxs[i] + T - len(gt_trajs["state"])            
            trajs["state"].append(
                torch.cat(
                    [gt_trajs["state"][idxs[i] :], gt_trajs["state"][:padding] * 0.0],
                    dim=0,
                )
            )
            trajs["action"].append(
                torch.cat(
                    [gt_trajs["action"][idxs[i] :], gt_trajs["action"][:padding] * 0.0],
                    dim=0,
                )
            )
            trajs["mask"].append(
                torch.cat(
                    [gt_trajs["mask"][idxs[i] :], gt_trajs["mask"][:padding] * 0.0], dim=0
                )
            )
        i += 1
        j += 1
    trajs["obs"] = torch.stack(trajs["obs"])
    trajs["state"] = torch.stack(trajs["state"])
    trajs["action"] = torch.stack(trajs["action"])
    trajs["mask"] = torch.stack(trajs["mask"])
    trajs["obs_action"] = torch.stack(trajs["obs_action"])
    for i in reversed(range(T)):
        trajs["mask"][:, i] = torch.prod(trajs["mask"][:, :i+1], dim=1)
    return trajs

def seeding(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

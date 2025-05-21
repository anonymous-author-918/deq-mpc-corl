# Deep Equilibrium Model Predictive Control

This repository contains supplementary code for our CoRL submission on Deep Equilibrium Model Predictive Control.

## Running Instructions

We provide a convenient `run.sh` script located at `deqmpc/run.sh`. This script contains commands with all required flags to train each environment. To use it:

1. Open the script
2. Uncomment the command corresponding to your desired environment
3. Adjust the configuration flags as needed

### Configuration Flags

| Flag | Description | Default/Example |
|------|-------------|-----------------|
| `--model_type` | Model Type | Options: `deq-mpc-deq`, `deq-mpc-nn`, `diff-mpc-deq`, `diff-mpc-nn`, `deq`, `nn` |
| `--deq_iter` | Number of DEQ-MPC iterations (N) | `6` |
| `--T` | Time horizon of the MPC problem | `5` |
| `--env` | Environment name | `pendulum` |
| `--hdim` | Hidden dimension of the network | `256` |
| `--num_trajs_frac` | Training set fraction | `1.0` |
| `--streaming` | Enable streaming setup with warm-starting | Flag |
| `--streaming_steps` | Number of time-steps for streaming (L) | `2` |

### Evaluating Pre-trained Models

To evaluate a pre-trained model, use the same commands with these additional flags:

```bash
--load --eval --ckpt <ckpt_path>
```

Replace `<ckpt_path>` with the path to your checkpoint file.

# Cluster Launch Templates

These SLURM templates run the multi-turn pipeline on a university cluster. Adjust the partition, account, GPU type, walltime, and module names for your site.

## Files

- `train_grpo.slurm`: runs the preflight check, trains both arms, evaluates both adapters, and writes the paired comparison.
- `train_grpo_apptainer.slurm`: same workflow inside an Apptainer image with `--nv` and bind mounts.

## Before Submitting

1. Create the log directory because SLURM does not create it for you.

```bash
mkdir -p logs outputs/eval_results
```

2. Edit the `#SBATCH` headers. Zaratan and similar clusters usually need an explicit `--partition` and `--account`.
3. Confirm that the GPU count matches the configuration. Training uses one GPU and the verifier worker uses `verifier_device`. The default YAML files set `cuda:0` and `cuda:1`.
4. Set `HF_HOME` if the default cache location is not on shared storage.
5. Run `python scripts/check_env.py` once inside the target environment.

## Submitting

```bash
sbatch scripts/slurm/train_grpo.slurm
```

For the container template:

```bash
export APPTAINER_IMAGE=/path/to/cuda_rl.sif
sbatch scripts/slurm/train_grpo_apptainer.slurm
```

## Resuming

The trainer writes `trainer_state.pt` next to each adapter checkpoint. Pass the checkpoint directory to `train_grpo.py`:

```bash
python train_grpo.py \
  --arm arm_a \
  --config configs/multiturn_grpo.yaml \
  --resume_from_checkpoint outputs/arm_a_contrastive/checkpoint-50
```

The resumed run restores the optimizer, scheduler, and random number generator state. The archive is reloaded from `archive_path`, so contrastive anchors survive a preemption.

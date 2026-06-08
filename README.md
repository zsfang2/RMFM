# RMFM: RadioMap U-Net Flow Prior

Standalone experiments for training a RadioMapSeer-specific flow-matching prior
with a U-Net backbone, then using a FlowDPS-style sparse-recovery sampler.

This folder is intentionally independent from `ControlFlow-RM`. It does not
modify the existing SD3 sampler or `main.py`.

## Paths

- Code: `/home/Users_Work_Space/zsfang/rmfm`
- Default dataset: `/home/DataDisk/zsfang/dataset/RadioMapSeer`
- Default checkpoints/results: `/home/DataDisk/zsfang/rmfm`

## Train

Quick smoke test:

```bash
cd /home/Users_Work_Space/zsfang/rmfm
python scripts/train_unet_flow.py \
  --dataset_root /home/DataDisk/zsfang/dataset/RadioMapSeer \
  --output_dir /home/DataDisk/zsfang/rmfm/checkpoints/smoke_unet_flow \
  --gain_modes DPM \
  --base_channels 16 \
  --channel_mults 1,2,2 \
  --batch_size 2 \
  --max_steps 10 \
  --max_train_samples 32 \
  --max_val_samples 8 \
  --num_workers 0 \
  --mixed_precision no \
  --device cuda:0
```

Longer first run:

```bash
cd /home/Users_Work_Space/zsfang/rmfm
python scripts/train_unet_flow.py \
  --dataset_root /home/DataDisk/zsfang/dataset/RadioMapSeer \
  --output_dir /home/DataDisk/zsfang/rmfm/checkpoints/radiomapseer_unet_flow_dpm \
  --gain_modes DPM \
  --base_channels 64 \
  --channel_mults 1,2,4,4 \
  --batch_size 8 \
  --max_steps 100000 \
  --lr 1e-4 \
  --num_workers 4 \
  --mixed_precision fp16 \
  --device cuda:0 \
  --channels_last \
  --allow_tf32
```

You can also select a GPU with `CUDA_VISIBLE_DEVICES`, for example:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_unet_flow.py ...
```

## Improving GPU Utilization

If GPU utilization is low, first increase the real batch size until VRAM is
reasonably used:

```bash
python scripts/train_unet_flow.py \
  --dataset_root /home/DataDisk/zsfang/dataset/RadioMapSeer \
  --output_dir /home/DataDisk/zsfang/rmfm/checkpoints/radiomapseer_unet_flow_dpm \
  --gain_modes DPM \
  --batch_size 32 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --mixed_precision fp16 \
  --channels_last \
  --allow_tf32 \
  --device cuda:0
```

If the run is stable, you can test:

```bash
--compile
```

Use `grad_accum_steps` only when you need a larger effective batch than memory
allows; it does not by itself increase per-step GPU utilization.

The split is by city id, not by image, to avoid leaking the same building layout
into train and validation.

## Sparse Recovery

The sparse observation is represented on the image grid as:

```text
y = Mx + Mn
```

where `M` is the binary sampling mask. Noise is only added at observed pixels,
so unobserved pixels do not carry measurement information. The FlowDPS sampler
then applies data consistency only on the known sparse samples.

Single sampling rate:

```bash
cd /home/Users_Work_Space/zsfang/rmfm
python scripts/sample_flowdps.py \
  --dataset_root /home/DataDisk/zsfang/dataset/RadioMapSeer \
  --checkpoint /home/DataDisk/zsfang/rmfm/checkpoints/radiomapseer_unet_flow_dpm/latest.pt \
  --output_dir /home/DataDisk/zsfang/rmfm/results/unet_flowdps \
  --gain_mode DPM \
  --split test \
  --sampling_rate 0.01 \
  --num_samples 50 \
  --samples_per_city 0 \
  --num_steps 20 \
  --step_size 50.0 \
  --dc_iters 3
```

Set `--samples_per_city > 0` to select a fixed number of samples from each city
after split filtering. If a city has fewer samples than requested, all available
samples from that city are used.

Multiple sampling rates:

```bash
cd /home/Users_Work_Space/zsfang/rmfm
python scripts/benchmark_sampling_rates.py \
  --dataset_root /home/DataDisk/zsfang/dataset/RadioMapSeer \
  --checkpoint /home/DataDisk/zsfang/rmfm/checkpoints/radiomapseer_unet_flow_dpm/latest.pt \
  --output_dir /home/DataDisk/zsfang/rmfm/results/benchmark_dpm \
  --gain_modes DPM \
  --split test \
  --sampling_rates 0.005 0.01 0.02 0.03 0.05 \
  --num_samples 50
```

Condition ablations can be run at inference time with the same checkpoint and
the same sparse masks:

```bash
cd /home/Users_Work_Space/zsfang/rmfm
python scripts/benchmark_sampling_rates.py \
  --dataset_root /home/DataDisk/zsfang/dataset/RadioMapSeer \
  --checkpoint /home/DataDisk/zsfang/rmfm/checkpoints/radiomapseer_unet_flow_dpm/checkpoint_step_0015000.pt \
  --output_dir /home/DataDisk/zsfang/rmfm/results/condition_ablation_dpm \
  --gain_modes DPM \
  --split test \
  --sampling_rates 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 0.0010 \
  --condition_modes full no_source no_building no_building_source zero_all \
  --num_samples -1 \
  --num_steps 20 \
  --step_size 50.0 \
  --dc_iters 3 \
  --device cuda:1
```

`condition_modes`:

- `full`: keep building mask, source heatmap, and third channel.
- `no_source`: keep building mask and third channel, zero source heatmap.
- `no_building`: keep source heatmap and third channel, zero building mask.
- `no_building_source`: zero building and source, keep third channel.
- `building_only`: keep only building mask, zero source and later channels.
- `source_only`: keep only source heatmap, zero building and later channels.
- `zero_all`: zero every condition channel.

For the information-ablation matrix discussed in the project notes, use:

```bash
cd /home/Users_Work_Space/zsfang/rmfm
bash scripts/run_information_ablation_dpm.sh
```

This writes `information_ablation_summary.csv` with explicit columns for
whether sparse samples, building, source, and the third channel are used.
Edit `scripts/run_information_ablation_dpm.sh` to change paths, `GPU_ID`,
sampling rates, experiment groups, or sampling hyperparameters. The script also
writes `run.log`, `run.pid`, `run_command.txt`, and `launcher_config.txt` into
the output directory.

The default experiment groups are:

- `sampling_only`: sparse samples only, with the DPM constant third channel.
- `sampling_building`: sparse samples plus building mask.
- `sampling_source`: sparse samples plus source heatmap.
- `sampling_building_source`: sparse samples plus building mask and source heatmap.
- `building_source_no_sampling`: building mask and source heatmap only, with `sampling_rate=0.0`.

`benchmark_information_ablations.py` also supports regular sampling-rate
ranges:

```bash
python scripts/benchmark_information_ablations.py \
  --checkpoint /home/DataDisk/zsfang/rmfm/checkpoints/radiomapseer_unet_flow_dpm/checkpoint_step_0015000.pt \
  --output_dir /home/DataDisk/zsfang/rmfm/results/example_range \
  --gain_modes DPM \
  --split test \
  --sampling_rate_start 0.01 \
  --sampling_rate_end 1.0 \
  --sampling_rate_step 0.01 \
  --experiments sampling_only \
  --samples_per_city 10 \
  --num_samples -1 \
  --device cuda:0
```

## Sample-Only Collapse Sweep

To estimate where `sampling_only` inference collapses at low sampling rates,
run:

```bash
cd /home/Users_Work_Space/zsfang/rmfm
bash scripts/run_sampling_only_collapse_sweep.sh
```

The launcher is configured near the top of the shell script. By default it uses:

- checkpoint: `/home/DataDisk/zsfang/rmfm/checkpoints/radiomapseer_unet_flow_dpm/checkpoint_step_0015000.pt`
- gain modes: `DPM IRT2 IRT4`
- split: `test`
- experiment: `sampling_only`
- rates: `0.0001` plus `0.01, 0.02, ..., 1.00`
- samples per city: `10`
- GPU selector: `GPU_ID=1`

With the current split, this selects:

- DPM: 70 test cities x 10 samples = 700 images per rate
- IRT2: 70 test cities x 10 samples = 700 images per rate
- IRT4: 70 test cities x 2 samples = 140 images per rate

The default sweep has 101 rates, so it runs 155,540 recoveries. It saves all
sample outputs and metrics:

```text
output_dir/
  sampling_only/
    DPM/
      sr_0p0001/
        input/
        recon/
        label/
        masks/
        metrics.csv
        summary.json
    IRT2/
    IRT4/
  information_ablation_summary.csv
  run.log
  run.pid
  run_command.txt
  launcher_config.txt
```

The process runs in the background by default. Monitor it with:

```bash
tail -f /home/DataDisk/zsfang/rmfm/results/sample_only_collapse_sweep_0p01/run.log
cat /home/DataDisk/zsfang/rmfm/results/sample_only_collapse_sweep_0p01/run.pid
```

At roughly 0.15 seconds per recovery, pure sampling takes about 6.5 hours.
Including subprocess startup, model loading, image writing, and mask writing,
plan for roughly 8-14 hours. Because every sample writes `input`, `recon`,
`label`, and `mask`, reserve at least 100 GB in the result filesystem.

Multiple checkpoints:

```bash
cd /home/Users_Work_Space/zsfang/rmfm
python scripts/benchmark_checkpoints.py \
  --checkpoint_dir /home/DataDisk/zsfang/rmfm/checkpoints/radiomapseer_unet_flow_dpm \
  --output_dir /home/DataDisk/zsfang/rmfm/results/checkpoint_sweep_dpm \
  --dataset_root /home/DataDisk/zsfang/dataset/RadioMapSeer \
  --gain_modes DPM \
  --split test \
  --sampling_rates 0.01 0.03 0.05 \
  --num_samples 50 \
  --num_steps 20 \
  --device cuda:1 \
  --include_best \
  --include_latest
```

To test selected checkpoints only:

```bash
python scripts/benchmark_checkpoints.py \
  --checkpoint_dir /home/DataDisk/zsfang/rmfm/checkpoints/radiomapseer_unet_flow_dpm \
  --checkpoints \
    /home/DataDisk/zsfang/rmfm/checkpoints/radiomapseer_unet_flow_dpm/checkpoint_step_0010000.pt \
    /home/DataDisk/zsfang/rmfm/checkpoints/radiomapseer_unet_flow_dpm/checkpoint_step_0015000.pt \
  --output_dir /home/DataDisk/zsfang/rmfm/results/checkpoint_sweep_selected \
  --dataset_root /home/DataDisk/zsfang/dataset/RadioMapSeer \
  --gain_modes DPM \
  --split test \
  --sampling_rates 0.01 0.03 0.05 \
  --num_samples 50 \
  --device cuda:1
```

Outputs:

```text
output_dir/
  DPM/
    sr_0p0100/
      input/
      recon/
      label/
      masks/
      metrics.csv
      summary.json
  benchmark_summary.csv
```

By default, `benchmark_sampling_rates.py` and `benchmark_checkpoints.py` use
`--split test`. The split is loaded from `city_splits.json` next to the
checkpoint. Use `--split all` only for debugging.

## Version Control

This folder is a git repository. The `.gitignore` excludes Python caches,
local environments, logs, PID files, result folders, checkpoint folders, and
model weight files such as `*.pt`, `*.pth`, and `*.ckpt`.

Recommended initial commit:

```bash
cd /home/Users_Work_Space/zsfang/rmfm
git add .
git commit -m "Initial rmfm experiment code"
```

## Model

Training uses rectified-flow style flow matching:

```text
x0 = clean radio map in [-1, 1]
x1 = Gaussian noise
t ~ Uniform(0, 1)
xt = (1 - t) * x0 + t * x1
target velocity = x1 - x0
```

The U-Net input is:

```text
[xt, building_mask, source_heatmap, car_or_ones]
```

The sampler starts from noise and integrates backward from `t=1` to `t=0`.
At each step it estimates the clean endpoint, applies data consistency on the
known sparse samples, then continues the flow update.

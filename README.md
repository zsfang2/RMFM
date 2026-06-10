# RMFM: RadioMap U-Net Flow Prior

Standalone experiments for training a RadioMapSeer-specific flow-matching prior
with a U-Net backbone, then using a FlowDPS-style sparse-recovery sampler.

This folder is intentionally independent from `ControlFlow-RM`. It does not
modify the existing SD3 sampler or `main.py`.

## Paths

- Code: `/path/to/your/rmfm`
- Default dataset: `/path/to/your/RadioMapSeer`
- Default checkpoints/results: `/path/to/your/rmfm_runs`

## Environment

Use Python 3.10 or newer. The current code has been checked with:

```text
Python 3.10.20
torch 2.4.1+cu121
diffusers 0.30.1
numpy 2.2.6
pillow 12.2.0
scikit-image 0.25.2
tqdm 4.67.3
```

Create a clean environment:

```bash
conda create -n rmfm python=3.10 -y
conda activate rmfm
```

Install PyTorch for your CUDA driver from the official PyTorch index. For
example, for CUDA 12.1 wheels:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

Then install the project dependencies. The PyTorch range in
`requirements.txt` is compatible with the version installed above:

```bash
cd /path/to/your/rmfm
pip install -r requirements.txt
```

Check the environment before running long experiments:

```bash
python -c "import torch, diffusers, numpy, PIL, skimage; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m py_compile rmfm/metrics.py scripts/sample_token_flowdps.py scripts/benchmark_token_unet_phase1.py
```

If CUDA is not available, training and sampling will fall back to CPU only when
the selected script supports it, but full experiments are intended for a CUDA
GPU.

## Train

Quick smoke test:

```bash
cd /path/to/your/rmfm
python scripts/train_unet_flow.py \
  --dataset_root /path/to/your/RadioMapSeer \
  --output_dir /path/to/your/rmfm_runs/checkpoints/smoke_unet_flow \
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
cd /path/to/your/rmfm
python scripts/train_unet_flow.py \
  --dataset_root /path/to/your/RadioMapSeer \
  --output_dir /path/to/your/rmfm_runs/checkpoints/radiomapseer_unet_flow_dpm \
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
  --dataset_root /path/to/your/RadioMapSeer \
  --output_dir /path/to/your/rmfm_runs/checkpoints/radiomapseer_unet_flow_dpm \
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
cd /path/to/your/rmfm
python scripts/sample_flowdps.py \
  --dataset_root /path/to/your/RadioMapSeer \
  --checkpoint /path/to/your/rmfm_runs/checkpoints/radiomapseer_unet_flow_dpm/latest.pt \
  --output_dir /path/to/your/rmfm_runs/results/unet_flowdps \
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
cd /path/to/your/rmfm
python scripts/benchmark_sampling_rates.py \
  --dataset_root /path/to/your/RadioMapSeer \
  --checkpoint /path/to/your/rmfm_runs/checkpoints/radiomapseer_unet_flow_dpm/latest.pt \
  --output_dir /path/to/your/rmfm_runs/results/benchmark_dpm \
  --gain_modes DPM \
  --split test \
  --sampling_rates 0.005 0.01 0.02 0.03 0.05 \
  --num_samples 50
```

Condition ablations can be run at inference time with the same checkpoint and
the same sparse masks:

```bash
cd /path/to/your/rmfm
python scripts/benchmark_sampling_rates.py \
  --dataset_root /path/to/your/RadioMapSeer \
  --checkpoint /path/to/your/rmfm_runs/checkpoints/radiomapseer_unet_flow_dpm/checkpoint_step_0015000.pt \
  --output_dir /path/to/your/rmfm_runs/results/condition_ablation_dpm \
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
cd /path/to/your/rmfm
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
  --checkpoint /path/to/your/rmfm_runs/checkpoints/radiomapseer_unet_flow_dpm/checkpoint_step_0015000.pt \
  --output_dir /path/to/your/rmfm_runs/results/example_range \
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
cd /path/to/your/rmfm
bash scripts/run_sampling_only_collapse_sweep.sh
```

The launcher is configured near the top of the shell script. By default it uses:

- checkpoint: `/path/to/your/rmfm_runs/checkpoints/radiomapseer_unet_flow_dpm/checkpoint_step_0015000.pt`
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
tail -f /path/to/your/rmfm_runs/results/sample_only_collapse_sweep_0p01/run.log
cat /path/to/your/rmfm_runs/results/sample_only_collapse_sweep_0p01/run.pid
```

At roughly 0.15 seconds per recovery, pure sampling takes about 6.5 hours.
Including subprocess startup, model loading, image writing, and mask writing,
plan for roughly 8-14 hours. Because every sample writes `input`, `recon`,
`label`, and `mask`, reserve at least 100 GB in the result filesystem.

## RMFM-TokenUNet-v1

TokenUNet-v1 keeps the existing flow-matching objective and sparse recovery
sampler, but changes how sparse observations enter the model:

```text
baseline:
  x_t + building + source + sparse image -> U-Net

TokenUNet-v1:
  x_t + building + source -> U-Net dense branch
  sparse points -> tokenizer -> point tokens -> U-Net cross-attention
```

The implementation is intentionally separate from the baseline:

- `rmfm/obs_tokenizer.py`: sparse point tokenizer with Fourier coordinate features and a small Transformer encoder.
- `rmfm/token_utils.py`: sparse mask sampling, point extraction, padding, and K-max subsampling.
- `rmfm/modeling_token_unet_flow.py`: TokenUNet wrapper around `diffusers.UNet2DConditionModel`.
- `scripts/train_token_unet_flow.py`: flow-matching training for TokenUNet.
- `scripts/sample_token_flowdps.py`: TokenUNet sampling with `no_dc` and `dc` modes.
- `scripts/benchmark_token_unet_phase1.py`: phase-1 validation matrix with a combined summary CSV.

Dense inputs use only `x_t`, building mask, and source heatmap. The sparse
measurements are encoded as point tokens with shape `[B, K+1, C]`, including a
CLS token. The default token dimension is 256, with 2 tokenizer Transformer
layers and 4 heads. `K_MAX=256` limits token attention cost; if a mask contains
more observed points, the token branch subsamples points, while `dc` mode still
uses the full sparse mask for data consistency.

Train TokenUNet-v1:

```bash
cd /path/to/your/rmfm
bash scripts/run_token_unet_phase1_train.sh
```

The training launcher writes `run.log`, `run.pid`, `run_command.txt`, and
`launcher_config.txt` into the checkpoint directory. Defaults:

- checkpoint directory: `/path/to/your/rmfm_runs/checkpoints/radiomapseer_token_unet_flow_dpm`
- gain modes: `DPM`
- random training sparse rates: `0.0,0.0001,0.0003,0.001,0.003,0.005,0.01`
- token dim: `256`
- K max: `256`
- dense condition channels: building and source only

Run the first-stage evaluation after training:

```bash
cd /path/to/your/rmfm
bash scripts/run_token_unet_phase1_eval.sh
```

The default evaluation is small-scale and trend-focused:

- checkpoint: `/path/to/your/rmfm_runs/checkpoints/radiomapseer_token_unet_flow_dpm/best.pt`
- gain modes: `DPM`
- split: `test`
- samples: one sample per test city (`NUM_SAMPLES=-1`, `SAMPLES_PER_CITY=1`)
- rates: `0.0001, 0.0003, 0.001, 0.003, 0.01`
- sampler modes: `no_dc` and `dc`
- condition modes: `sampling_only`, `sampling_building`, `sampling_source`, `sampling_building_source`
- reference: `building_source_no_sampling`
- dtype: `fp32`

The output layout is:

```text
output_dir/
  token_unet_no_dc/
    no_building_source/
      DPM/
        sr_0p0001/
          input/
          recon/
          label/
          masks/
          metrics.csv
          summary.json
  token_unet_dc/
    ...
  token_unet_phase1_summary.csv
  run_config.json
  run.log
  run.pid
  run_command.txt
  launcher_config.txt
```

First-stage success criteria are trend-based:

- `TokenUNet-noDC` should improve over baseline `sampling_only`.
- `TokenUNet+DC` should improve over `TokenUNet-noDC`.
- `sampling_only` should collapse less severely at very low rates.
- `sampling_building` should clearly improve over `sampling_only`.
- `sampling_building_source` should not lose SSIM because of sparse guidance.

To quantify whether sparse sampling adds information beyond condition-only
priors, run the v1 supplementary evaluation:

```bash
cd /path/to/your/rmfm
bash scripts/run_token_unet_v1_supplement_eval.sh
```

This launcher runs in the background by default and writes `run.log`,
`run.pid`, `run_command.txt`, and `launcher_config.txt` into:

```text
/path/to/your/rmfm_runs/results/token_unet_v1_supplement_per_city1_fp32
```

It adds matched no-sampling baselines:

- `no_condition_no_sampling`
- `building_no_sampling`
- `source_no_sampling`
- `building_source_no_sampling`

and compares them with:

- `sampling_only`
- `sampling_building`
- `sampling_source`
- `sampling_building_source`

The default supplement uses `DPM`, `test`, `fp32`, one sample per test city,
and rates `0.0001, 0.001, 0.01`. It also runs a bounded DC sweep for
`sampling_only` and `sampling_building` with step sizes `1, 3, 5, 10, 20, 50`.

`sample_token_flowdps.py` records global metrics and additional masked metrics:

- `observed_*`: error on sampled pixels against the clean label.
- `unobserved_*`: error on unsampled pixels against the clean label.
- `measurement_*`: consistency with the noisy sparse measurement at sampled pixels.

The masked metrics include PSNR, MSE, NMSE, RMSE, and MAE. SSIM is reported only
for the full image because masked SSIM is not well-defined for arbitrary sparse
point sets.

Multiple checkpoints:

```bash
cd /path/to/your/rmfm
python scripts/benchmark_checkpoints.py \
  --checkpoint_dir /path/to/your/rmfm_runs/checkpoints/radiomapseer_unet_flow_dpm \
  --output_dir /path/to/your/rmfm_runs/results/checkpoint_sweep_dpm \
  --dataset_root /path/to/your/RadioMapSeer \
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
  --checkpoint_dir /path/to/your/rmfm_runs/checkpoints/radiomapseer_unet_flow_dpm \
  --checkpoints \
    /path/to/your/rmfm_runs/checkpoints/radiomapseer_unet_flow_dpm/checkpoint_step_0010000.pt \
    /path/to/your/rmfm_runs/checkpoints/radiomapseer_unet_flow_dpm/checkpoint_step_0015000.pt \
  --output_dir /path/to/your/rmfm_runs/results/checkpoint_sweep_selected \
  --dataset_root /path/to/your/RadioMapSeer \
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
cd /path/to/your/rmfm
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

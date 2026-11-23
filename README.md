# Food500Cap Text-to-Image Experiment

Two groups are compared on the same test captions and the same generation
settings:

| Group | What it is |
| --- | --- |
| `sd` | the original pretrained SDXL, used as it is |
| `sd_lora` | the same SDXL, fine-tuned on Food500Cap with LoRA |

Food500Cap captions are long, so every caption token is kept: a long caption is
split into CLIP-sized chunks, and both groups and the training loop use the same
encoding function.

Two metrics are reported. **CLIPScore** measures how well an image matches its
caption; higher is better. **FID** measures the distance between the generated
image distribution and the real image distribution; lower is better.

## 1. Requirements

Verified working stack (this is what produced the results below):

- Python **3.12**
- PyTorch **2.15.0** + torchvision **0.29.2**, CUDA **13.2**
- 1x NVIDIA A16 GPU, Linux

A CUDA GPU is required. Generation, evaluation and training all default to
`cuda`. Apple Silicon / MPS is not supported by this code, and a full run is not
realistic on a laptop: each group generates **4,938** 1024x1024 images and LoRA
training runs **5,000** steps.

Install PyTorch first, matched to your CUDA driver, then the rest:

```bash
pip install torch==2.15.0 torchvision==0.29.2
pip install -r requirements.txt
```

## 2. Get the data

**The dataset is not in this repository.** A fresh clone has no `data/` folder,
so this is the first thing to run:

```bash
python data.py download
```

This downloads Food500Cap into `data/raw/Food500Cap/` (pinned to dataset
revision `1b2e0cb0`). Running `data.py` without this step will fail, because
`data.py` with no arguments only audits data that is already on disk.

Model weights (SDXL, CLIP, Inception) are downloaded automatically the first
time you run generation or evaluation.

## 3. Run the experiment

Run the three scripts in order. Let each one finish before starting the next.

```bash
python data.py download    # once, see step 2
python data.py             # audit data, drop train images that duplicate test, write splits
python sd.py               # generate + score the original SDXL
python sd_lora.py          # train LoRA, then generate + score with the best checkpoint
```

`python data.py` prints `PREPARED` when the splits are ready. Once both groups
have scores, the comparison report is written automatically.

All three scripts read `config.yaml` from the same directory as the script, so
the working directory does not matter.

### In an IDE (PyCharm or VS Code)

Opening the `final` folder and using **Run** on a file with no arguments does
the full default job for that file, exactly like the commands above. The one
exception is the dataset download, which has no no-argument mode — run
`python data.py download` in a terminal first.

### Subcommands

If you want to run one stage at a time:

```bash
python data.py {download,audit,verify,prepare}
python sd.py {generate,evaluate,report,all} [--split val|test] [--limit N]
python sd_lora.py train [--resume [CHECKPOINT]]
python sd_lora.py {generate,evaluate,all} [--adapter DIR] [--split val|test] [--limit N]
```

`--limit N` writes to a separate diagnostic output directory and is for smoke
tests only; leave it off for the final scores. Use it to check the pipeline
end to end without waiting for 4,938 images:

```bash
python sd.py all --limit 8
```

### Resuming and multiple GPUs

Training checkpoints every 500 steps. If training is interrupted, running
`python sd_lora.py` again resumes from the newest complete checkpoint. With the
explicit subcommand, use `python sd_lora.py train --resume`.

For multiple GPUs, launch the explicit `train` subcommand through Accelerate:

```bash
accelerate launch sd_lora.py train
```

The no-argument IDE mode is single-device only and refuses to start if
`WORLD_SIZE` is not 1.

### Running on a remote GPU cluster

Copy the code up without the local virtual environment, then run it there:

```bash
rsync -av --exclude .venv --exclude outputs --exclude data ./ user@cluster:~/final/
ssh user@cluster
cd ~/final && python data.py download && python data.py && python sd.py
```

Long jobs should run under `tmux` or `screen` so they survive a dropped SSH
connection.

## 4. Results

Output goes to `outputs/food500cap_sdxl/`.

| File | What it holds |
| --- | --- |
| `sd/test/seed_2026/metrics.json` | CLIPScore and FID for the original SDXL |
| `sd_lora/test/seed_2026/metrics.json` | the same for SDXL + LoRA |
| `*/test/seed_2026/clipscore_per_sample.csv` | per-image CLIPScore |
| `*/test/seed_2026/images/` | the generated images |
| `*/test/seed_2026/environment.json` | exact package versions, GPU, config and source hashes of that run |
| `comparison/test/results.csv` | both groups' scores side by side |
| `comparison/test/comparison.png` | 8 random captions, both groups |
| `comparison/test/sd_{high,low}_clipscore.png` | best and worst 8 captions, ranked by the original SDXL |
| `comparison/test/sd_lora_{high,low}_clipscore.png` | best and worst 8 captions, ranked by the LoRA group |
| `sd_lora/checkpoints/best/` | the LoRA weights chosen on the validation split |

Each row of a comparison grid shows the real image, both generated images, and
each one's CLIPScore. The number of rows is `evaluation.comparison_samples` in
`config.yaml`. FID scores a whole population, so it is not used for per-image
ranking; the high/low grids are ranked by CLIPScore.

### Current status

The `sd` group has finished on the full test split:

| Group | n | CLIPScore | FID |
| --- | --- | --- | --- |
| `sd` (original SDXL) | 4,938 | **33.07** | **25.45** |
| `sd_lora` | - | not run yet | not run yet |

Only a few sample images are committed here; the full 4,938-image sets stay on
the machine that generated them. LoRA checkpoints are not committed either (see
`.gitignore`). The comparison report appears once the `sd_lora` group is scored.

## 5. Files

| File | Purpose |
| --- | --- |
| `config.yaml` | all settings, shared by both groups. Relative paths resolve against this file |
| `data.py` | download, audit and split Food500Cap; dataset loading used by the other scripts |
| `sd.py` | caption encoding, generation, CLIPScore, FID, comparison report |
| `sd_lora.py` | LoRA training loop, validation-based checkpoint selection, LoRA inference |

The values in `config.yaml` are initial settings, not hyperparameters tuned on
finished runs. If you change them, point `experiment.output_dir` at a new
directory: the scripts refuse to mix results from different settings.

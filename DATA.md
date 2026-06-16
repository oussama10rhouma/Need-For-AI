# Data

## Dataset

The face crops and labels used in this project are the **IDEMIA Face
Occlusion Prediction** data-challenge dataset. They are **not redistributed**
from this repository: the dataset is governed by the challenge's terms of
use and must be obtained through the official challenge channel.

Roughly:

- ~100k aligned face crops at 224×224 (`.webp`), one per row of the CSVs.
- A `train.csv` with `filename`, `gender ∈ {0, 1}`, and a continuous
  `FaceOcclusion` score in `[0, 1]`.
- A `test_students.csv` with the same `filename` column (no labels).

## Where to put it

Drop the unpacked dataset into the repository root so the relative paths
used by `src/final_pipeline.py` resolve directly:

```
face-occlusion-idemia/
├── occlusion_datasets/
│   ├── train.csv
│   └── test_students.csv
└── Crop_224_5fp_100K/
    └── database3/database3/<id>/<crop>.webp
```

These two directories are listed in `.gitignore`.

## Checkpoints

The training notebooks and `final_pipeline.py` write checkpoints and OOF
caches into the following directories (also gitignored):

- `checkpoints_v2/` — v2 ensemble (5-fold EVA-02 / ConvNeXt-Base / EffNetV2-S)
- `checkpoints_v1/` — earlier partial-fold ConvNeXt / EVA runs
- `models_efficientnet/` — earlier EffNetV2-S checkpoints (`best_model_fold*.pth`)
- `checkpoints_fair/` — fair-loss EVA-02 retrains
- `final_cache/` — cached OOF + test predictions, keyed by source name

`final_pipeline.py` is resume-safe: it caches every fold's output so a
killed session can pick up where it left off the next time it runs.

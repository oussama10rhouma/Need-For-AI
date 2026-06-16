# Face Occlusion Prediction — IDEMIA Data Challenge

> Predicting a continuous face-occlusion score in `[0, 1]` from 224×224
> aligned face crops, under a fairness-penalised metric.
> Team **Need for AI** finished near the top of the public leaderboard
> with a final score of **0.00093**.

---

## Task and metric

For each 224×224 face crop the model emits a single scalar in `[0, 1]`
estimating how occluded the face is. The challenge metric penalises
*both* prediction error and the gap between male and female error:

```text
score   = (err_female + err_male) / 2  +  | err_female − err_male |
err_g   = Σ w · (pred − target)²  /  Σ w     with   w = 1/30 + target
```

Lower is better. The `1/30 + target` weighting up-weights heavily
occluded faces, and the `|err_F − err_M|` term means halving overall
error doesn't help if it widens the gender gap.

---

## Results

| Submission                           | Public LB | Notes                                                                 |
| ------------------------------------ | --------- | --------------------------------------------------------------------- |
| `submissions/best_mix.csv`           | **0.00093** | Final blend: `0.70 · v2-ensemble + 0.30 · old-ensemble` (this repo)  |
| `submissions/final_submission.csv`   | ~0.00098  | v2 ensemble alone (EVA-02 + ConvNeXt-Base + EffNetV2-S, OOF-weighted) |
| `submissions/old_regen_submission.csv` | ~0.00098 | Earlier equal-weight ensemble, regenerated from its checkpoints       |

The honest CAL/VAL estimate before submission was **~0.00096 (CI reaching
below 0.00092)**, and the realised leaderboard landed inside that interval —
see *Honest evaluation* below.

---

## Approach

A cross-validated ensemble of complementary backbones, blended with
weights chosen on out-of-fold (OOF) predictions, then fused with an
earlier independent ensemble.

| Backbone                                       | Resolution | Folds | Role                              |
| ---------------------------------------------- | ---------- | ----- | --------------------------------- |
| EVA-02 base (`eva02_base_patch14_224`)         | 224        | 5     | Strongest single model            |
| ConvNeXt-Base CLIP (`convnext_base.clip_laion2b`) | 224     | 5     | Second pillar, decorrelates EVA   |
| EffNetV2-S (`tf_efficientnetv2_s`)             | 224        | 5     | Diversity                         |
| EffNetV2-S (old run)                           | 224        | 5     | Re-used in old-ensemble blend     |
| ConvNeXtV2-Tiny (old run)                      | 224        | partial | Re-used in old-ensemble blend  |

Inference uses **6-view test-time augmentation** (identity, h-flip, two
brightness shifts, contrast shift, h-flip + brightness). Final blend
weights are tuned on OOF with a Dirichlet random search + coordinate
refine; weak/redundant sources are driven to ~0 automatically.

---

## Incremental ensembling under 12-hour session limits

The work was done in Kaggle / Lightning sessions capped at ~12 hours each.
That single constraint shaped the entire codebase:

- **One ensemble member per session.** A session trained one model
  version (a backbone × seed × resolution × loss variant), saved its
  checkpoints and OOF predictions, and that became one more source in
  the growing ensemble. This is why the artifacts are versioned
  (`v1`, `v2`, `hires`, `seed2`, `fair`) and why there are partial-fold
  checkpoints living alongside full 5-fold ones.

- **Resume-safe everywhere.** `final_pipeline.py` caches every fold and
  every source's OOF + test predictions to `final_cache/sources/*.npz`.
  Each training stage checkpoints after every epoch (`checkpoints_*/progress/`).
  A session that dies at hour 12 can be restarted in the next session
  and skips finished work entirely.

- **OOF is the integration layer.** Because every model is evaluated on
  the same stratified 5-fold split (seed 42), their OOF arrays are
  drop-in compatible: adding a new source is just dropping a new `.npz`
  into `final_cache/sources/` and re-running the blend stage.

This is a deliberate response to compute limits, not disorganisation —
the repo is structured so the *blend* is a quick, deterministic
post-processing step that re-runs in seconds whenever a new source
appears.

---

## Honest evaluation (CAL/VAL) — the project's differentiator

The public test labels are hidden, so the challenge metric was
re-implemented locally and validated with a strict held-out protocol
that mirrors the leaderboard:

- One fixed-seed split of OOF data into **CAL** and **VAL**.
- *Every* choice — blend weights, calibration parameters, whether to
  calibrate at all — is made looking only at CAL.
- **VAL is read once, frozen**, and is never optimised against. Because
  train and test are random splits of the same distribution, an untouched
  VAL chunk behaves like unseen test data, and its score (anchored to a
  known leaderboard result) predicts the leaderboard.

This framework drove the key calls in the final submission:

- **Per-gender calibration was rejected.** It won on CAL on every split
  (improving the score by ~+0.0001 on average), but VAL gain over five
  repeated splits averaged ~−0.0001 — it was fitting CAL noise. Disabled
  in the final blend.
- **Fair-loss retraining** (metric-as-loss EVA-02) and **higher-resolution
  add-ons** were measured to contribute within VAL noise, and were
  dropped from the final blend to keep it simple.
- The anchored leaderboard prediction (~0.00096, 90% CI reaching below
  0.00092) matched the realised result (0.00093).

The takeaway is methodological as much as numerical: a disciplined
held-out evaluation that says *when to stop* is what prevented
leaderboard overfitting and kept the final model simple.

### What didn't work (measured, not guessed)

- **Per-gender affine calibration** — within VAL noise, mean negative
  gain on held-out splits.
- **Fair-loss retraining** with metric-as-loss on EVA-02 — converged
  but added nothing measurable to the blend.
- **High-resolution (384) add-ons** — improved CAL slightly, did not
  transfer to VAL beyond noise.

Documenting these negative results is part of what kept the leaderboard
prediction honest.

---

## Reproduce the 0.00093 submission

```bash
# 0. Place the dataset (see DATA.md)
#    occlusion_datasets/{train.csv, test_students.csv}
#    Crop_224_5fp_100K/...

pip install -r requirements.txt

# 1. Build the v2 ensemble's predictions (trains every missing model;
#    otherwise just reads cached checkpoints + writes final_cache/).
python src/final_pipeline.py            # -> final_submission.csv

# 2. Regenerate the earlier independent ensemble from its checkpoints.
python src/generate_old_regen.py        # -> old_regen_submission.csv

# 3. Blend the two at the OOF-tuned ratio.
python src/generate_mix.py              # -> best_mix.csv  (leaderboard 0.00093)
```

Each step writes its CSV to the *current working directory*. Run from
the repo root to keep outputs alongside `submissions/`.

### Verify before submitting

```bash
python evaluation/verify_submission.py  # structural + ranking + CAL/VAL CI
python evaluation/mix_csv.py            # score every subset of cached OOFs
```

`verify_submission.py` is no-GPU and reads only the cached `.npz` files
plus the CSV on disk — it confirms the submission is structurally valid,
matches the intended blend, and is best-or-tied-best in a fair held-out
ranking against every other candidate on disk.

---

## Repository layout

```text
face-occlusion-idemia/
├── README.md                # this file
├── DATA.md                  # where to put the (non-redistributable) dataset
├── LICENSE                  # MIT
├── requirements.txt
├── .gitignore
│
├── src/                     # the 3-script reproduction chain
│   ├── final_pipeline.py        # trains / loads v2 ensemble + writes final_submission.csv
│   ├── generate_old_regen.py    # rebuilds the prior 0.00098 ensemble
│   └── generate_mix.py          # blends the two at OOF-tuned alpha -> best_mix.csv
│
├── notebooks/               # original training notebooks, outputs cleared
│   ├── 01_train_efficientnetv2.ipynb   # earlier EffNetV2 / ConvNeXt-V2-Tiny ensemble
│   ├── 02_train_v2_ensemble.ipynb      # EVA-02 + ConvNeXt-Base + EffNetV2-S (3-fold/multi-backbone)
│   └── 03_fusion_experiments.ipynb     # ensembling / fusion experiments
│
├── evaluation/              # analysis (no GPU, no recomputation)
│   ├── verify_submission.py     # validates submission + frozen-VAL ranking
│   └── mix_csv.py               # scores every OOF combination over 5 CAL/VAL splits
│
├── results/
│   ├── training_logs/           # the real .txt logs from each training run
│   │   ├── training_log.txt
│   │   ├── training_log_v2.txt
│   │   ├── training_log_v4.txt
│   │   └── training_log_vhires.txt
│   └── scores/
│       └── mix_csv_scores.csv   # CAL/VAL scores of every OOF subset
│
└── submissions/
    ├── best_mix.csv             # leaderboard 0.00093 (the final submission)
    ├── final_submission.csv     # v2 ensemble alone
    └── old_regen_submission.csv # regenerated 0.00098 ensemble
```

---

## License

Code is released under the MIT License — see `LICENSE`. The dataset is
not part of this repository and remains subject to the IDEMIA challenge's
terms of use (see `DATA.md`).

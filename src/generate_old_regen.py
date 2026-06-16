"""
generate_old_regen.py
----------------------
Regenerates the original 0.00098 fusion as `old_regen_submission.csv`.

The old recipe is an equal-weight average of three independently trained models:
  - EffNetV2-S          (models_efficientnet/best_model_fold*.pth)
  - ConvNeXt-Base 224   (checkpoints_v1, 3 folds)
  - EVA-02 224          (checkpoints_v1, 1 fold)

Each model's test predictions are produced with 6-view TTA, averaged across its
available folds, then the three are averaged with equal weights.

Prereq: run `final_pipeline.py` once so the cached source predictions exist under
final_cache/. This script reuses those caches (no GPU recompute needed).

    python generate_old_regen.py
"""

import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_pipeline import CFG, SRC_DIR, V1_EXTRA_DIR, df_test

def load_test(npz_path):
    return np.load(npz_path, allow_pickle=True)['test']

# EffNetV2-S test preds (cached by final_pipeline's stage_old_effnet)
eff = load_test(os.path.join(SRC_DIR, 'old_effnet.npz'))

# v1 ConvNeXt + v1 EVA test preds (cached by stage_v1_extra)
conv = load_test(os.path.join(V1_EXTRA_DIR, 'v1_convnext_base.npz'))
eva = load_test(os.path.join(V1_EXTRA_DIR, 'v1_eva02_base_patch14_224.npz'))

# equal-weight fusion — the original 0.00098 recipe
old = np.clip(np.mean([eff, conv, eva], axis=0), 0.0, 1.0)

sub = pd.DataFrame({'filename': df_test['filename'],
                    'FaceOcclusion': old, 'gender': 'x'})
sub.to_csv('old_regen_submission.csv', index=False)
print(f'old_regen_submission.csv written ({len(sub)} rows, mean {old.mean():.4f})')

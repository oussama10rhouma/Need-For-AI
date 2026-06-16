"""
generate_mix.py
---------------
Produces the final blended submission that scored 0.00093 on the leaderboard.

It mixes the two strongest, most-decorrelated submissions:
  - final_submission.csv       (the v2 ensemble: EVA-02 + ConvNeXt-Base + EffNetV2-S,
                                OOF-tuned weights, no calibration)
  - old_regen_submission.csv   (the original equal-weight 0.00098 fusion)

Weight (ALPHA = current weight) was selected on out-of-fold data with a held-out
CAL/VAL protocol — see final_pipeline.py for the honest-evaluation framework that
chose it and disproved per-gender calibration. ALPHA = 0.70 was the OOF optimum.

    python generate_mix.py
Output: best_mix.csv  (the 0.00093 leaderboard file)
"""

import numpy as np
import pandas as pd

ALPHA = 0.70   # weight on final_submission; (1-ALPHA) on old_regen. OOF-tuned.

a = pd.read_csv('final_submission.csv')
b = pd.read_csv('old_regen_submission.csv')

m = a[['filename', 'FaceOcclusion']].merge(
    b[['filename', 'FaceOcclusion']], on='filename', suffixes=('_final', '_old'))
assert len(m) == len(a), 'filename mismatch between the two submissions'

mixed = np.clip(ALPHA * m['FaceOcclusion_final'].values
                + (1 - ALPHA) * m['FaceOcclusion_old'].values, 0.0, 1.0)

sub = pd.DataFrame({'filename': m['filename'], 'FaceOcclusion': mixed, 'gender': 'x'})
sub.to_csv('best_mix.csv', index=False)
print(f'best_mix.csv written ({len(sub)} rows, mean {mixed.mean():.4f}, '
      f'alpha={ALPHA})')

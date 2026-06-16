"""
MIX_CSV — score every subset of cached OOF sources via 5 CAL/VAL splits.

Produces results/scores/mix_csv_scores.csv: each row is one combination of
candidate sources (equal-weight average), its mean and std over 5 held-out
VAL splits, and its leaderboard estimate anchored to OLD_LB = 0.00098.

The CAL/VAL framework is identical to verify_submission.py: a fixed-seed
permutation, mean over five splits, never optimizing on VAL. This makes the
ranking comparable across very different combination sizes and isolates
which subsets are actually decorrelated.

No GPU, no recomputation — only reads ./final_cache/sources/*.npz.

    python mix_csv.py
"""

import os, sys, glob, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from final_pipeline import CFG, SRC_DIR, df_train, y_all, g_all, official_score

OLD_LB = 0.00098
N_SPLITS = 5
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'results', 'scores', 'mix_csv_scores.csv')


def load_sources():
    out = {}
    for path in sorted(glob.glob(os.path.join(SRC_DIR, '*.npz'))):
        z = np.load(path)
        name = os.path.basename(path).replace('.npz', '')
        out[name] = dict(
            oof=z['oof'],
            mask=z['mask'] if 'mask' in z.files else np.ones(len(df_train), dtype=bool))
    return out


def cal_val_score(oof_vec, mask):
    y = y_all[mask]; g = g_all[mask]; p = oof_vec[mask]
    n = int(mask.sum())
    scores = []
    for si in range(N_SPLITS):
        rs = np.random.default_rng(1000 + si)
        perm = rs.permutation(n)
        v_n = max(4000, int(0.2 * n))
        v_idx = perm[:v_n]
        scores.append(official_score(p[v_idx], y[v_idx], g[v_idx])[0])
    return float(np.mean(scores)), float(np.std(scores))


def main():
    sources = load_sources()
    if not sources:
        print('No sources found in', SRC_DIR); return
    names = sorted(sources)
    print(f'Found {len(names)} sources: {names}')

    # anchor: equal-weight average of all sources -> reference val score
    common_all = np.ones(len(df_train), dtype=bool)
    for n in names: common_all &= sources[n]['mask']
    ref = np.mean([sources[n]['oof'] for n in names], axis=0)
    ref_mu, _ = cal_val_score(ref, common_all)
    anchor_ratio = OLD_LB / ref_mu if ref_mu > 0 else 1.0

    rows = []
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            mask = np.ones(len(df_train), dtype=bool)
            for n in combo: mask &= sources[n]['mask']
            if mask.sum() < 1000: continue
            mix = np.mean([sources[n]['oof'] for n in combo], axis=0)
            mu, sd = cal_val_score(mix, mask)
            rows.append(dict(combo=' + '.join(combo), k=k,
                             val_score=round(mu, 6),
                             val_std=round(sd, 6),
                             anchored_LB=round(mu * anchor_ratio, 6)))

    df = pd.DataFrame(rows).sort_values('val_score').reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f'Wrote {OUT_PATH} ({len(df)} rows)')
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()

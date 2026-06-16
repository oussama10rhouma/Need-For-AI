"""
VERIFY_SUBMISSION — confirm final_submission.csv is (a) structurally valid,
(b) exactly the intended blend, and (c) the best of the available candidates.

No GPU, no recomputation. Uses only the cached npz files and the CSVs on disk.

    python verify_submission.py
"""

import os, sys, glob
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from final_pipeline import CFG, SRC_DIR, V1_EXTRA_DIR, df_train, df_test, \
    y_all, g_all, official_score

OLD_LB = 0.00098
SUB = 'final_submission.csv'

CURRENT_W = {
    'fair_eva': 0.000,
    'old_effnet': 0.175,
    'v2_convnext_base': 0.231,
    'v2_eva02_base_patch14_224': 0.586,
    'v2_tf_efficientnetv2_s': 0.008,
}

ok = True
def check(cond, msg):
    global ok
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    ok = ok and cond

# ============ 1. STRUCTURAL INTEGRITY ============
print(f'== 1. Structure of {SUB} ==')
assert os.path.exists(SUB), f'{SUB} not found'
sub = pd.read_csv(SUB)
check(list(sub.columns) == ['filename', 'FaceOcclusion', 'gender'],
      f'columns {list(sub.columns)}')
check(len(sub) == len(df_test), f'rows {len(sub)} == test {len(df_test)}')
check(set(sub['filename']) == set(df_test['filename']),
      'filenames exactly match test_students.csv')
check((sub['filename'].values == df_test['filename'].values).all(),
      'filename ORDER matches test csv')
v = sub['FaceOcclusion']
check(v.notna().all(), 'no NaN')
check(pd.api.types.is_numeric_dtype(v), 'numeric dtype')
check(float(v.min()) >= 0.0 and float(v.max()) <= 1.0,
      f'values in [0,1] (range [{v.min():.4f}, {v.max():.4f}])')
check(0.10 < float(v.mean()) < 0.25,
      f'mean {v.mean():.4f} plausible vs train GT {y_all.mean():.4f}')

# ============ 2. IS IT EXACTLY THE INTENDED BLEND? ============
print('\n== 2. Blend reconstruction ==')
src = {}
for path in sorted(glob.glob(os.path.join(SRC_DIR, '*.npz'))):
    z = np.load(path)
    src[os.path.basename(path).replace('.npz', '')] = dict(
        test=z['test'], oof=z['oof'],
        mask=z['mask'] if 'mask' in z.files else np.ones(len(df_train), dtype=bool))
needed = [n for n, w in CURRENT_W.items() if w > 0]
if all(n in src for n in needed):
    tests = {n: src[n]['test'].copy() for n in needed}
    # replicate the pipeline's v1 merge
    for path in sorted(glob.glob(os.path.join(V1_EXTRA_DIR, '*.npz'))):
        z = np.load(path, allow_pickle=True)
        tgt = str(z['merge_into'])
        if tgt in tests:
            k = int(z['n_folds'])
            tests[tgt] = (5 * tests[tgt] + k * z['test']) / (5 + k)
    expect = np.clip(sum(CURRENT_W[n] * tests[n] for n in needed), 0, 1)
    mae = float(np.abs(expect - v.values).mean())
    check(mae < 1e-4, f'CSV == blend(weights x cached test preds), MAE {mae:.6f}')
    if mae >= 1e-4:
        print('         (caches may have been regenerated since the CSV was written '
              '— this does not invalidate the CSV itself)')
else:
    print('  SKIP  source caches missing — cannot reconstruct '
          f'(missing: {[n for n in needed if n not in src]})')

# ============ 3. BEST AMONG CANDIDATES? ============
print('\n== 3. Ranking candidates (anchored to LB=0.00098) ==')
# OOF vectors for the three known configs
v1r = torch.load(os.path.join(CFG['v1_dir'], 'all_results.pt'),
                 map_location='cpu', weights_only=False)
conv_name = [n for n in v1r['model_meta'] if 'convnext' in n][0]
conv_oof = v1r['oof_pred'][conv_name]
conv_mask = v1r['oof_filled'][conv_name].astype(bool)
zev = np.load(os.path.join(CFG['cache'], 'v1_oof_eva02.npz'))
eva_oof, eva_mask = zev['oof'], zev['mask'].astype(bool)

old_oof = (conv_oof + eva_oof + src['old_effnet']['oof']) / 3.0
cur_oof = sum(w * src[n]['oof'] for n, w in CURRENT_W.items() if w > 0)
common = conv_mask & eva_mask & src['old_effnet']['mask']
for n in needed:
    common &= src[n]['mask']
yc, gc = y_all[common], g_all[common]
s_old = official_score(old_oof[common], yc, gc)[0]

# Honest ranking: every config evaluated FROZEN on 5 held-out VAL splits.
# The mix's alpha is refit on each split's CAL portion (never on VAL), so its
# number is comparable to the frozen configs — a point estimate with alpha
# fit on all rows would be optimistically biased.
po, pc = old_oof[common], cur_oof[common]
n = int(common.sum())
val = {'final_submission.csv': [], 'old submission (0.00098)': [],
       'mixed_submission.csv': []}
for si in range(5):
    rs = np.random.default_rng(1000 + si)
    perm = rs.permutation(n)
    v_n = max(4000, int(0.2 * n))
    vi, ci = perm[:v_n], perm[v_n:]
    ba, bs = 0.5, np.inf
    for a in np.linspace(0, 1, 51):
        s_ = official_score(a * pc[ci] + (1 - a) * po[ci], yc[ci], gc[ci])[0]
        if s_ < bs: bs, ba = s_, a
    val['final_submission.csv'].append(official_score(pc[vi], yc[vi], gc[vi])[0])
    val['old submission (0.00098)'].append(official_score(po[vi], yc[vi], gc[vi])[0])
    val['mixed_submission.csv'].append(
        official_score(ba * pc[vi] + (1 - ba) * po[vi], yc[vi], gc[vi])[0])

mu_old_val = float(np.mean(val['old submission (0.00098)']))
rank = sorted((OLD_LB * float(np.mean(v_)) / mu_old_val, name)
              for name, v_ in val.items())
spread = rank[-1][0] - rank[0][0]
for pred, name in rank:
    marker = ' <-- BEST' if (pred, name) == rank[0] else ''
    print(f'  {name:28s} anchored LB ~ {pred:.6f}{marker}')
NOISE = 0.000010
if spread < NOISE * 3 or rank[0][0] + NOISE > [p for p, nm in rank if nm == 'final_submission.csv'][0]:
    print(f'  note: top candidates within noise ({spread:.6f} spread) — '
          'statistically tied')
check([p for p, nm in rank if nm == 'final_submission.csv'][0] <= rank[0][0] + NOISE,
      'final_submission.csv is best or tied-best (honest split evaluation)')

# ============ 4. LB PREDICTOR — CAL/VAL PROTOCOL ============
# Exact protocol: one fixed-seed permutation; CAL (10k) is the only data any
# choice may look at; VAL (4k) is never used to choose anything — it is read
# once, frozen, and bootstrapped for a 90% CI of what the LB will show.
print('\n== 4. LB predictor (CAL 10k / VAL 4k, frozen) ==')
CAL_N, VAL_N = 10000, 4000
rng_p = np.random.default_rng(42)
perm2 = rng_p.permutation(int(common.sum()))
cal_i = perm2[:CAL_N]
val_i = perm2[CAL_N:CAL_N + VAL_N]
y_cal, g_cal = yc[cal_i], gc[cal_i]
y_val, g_val = yc[val_i], gc[val_i]

# (a) the submission's config, frozen (weights were chosen long before this split)
pv_final = cur_oof[common][val_i]
s_val, vf, vm = official_score(pv_final, y_val, g_val)
boots = [official_score(pv_final[i], y_val[i], g_val[i])[0]
         for i in [np.random.randint(0, VAL_N, VAL_N) for _ in range(200)]]
lo, hi = np.percentile(boots, [5, 95])
print(f'  final_submission frozen VAL: {s_val:.6f} | F {vf:.6f} M {vm:.6f}')
print(f'  bootstrap 90% CI:            [{lo:.6f}, {hi:.6f}]')

# (b) protocol sanity: re-select blend weights on CAL ONLY, score frozen on VAL
names_all = [n for n in CURRENT_W if n in src]
Pm = np.stack([src[n]['oof'][common] for n in names_all])
rng_w = np.random.default_rng(0)
bs_, bw_ = np.inf, np.ones(len(names_all)) / len(names_all)
for _ in range(5000):
    w_ = rng_w.dirichlet(np.ones(len(names_all)) * 0.7)
    s_ = official_score(w_ @ Pm[:, cal_i], y_cal, g_cal)[0]
    if s_ < bs_: bs_, bw_ = s_, w_
for step in [0.05, 0.02, 0.01]:
    improved = True
    while improved:
        improved = False
        for i in range(len(names_all)):
            for d in (+step, -step):
                w2 = bw_.copy(); w2[i] = max(0.0, w2[i] + d)
                if w2.sum() == 0: continue
                w2 = w2 / w2.sum()
                s_ = official_score(w2 @ Pm[:, cal_i], y_cal, g_cal)[0]
                if s_ < bs_ - 1e-12: bs_, bw_, improved = s_, w2, True
s_val_refit = official_score(bw_ @ Pm[:, val_i], y_val, g_val)[0]
print(f'  CAL-refit weights, frozen VAL: {s_val_refit:.6f} '
      f'(close to (a) => weights are stable, not split luck)')

# scale note: in THIS challenge the LB has historically read ~0.63-0.68x the
# train-scale score (test split scores lower). Anchored translation:
r_anchor = OLD_LB / s_old
print(f'  anchored to your 0.00098: predicted LB '
      f'{s_val * r_anchor:.6f} | 90% CI [{lo * r_anchor:.6f}, {hi * r_anchor:.6f}]')
check(lo * r_anchor < 0.00097, 'CI overlaps beating 0.00097 (3rd->2nd is possible)')

# correlate any other csvs on disk against the known configs (informational)
known = {'final_submission.csv', 'mixed_submission.csv',
         'old_regen_submission.csv', 'test_predictions_old.csv'}
others = [f for f in glob.glob('*.csv') if f not in known]
for f in others:
    try:
        d = pd.read_csv(f)
        if 'FaceOcclusion' in d.columns and len(d) == len(df_test):
            m = d.merge(sub, on='filename', suffixes=('_x', '_y'))
            c = float(np.corrcoef(m['FaceOcclusion_x'], m['FaceOcclusion_y'])[0, 1])
            print(f'  info: {f}: corr to final_submission {c:.4f}')
    except Exception:
        pass

print('\n' + ('VERDICT: final_submission.csv is valid and the best candidate. SUBMIT IT.'
              if ok else
              'VERDICT: at least one check FAILED — review the lines above before submitting.'))
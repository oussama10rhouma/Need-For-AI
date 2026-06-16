"""
FINAL PIPELINE — Option A + B combined (~8-10h on L40S)

Stages (all cached in ./final_cache — rerun skips finished work):
  1. GENDER CLF   (~40min) : effnet_b0, 5-fold x 2 epochs -> OOF/test gender probs
  2. FAIR EVA     (~5.5h)  : eva02_base_224, 5 folds, metric-as-loss, simple head
  3. OLD SOURCES  (~2h)    : v2 ConvNeXt / v2 EVA / v2 effnet (OOF from all_results.pt,
                             test recomputed) + old strong run-1 EffNet (OOF+test
                             recomputed, head auto-detected from checkpoint)
  4. BLEND                 : Dirichlet random search + coordinate refine on OOF
  5. CALIBRATE + SUBMIT    : per-(predicted)gender affine, only kept if it helps OOF

Folds are identical to v2 (seed 42, same stratification) so all OOF arrays align.
"""

import subprocess, sys
def pipi(*args): subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])
try:
    import timm
    from packaging import version
    if version.parse(timm.__version__) < version.parse('1.0.19'):
        pipi('-U', 'timm')
except Exception:
    pipi('-U', 'timm')

import os, gc, math, random, warnings, time, glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from torch.amp import GradScaler, autocast
    def amp_ctx(): return autocast('cuda')
    def make_scaler(): return GradScaler('cuda')
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    def amp_ctx(): return autocast()
    def make_scaler(): return GradScaler()

import torchvision.transforms as T
import timm
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
print('torch', torch.__version__, '| timm', timm.__version__)

# ============ CONFIG ============
CFG = {
    'seed': 42,
    'image_dir': 'Crop_224_5fp_100K',
    'train_csv': 'occlusion_datasets/train.csv',
    'test_csv': 'occlusion_datasets/test_students.csv',
    'cache': './final_cache',
    'v2_results': './checkpoints_v2/all_results.pt',
    'v2_ckpt_dir': './checkpoints_v2',
    'v1_dir': './checkpoints_v1',
    # old strong run-1 EffNet checkpoints (V1 head, auto-detected):
    'old_effnet_glob': 'models_efficientnet/best_model_fold*.pth',
    'old_effnet_backbone': 'tf_efficientnetv2_s.in21k_ft_in1k',
    'n_folds': 5,
    'num_workers': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # fair EVA stage:
    'fair_model': 'eva02_base_patch14_224.mim_in22k',
    'fair_bs': 128, 'fair_lr': 6e-5, 'fair_epochs': 18, 'fair_bb_mult': 0.2,
    'fair_patience': 6, 'fair_lambda': 1.0, 'gender_aux_weight': 0.05,
    'mixup_prob': 0.3, 'mixup_alpha': 0.2,
    'ema_decay': 0.999, 'weight_decay': 1e-4,
}
DEV = CFG['device']
os.makedirs(CFG['cache'], exist_ok=True)
SRC_DIR = os.path.join(CFG['cache'], 'sources'); os.makedirs(SRC_DIR, exist_ok=True)
FAIR_DIR = './checkpoints_fair'; os.makedirs(FAIR_DIR, exist_ok=True)
PROG_DIR = os.path.join(FAIR_DIR, 'progress'); os.makedirs(PROG_DIR, exist_ok=True)

def seed_everything(seed):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
seed_everything(CFG['seed'])

# ============ DATA & FOLDS (identical to v2) ============
df_train = pd.read_csv(CFG['train_csv']).dropna().reset_index(drop=True)
df_test = pd.read_csv(CFG['test_csv']).dropna().reset_index(drop=True)
df_train['occ_bin'] = pd.cut(df_train['FaceOcclusion'], bins=10, labels=False)
df_train['strat'] = df_train['gender'].astype(int).astype(str) + '_' + df_train['occ_bin'].astype(str)
skf = StratifiedKFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG['seed'])
df_train['fold'] = -1
for f, (_, vidx) in enumerate(skf.split(df_train, df_train['strat'])):
    df_train.loc[vidx, 'fold'] = f
y_all = df_train['FaceOcclusion'].values
g_all = df_train['gender'].values
print(f'train {len(df_train):,} | test {len(df_test):,}')

# ============ SHARED PIECES ============
class FaceDS(Dataset):
    def __init__(self, df, tf, mode='train'):  # mode: train / test / gender
        self.df = df.reset_index(drop=True); self.tf = tf; self.mode = mode
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self.tf(Image.open(f"{CFG['image_dir']}/{r['filename']}").convert('RGB'))
        if self.mode == 'test':
            return img, 0
        if self.mode == 'gender':
            return img, torch.tensor(r['gender'], dtype=torch.float32)
        return img, torch.tensor(r['FaceOcclusion'], dtype=torch.float32), \
               torch.tensor(r['gender'], dtype=torch.float32)

def norm_tfs(mean, std, img_size, train=False):
    norm = T.Normalize(mean=mean, std=std)
    if not train:
        return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(0.5),
        T.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ColorJitter(0.2, 0.2, 0.15, 0.04),
        T.RandomGrayscale(p=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        T.ToTensor(), norm,
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

def hflip_tf(mean, std, img_size):
    return T.Compose([T.Resize((img_size, img_size)), T.RandomHorizontalFlip(1.0),
                      T.ToTensor(), T.Normalize(mean=mean, std=std)])

def tta6(mean, std, img_size):
    """6-view TTA — same recipe as the original 0.00098 fusion."""
    norm = T.Normalize(mean=mean, std=std)
    R = T.Resize((img_size, img_size))
    return [
        T.Compose([R, T.ToTensor(), norm]),
        T.Compose([R, T.RandomHorizontalFlip(1.0), T.ToTensor(), norm]),
        T.Compose([R, T.ColorJitter(brightness=(1.1, 1.1)), T.ToTensor(), norm]),
        T.Compose([R, T.ColorJitter(brightness=(0.9, 0.9)), T.ToTensor(), norm]),
        T.Compose([R, T.ColorJitter(contrast=(1.1, 1.1)), T.ToTensor(), norm]),
        T.Compose([R, T.RandomHorizontalFlip(1.0),
                   T.ColorJitter(brightness=(1.05, 1.05)), T.ToTensor(), norm]),
    ]

def model_norm(name):
    m = timm.create_model(name, pretrained=False, num_classes=0)
    dc = timm.data.resolve_model_data_config(m)
    del m; gc.collect()
    return dc['mean'], dc['std']

def weighted_err(pred, target):
    w = 1.0 / 30.0 + target
    return float(np.sum(w * (pred - target) ** 2) / np.sum(w))

def official_score(pred, target, gender):
    mf, mm = gender == 0.0, gender == 1.0
    ef = weighted_err(pred[mf], target[mf]) if mf.sum() else 0.0
    em = weighted_err(pred[mm], target[mm]) if mm.sum() else 0.0
    return (ef + em) / 2.0 + abs(ef - em), ef, em

def gender_balanced_sampler(df):
    counts = df['gender'].value_counts().to_dict()
    w = df['gender'].map(lambda g: 1.0 / counts[g]).values
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(df), replacement=True)

# =====================================================================
# STAGE 1 — GENDER CLASSIFIER (~40 min)
# =====================================================================
GENDER_NPZ = os.path.join(CFG['cache'], 'gender.npz')

def stage_gender():
    if os.path.exists(GENDER_NPZ):
        z = np.load(GENDER_NPZ)
        print(f"[gender] cached | OOF acc {z['acc']:.4f}")
        return z['oof'], z['test']
    print('[gender] training effnet_b0 gender classifier (5 folds x 2 epochs)...')
    name = 'tf_efficientnet_b0.ns_jft_in1k'
    mean, std = model_norm(name)
    tr_tf = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(0.5),
                       T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(),
                       T.Normalize(mean=mean, std=std)])
    va_tf = norm_tfs(mean, std, 224)
    oof = np.zeros(len(df_train)); test_acc_probs = []
    for fold in range(CFG['n_folds']):
        tr = df_train[df_train.fold != fold]; va = df_train[df_train.fold == fold]
        model = timm.create_model(name, pretrained=True, num_classes=1).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scaler = make_scaler()
        tl = DataLoader(FaceDS(tr, tr_tf, 'gender'), batch_size=256, shuffle=True,
                        num_workers=CFG['num_workers'], pin_memory=True, drop_last=True)
        for ep in range(2):
            model.train()
            for x, gb in tqdm(tl, desc=f'[gender] f{fold} ep{ep+1}', leave=False):
                x, gb = x.to(DEV), gb.to(DEV)
                with amp_ctx():
                    loss = F.binary_cross_entropy_with_logits(model(x).squeeze(-1), gb)
                opt.zero_grad(); scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
        @torch.no_grad()
        def infer(df, mode):
            model.eval()
            dl = DataLoader(FaceDS(df, va_tf, mode), batch_size=512, shuffle=False,
                            num_workers=CFG['num_workers'], pin_memory=True)
            out = []
            for x, _ in dl:
                with amp_ctx():
                    out.append(torch.sigmoid(model(x.to(DEV)).squeeze(-1)).float().cpu().numpy())
            return np.concatenate(out)
        oof[va.index.values] = infer(va, 'gender')
        test_acc_probs.append(infer(df_test, 'test'))
        acc_f = float((((oof[va.index.values]) > 0.5) == (va['gender'].values > 0.5)).mean())
        print(f'[gender] fold {fold} acc {acc_f:.4f}')
        del model, opt; gc.collect(); torch.cuda.empty_cache()
    test_prob = np.mean(test_acc_probs, axis=0)
    acc = float(((oof > 0.5) == (g_all > 0.5)).mean())
    print(f'[gender] OOF accuracy {acc:.4f}')
    np.savez(GENDER_NPZ, oof=oof, test=test_prob, acc=acc)
    return oof, test_prob

# =====================================================================
# STAGE 2 — FAIR EVA-02 224 (~5.5h) : metric-as-loss + simple head
# =====================================================================
class FairModel(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        feat = self.backbone.num_features
        self.head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(0.2),
                                  nn.Linear(feat, 256), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(256, 1))
        self.gender_head = nn.Linear(feat, 1)
    def forward(self, x):
        f = self.backbone(x)
        return torch.sigmoid(self.head(f)).squeeze(-1), self.gender_head(f).squeeze(-1)

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)
    def copy_to(self, model): model.load_state_dict(self.shadow, strict=True)

def _soft_group_err(pred, target, gmask):
    w = 1.0 / 30.0 + target
    return (gmask * w * (pred - target) ** 2).sum() / (gmask * w).sum().clamp(min=1e-6)

def fair_loss(pred, target, g, lam):
    ef = _soft_group_err(pred, target, 1.0 - g)
    em = _soft_group_err(pred, target, g)
    return 0.5 * (ef + em) + lam * torch.sqrt((ef - em) ** 2 + 1e-10)

def mixup(x, yb, gb, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * yb + (1 - lam) * yb[idx], lam * gb + (1 - lam) * gb[idx]

def fair_prog_path(fold): return os.path.join(PROG_DIR, f'fair_f{fold}.pt')

def train_fair_fold(fold, mean, std):
    name = CFG['fair_model']
    tr = df_train[df_train.fold != fold]; va = df_train[df_train.fold == fold]
    print(f"\n[fair] {name} fold {fold} | train {len(tr):,} val {len(va):,}")
    tr_tf = norm_tfs(mean, std, 224, train=True); va_tf = norm_tfs(mean, std, 224)
    tl = DataLoader(FaceDS(tr, tr_tf), batch_size=CFG['fair_bs'],
                    sampler=gender_balanced_sampler(tr),
                    num_workers=CFG['num_workers'], pin_memory=True, drop_last=True)
    vl = DataLoader(FaceDS(va, va_tf), batch_size=CFG['fair_bs'] * 2, shuffle=False,
                    num_workers=CFG['num_workers'], pin_memory=True)
    model = FairModel(name).to(DEV).to(memory_format=torch.channels_last)
    head_p = [p for n_, p in model.named_parameters() if n_.startswith(('head', 'gender_head'))]
    bb_p = [p for n_, p in model.named_parameters() if not n_.startswith(('head', 'gender_head'))]
    opt = torch.optim.AdamW([
        {'params': bb_p, 'lr': CFG['fair_lr'] * CFG['fair_bb_mult'], 'weight_decay': CFG['weight_decay']},
        {'params': head_p, 'lr': CFG['fair_lr'], 'weight_decay': CFG['weight_decay']}])
    total_steps = len(tl) * CFG['fair_epochs']
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[grp['lr'] for grp in opt.param_groups], total_steps=total_steps,
        pct_start=min(0.3, len(tl) / total_steps), anneal_strategy='cos',
        div_factor=10, final_div_factor=100)
    scaler = make_scaler()
    ema = EMA(model, CFG['ema_decay'])
    best = math.inf; best_oof = None; patience = 0; start_ep = 0
    ckpt_path = os.path.join(FAIR_DIR, f'best_fair_f{fold}.pth')

    if os.path.exists(fair_prog_path(fold)):
        st = torch.load(fair_prog_path(fold), map_location='cpu', weights_only=False)
        start_ep = st['epoch'] + 1
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched']); scaler.load_state_dict(st['scaler'])
        ema.shadow = {k: v.to(DEV) for k, v in st['ema'].items()}
        best, best_oof, patience = st['best'], st['best_oof'], st['patience']
        print(f'[fair] resumed fold {fold} @ epoch {start_ep} (best {best:.6f})')
        del st; gc.collect()

    @torch.no_grad()
    def evaluate():
        tmp = FairModel(name, pretrained=False).to(DEV).to(memory_format=torch.channels_last)
        ema.copy_to(tmp); tmp.eval()
        P, Tt, G = [], [], []
        for x, yb, gb in vl:
            x = x.to(DEV, memory_format=torch.channels_last)
            with amp_ctx():
                p, _ = tmp(x)
            P.append(p.float().cpu().numpy()); Tt.append(yb.numpy()); G.append(gb.numpy())
        del tmp
        P, Tt, G = map(np.concatenate, (P, Tt, G))
        return official_score(P, Tt, G) + (P,)

    for ep in range(start_ep, CFG['fair_epochs']):
        model.train(); rl = 0.0
        pbar = tqdm(tl, desc=f'[fair] f{fold} ep {ep+1}/{CFG["fair_epochs"]}')
        for bi, (x, yb, gb) in enumerate(pbar):
            x = x.to(DEV, memory_format=torch.channels_last); yb = yb.to(DEV); gb = gb.to(DEV)
            if random.random() < CFG['mixup_prob']:
                x, yb, gb = mixup(x, yb, gb, CFG['mixup_alpha'])
            with amp_ctx():
                pred, gl = model(x)
                loss = fair_loss(pred, yb, gb, CFG['fair_lambda']) \
                     + CFG['gender_aux_weight'] * F.binary_cross_entropy_with_logits(gl, gb)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            rl += loss.item(); pbar.set_postfix(loss=f'{rl/(bi+1):.5f}')
        s, ef, em, oof = evaluate()
        print(f'[fair] score {s:.6f} | F {ef:.6f} M {em:.6f} | gap {abs(ef-em):.6f}')
        if s < best:
            best, best_oof, patience = s, oof, 0
            torch.save(ema.shadow, ckpt_path)
            print(f'[fair] >> NEW BEST {best:.6f}')
        else:
            patience += 1
        torch.save({'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                    'ema': ema.shadow, 'best': best, 'best_oof': best_oof,
                    'patience': patience}, fair_prog_path(fold))
        if patience >= CFG['fair_patience']:
            print(f'[fair] early stop @ ep {ep+1}'); break
    if os.path.exists(fair_prog_path(fold)): os.remove(fair_prog_path(fold))
    del model, opt, sched, scaler, tl, vl
    gc.collect(); torch.cuda.empty_cache()
    return best, best_oof, va.index.values, ckpt_path

@torch.no_grad()
def predict_generic(model, df, tfs, bs=256):
    model.eval()
    acc = None
    for tf in tfs:
        dl = DataLoader(FaceDS(df, tf, 'test'), batch_size=bs, shuffle=False,
                        num_workers=CFG['num_workers'], pin_memory=True)
        out = []
        for x, _ in tqdm(dl, leave=False):
            x = x.to(DEV, memory_format=torch.channels_last)
            with amp_ctx():
                r = model(x)
                p = r[0] if isinstance(r, tuple) else r
            out.append(p.float().cpu().numpy())
        p = np.concatenate(out)
        acc = p if acc is None else acc + p
    return acc / len(tfs)

def stage_fair():
    npz = os.path.join(SRC_DIR, 'fair_eva.npz')
    if os.path.exists(npz):
        print('[fair] cached'); return
    name = CFG['fair_model']; mean, std = model_norm(name)
    res_path = os.path.join(FAIR_DIR, 'fair_results.pt')
    if os.path.exists(res_path):
        R = torch.load(res_path, map_location='cpu', weights_only=False)
    else:
        R = {'oof': np.zeros(len(df_train)), 'done': [], 'ckpts': {}}
    for fold in range(CFG['n_folds']):
        if fold in R['done']: continue
        best, oof, vidx, ckpt = train_fair_fold(fold, mean, std)
        R['oof'][vidx] = oof; R['done'].append(fold); R['ckpts'][fold] = ckpt
        torch.save(R, res_path)
        print(f'[fair] fold {fold} done: {best:.6f} (saved)')
    s, ef, em = official_score(R['oof'], y_all, g_all)
    print(f'[fair] full OOF: {s:.6f} | gap {abs(ef-em):.6f}')
    # test preds: 2-view TTA (none + hflip), averaged over folds
    tfs = tta6(mean, std, 224)
    preds = []
    for fold in range(CFG['n_folds']):
        m = FairModel(name, pretrained=False).to(DEV).to(memory_format=torch.channels_last)
        m.load_state_dict(torch.load(R['ckpts'][fold], map_location=DEV, weights_only=True), strict=True)
        preds.append(predict_generic(m, df_test, tfs))
        del m; gc.collect(); torch.cuda.empty_cache()
    np.savez(npz, oof=R['oof'], test=np.mean(preds, axis=0))
    print('[fair] source saved')

# =====================================================================
# STAGE 3 — OLD SOURCES (v2 models + old strong run-1 EffNet)
# =====================================================================
class V2Model(nn.Module):  # the deep v2 head
    def __init__(self, name):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=False, num_classes=0)
        feat = self.backbone.num_features
        self.head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(0.3),
                                  nn.Linear(feat, 512), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(128, 1))
    def forward(self, x):
        return torch.sigmoid(self.head(self.backbone(x))).squeeze(-1)

def candidate_heads(feat):
    return {
        'v2_deep': nn.Sequential(nn.LayerNorm(feat), nn.Dropout(0.3),
                                 nn.Linear(feat, 512), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(128, 1)),
        'ln_512_1': nn.Sequential(nn.LayerNorm(feat), nn.Dropout(0.2),
                                  nn.Linear(feat, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, 1)),
        'ln_256_1': nn.Sequential(nn.LayerNorm(feat), nn.Dropout(0.2),
                                  nn.Linear(feat, 256), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(256, 1)),
        '512_1': nn.Sequential(nn.Linear(feat, 512), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(512, 1)),
        '256_1': nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.2),
                               nn.Linear(256, 1)),
        'drop_256_gelu': nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, 256), nn.GELU(),
                                       nn.Dropout(0.2), nn.Linear(256, 1)),
        'drop_256_relu': nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, 256), nn.ReLU(),
                                       nn.Dropout(0.2), nn.Linear(256, 1)),
        'linear': nn.Sequential(nn.Linear(feat, 1)),
    }

class OldModel(nn.Module):
    def __init__(self, backbone_name, head):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.head = head
    def forward(self, x):
        return torch.sigmoid(self.head(self.backbone(x))).squeeze(-1)

def build_old_model(sd, bb_name=None):
    """Return ALL candidate heads whose param structure matches the state_dict."""
    if bb_name is None:
        bb_name = CFG['old_effnet_backbone']
    probe = timm.create_model(bb_name, pretrained=False, num_classes=0)
    feat = probe.num_features; del probe
    head_sd = {k: v for k, v in sd.items() if not k.startswith('backbone.')}
    got_keys = {k: tuple(v.shape) for k, v in head_sd.items()}
    matches = []
    for hname, head in candidate_heads(feat).items():
        cand_keys = {f'head.{k}': tuple(v.shape) for k, v in head.state_dict().items()}
        if cand_keys == got_keys:
            m = OldModel(bb_name, head)
            m.load_state_dict(sd, strict=True)
            matches.append((hname, m))
    if not matches:
        print('[old] !! no candidate head matched. Checkpoint non-backbone keys:')
        for k, v in list(head_sd.items())[:25]:
            print(f'      {k}: {tuple(v.shape)}')
    return matches

def pick_old_model(sd, bb_name, va_df, va_tf, chosen=None):
    """If several heads match structurally (e.g. ReLU vs GELU), pick by val error."""
    cands = build_old_model(sd, bb_name)
    if not cands:
        return None, None
    if chosen is not None:
        for h, m in cands:
            if h == chosen:
                return m, chosen
    if len(cands) == 1:
        print(f'[old] head matched: {cands[0][0]}')
        return cands[0][1], cands[0][0]
    sub = va_df.iloc[:2000]
    yv = sub['FaceOcclusion'].values
    best_e, best_m, best_h = math.inf, None, None
    for h, m in cands:
        m = m.to(DEV).to(memory_format=torch.channels_last)
        e = weighted_err(predict_generic(m, sub, [va_tf]), yv)
        print(f'[old] candidate {h}: val err {e:.6f}')
        if e < best_e:
            if best_m is not None: best_m.cpu()
            best_e, best_m, best_h = e, m, h
        else:
            m.cpu()
        gc.collect(); torch.cuda.empty_cache()
    print(f'[old] selected head: {best_h}')
    return best_m, best_h

def stage_old_effnet():
    npz = os.path.join(SRC_DIR, 'old_effnet.npz')
    if os.path.exists(npz):
        print('[old] cached'); return
    import re
    ckpts = sorted(glob.glob(CFG['old_effnet_glob']))
    if not ckpts:
        print(f'[old] no ckpts found ({CFG["old_effnet_glob"]}) — SKIPPING'); return
    pairs = []
    for ck in ckpts:
        mt = re.search(r'fold[_]?(\d+)', os.path.basename(ck))
        if mt: pairs.append((int(mt.group(1)), ck))
    if len(pairs) != len(ckpts):
        print('[old] could not parse fold index from some filenames — assuming order 0..n')
        pairs = list(enumerate(ckpts))
    print(f'[old] using {len(pairs)} fold ckpts: folds {[f for f, _ in sorted(pairs)]}')
    mean, std = model_norm(CFG['old_effnet_backbone'])
    va_tf = norm_tfs(mean, std, 224)
    tfs_test = tta6(mean, std, 224)
    oof = np.zeros(len(df_train)); mask = np.zeros(len(df_train), dtype=bool)
    test_preds = []; chosen = None
    for fold, ck in sorted(pairs):
        sd = torch.load(ck, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'model_state' in sd: sd = sd['model_state']
        if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
        va = df_train[df_train.fold == fold]
        m, chosen = pick_old_model(sd, CFG['old_effnet_backbone'], va, va_tf, chosen)
        if m is None:
            print('[old] aborting source (paste the key printout to adapt the head)'); return
        m = m.to(DEV).to(memory_format=torch.channels_last)
        oof[va.index.values] = predict_generic(m, va, [va_tf])
        mask[va.index.values] = True
        test_preds.append(predict_generic(m, df_test, tfs_test))
        sf, _, _ = official_score(oof[va.index.values], y_all[va.index.values], g_all[va.index.values])
        print(f'[old] fold {fold} OOF {sf:.6f}')
        del m; gc.collect(); torch.cuda.empty_cache()
    s, ef, em = official_score(oof[mask], y_all[mask], g_all[mask])
    print(f'[old] OOF on covered {mask.sum():,} rows: {s:.6f} | gap {abs(ef-em):.6f}')
    np.savez(npz, oof=oof, test=np.mean(test_preds, axis=0), mask=mask)

def stage_v2_sources():
    if not os.path.exists(CFG['v2_results']):
        print('[v2] all_results.pt not found — SKIPPING v2 sources'); return
    saved = torch.load(CFG['v2_results'], map_location='cpu', weights_only=False)
    oof_pred, oof_filled, meta = saved['oof_pred'], saved['oof_filled'], saved['model_meta']
    for name, m in meta.items():
        tag = 'v2_' + name.split('.')[0]
        npz = os.path.join(SRC_DIR, f'{tag}.npz')
        if os.path.exists(npz):
            print(f'[v2] {tag} cached'); continue
        if not oof_filled[name].all():
            print(f'[v2] {name}: OOF incomplete ({oof_filled[name].sum()}/{len(df_train)}) — skipping')
            continue
        s, ef, em = official_score(oof_pred[name], y_all, g_all)
        print(f'[v2] {name} OOF {s:.6f} — computing test preds...')
        mean, std = m['mean'], m['std']
        tfs = tta6(mean, std, 224)
        preds = []
        for r in m['results']:
            mod = V2Model(name).to(DEV).to(memory_format=torch.channels_last)
            mod.load_state_dict(torch.load(r['ckpt'], map_location=DEV, weights_only=True), strict=True)
            preds.append(predict_generic(mod, df_test, tfs))
            del mod; gc.collect(); torch.cuda.empty_cache()
        np.savez(npz, oof=oof_pred[name], test=np.mean(preds, axis=0))
        print(f'[v2] {tag} saved')

# ---- v1 checkpoints (partial folds): merged into matching v2 sources at test time ----
V1_EXTRA_DIR = os.path.join(CFG['cache'], 'v1_extra')

def stage_v1_extra():
    os.makedirs(V1_EXTRA_DIR, exist_ok=True)
    import re
    ckpts = sorted(glob.glob(os.path.join(CFG['v1_dir'], 'best_*.pth')))
    if not ckpts:
        print('[v1] no checkpoints found — skipping'); return
    groups = {}
    for ck in ckpts:
        mt = re.match(r'best_(.+)_f(\d+)\.pth$', os.path.basename(ck))
        if mt: groups.setdefault(mt.group(1), []).append((int(mt.group(2)), ck))
    res_path = os.path.join(CFG['v1_dir'], 'all_results.pt')
    if os.path.exists(res_path):
        sv = torch.load(res_path, map_location='cpu', weights_only=False)
        for name in sv.get('model_meta', {}):
            mask = sv['oof_filled'][name]
            if mask.sum():
                s, _, _ = official_score(sv['oof_pred'][name][mask], y_all[mask], g_all[mask])
                print(f'[v1] {name} partial OOF ({mask.sum():,} rows): {s:.6f}')
    for name, lst in groups.items():
        tag = 'v1_' + name.split('.')[0]
        npz = os.path.join(V1_EXTRA_DIR, f'{tag}.npz')
        if os.path.exists(npz):
            print(f'[v1] {tag} cached'); continue
        mean, std = model_norm(name)
        tfs = tta6(mean, std, 224)
        preds = []; chosen = None
        for fold, ck in sorted(lst):
            sd = torch.load(ck, map_location='cpu', weights_only=False)
            if isinstance(sd, dict) and 'model_state' in sd: sd = sd['model_state']
            if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
            m, chosen = pick_old_model(sd, name, df_train[df_train.fold == fold], tfs[0], chosen)
            if m is None:
                print(f'[v1] {name}: head not matched — skipping this model'); preds = []; break
            m = m.to(DEV).to(memory_format=torch.channels_last)
            preds.append(predict_generic(m, df_test, tfs))
            print(f'[v1] {tag} fold {fold} done')
            del m; gc.collect(); torch.cuda.empty_cache()
        if preds:
            np.savez(npz, test=np.mean(preds, axis=0), n_folds=len(preds),
                     merge_into='v2_' + name.split('.')[0])
            print(f'[v1] {tag}: {len(preds)} fold(s) -> merges into v2_{name.split(".")[0]}')

# =====================================================================
# STAGE 4+5 — BLEND, CALIBRATE, SUBMIT
# =====================================================================
def stage_blend_submit(gender_oof, gender_test):
    sources = {}
    for path in sorted(glob.glob(os.path.join(SRC_DIR, '*.npz'))):
        z = np.load(path)
        nm = os.path.basename(path).replace('.npz', '')
        msk = z['mask'] if 'mask' in z.files else np.ones(len(df_train), dtype=bool)
        sources[nm] = {'oof': z['oof'], 'test': z['test'], 'mask': msk}
        s, ef, em = official_score(z['oof'][msk], y_all[msk], g_all[msk])
        print(f'[blend] source {nm:30s} OOF {s:.6f} gap {abs(ef-em):.6f} '
              f'(coverage {msk.sum():,}/{len(df_train):,})')
    names = list(sources.keys())
    assert names, 'No sources found.'
    # merge v1 fold checkpoints into matching v2 sources (test preds only)
    for path in sorted(glob.glob(os.path.join(V1_EXTRA_DIR, '*.npz'))):
        z = np.load(path, allow_pickle=True)
        tgt = str(z['merge_into'])
        if tgt in sources:
            k = int(z['n_folds'])
            sources[tgt]['test'] = (5 * sources[tgt]['test'] + k * z['test']) / (5 + k)
            print(f'[blend] merged {os.path.basename(path)} ({k} folds) into {tgt}')
    common = np.ones(len(df_train), dtype=bool)
    for n in names:
        common &= sources[n]['mask']
    yA, gA = y_all[common], g_all[common]
    P = np.stack([sources[n]['oof'][common] for n in names])
    Ptest = np.stack([sources[n]['test'] for n in names])
    gprob = gender_oof[common]

    # ---- selection procedure (identical on every split) ----
    def apply_cal(pred, gh, prm):
        out = pred.copy()
        for grp in (0.0, 1.0):
            a, b = prm[grp]; mmask = gh == grp
            out[mmask] = np.clip(a * pred[mmask] + b, 0.0, 1.0)
        return out

    def fit_select(Pc, y_c, g_c, ghat_c, n_rand=20000, verbose=False, src_names=None):
        k = Pc.shape[0]
        rng = np.random.default_rng(0)
        best_s, best_w = math.inf, np.ones(k) / k
        for _ in range(n_rand):
            w = rng.dirichlet(np.ones(k) * 0.7)
            s_ = official_score(w @ Pc, y_c, g_c)[0]
            if s_ < best_s: best_s, best_w = s_, w
        for step in [0.05, 0.02, 0.01, 0.005, 0.002]:
            improved = True
            while improved:
                improved = False
                for i in range(k):
                    for d in (+step, -step):
                        w2 = best_w.copy(); w2[i] = max(0.0, w2[i] + d)
                        if w2.sum() == 0: continue
                        w2 = w2 / w2.sum()
                        s_ = official_score(w2 @ Pc, y_c, g_c)[0]
                        if s_ < best_s - 1e-12:
                            best_s, best_w, improved = s_, w2, True
        mix_c = best_w @ Pc
        s_blend = official_score(mix_c, y_c, g_c)[0]
        params = {0.0: (1.0, 0.0), 1.0: (1.0, 0.0)}
        # Calibration DISABLED: repeated-split VAL showed mean negative gain
        # (+0.000109, -0.000167, +0.000182, -0.000229, -0.000358) — it fits
        # CAL noise and does not transfer. Set ENABLE_CAL = True to re-test.
        ENABLE_CAL = False
        best_cal = s_blend
        if ENABLE_CAL:
            for _ in range(3):
                for grp in (0.0, 1.0):
                    for a in np.linspace(0.90, 1.10, 41):
                        for b in np.linspace(-0.02, 0.02, 41):
                            trial = dict(params); trial[grp] = (a, b)
                            s_ = official_score(apply_cal(mix_c, ghat_c, trial), y_c, g_c)[0]
                            if s_ < best_cal - 1e-12:
                                best_cal, params = s_, trial
        s_cal = official_score(apply_cal(mix_c, ghat_c, params), y_c, g_c)[0]
        use_cal = ENABLE_CAL and s_cal < s_blend - 1e-7
        if verbose:
            print('[blend] weights:')
            for n, w in zip(src_names or names, best_w): print(f'  {n:30s}: {w:.3f}')
            print(f'[cal] F a={params[0.0][0]:.3f} b={params[0.0][1]:+.4f} | '
                  f'M a={params[1.0][0]:.3f} b={params[1.0][1]:+.4f} | use_cal={use_cal}')
        return best_w, params, use_cal, s_blend, s_cal

    # ---- repeated CAL/VAL evaluation of the PROCEDURE ----
    # The metric is very noisy on small samples (the |ef-em| term), so one
    # split is not trustworthy. We rerun the full selection on 5 independent
    # CAL/VAL splits; mean VAL score estimates the leaderboard.
    N_SPLITS = 5
    gh_all = (gprob > 0.5).astype(float)
    val_scores = []
    for si in range(N_SPLITS):
        rs = np.random.default_rng(1000 + si)
        perm = rs.permutation(P.shape[1])
        val_n = max(4000, int(0.2 * P.shape[1]))
        v_idx, c_idx = perm[:val_n], perm[val_n:]
        w_s, prm_s, uc_s, sb, sc = fit_select(
            P[:, c_idx], yA[c_idx], gA[c_idx], gh_all[c_idx])
        pv_raw = w_s @ P[:, v_idx]
        pv = apply_cal(pv_raw, gh_all[v_idx], prm_s) if uc_s else pv_raw
        sv = official_score(pv, yA[v_idx], gA[v_idx])[0]
        sv_raw = official_score(pv_raw, yA[v_idx], gA[v_idx])[0]
        val_scores.append(sv)
        print(f'[VAL] split {si}: frozen {sv:.6f} | no-cal {sv_raw:.6f} | '
              f'cal gain on VAL {sv_raw - sv:+.6f} (CAL {sb:.6f} -> {sc:.6f})')
    mu, sd = float(np.mean(val_scores)), float(np.std(val_scores))
    print(f'\n[VAL] procedure estimate over {N_SPLITS} splits: {mu:.6f} +/- {sd:.6f}')

    # ---- ablation: same subset, same splits, WITHOUT partial-coverage sources ----
    # Partial sources shrink the common subset (e.g. to folds 0-1), which changes
    # the evaluation data itself. This isolates their true contribution.
    partial = [n for n in names if sources[n]['mask'].sum() < len(df_train)
               and not sources[n]['mask'].all()]
    keep = [i for i, n in enumerate(names) if n not in partial]
    if partial and 2 <= len(keep) < len(names):
        base_scores = []
        for si in range(N_SPLITS):
            rs = np.random.default_rng(1000 + si)
            perm = rs.permutation(P.shape[1])
            val_n = max(4000, int(0.2 * P.shape[1]))
            v_idx, c_idx = perm[:val_n], perm[val_n:]
            Pb = P[keep]
            w_b, prm_b, uc_b, _, _ = fit_select(
                Pb[:, c_idx], yA[c_idx], gA[c_idx], gh_all[c_idx])
            pb = w_b @ Pb[:, v_idx]
            if uc_b:
                pb = apply_cal(pb, gh_all[v_idx], prm_b)
            base_scores.append(official_score(pb, yA[v_idx], gA[v_idx])[0])
        mu_b = float(np.mean(base_scores))
        print(f'[VAL] ablation without {partial} on SAME subset: {mu_b:.6f} '
              f'| true contribution: {mu_b - mu:+.6f}')

    # ---- final config: fit on ALL common rows, applied to test ----
    best_w, params, use_cal, s_blend, s_cal = fit_select(
        P, yA, gA, gh_all, n_rand=40000, verbose=True)
    print(f'[final] fit-on-all CAL-equivalent score: blend {s_blend:.6f} -> cal {s_cal:.6f} '
          f'(optimistic; trust the VAL estimate above)')

    final = best_w @ Ptest
    if use_cal:
        final = apply_cal(final, (gender_test > 0.5).astype(float), params)
    final = np.clip(final, 0.0, 1.0)
    sub = pd.DataFrame({'filename': df_test['filename'], 'FaceOcclusion': final, 'gender': 'x'})
    sub.to_csv('final_submission.csv', index=False)
    print(f'\n[done] final_submission.csv ({len(sub)} rows) | '
          f'range [{final.min():.4f}, {final.max():.4f}] mean {final.mean():.4f}')
    print(f'[done] expected LB ~ {mu:.6f} +/- {sd:.6f} (modulo train/test shift)')

# =====================================================================
# RUN
# =====================================================================
if __name__ == '__main__':
    t0 = time.time()
    g_oof, g_test = stage_gender()                       # ~40 min
    stage_v2_sources()                                   # ~1h (inference only)
    stage_v1_extra()                                     # ~40min (inference only)
    stage_old_effnet()                                   # ~1h (inference only)
    stage_fair()                                         # ~5.5h (training)
    stage_blend_submit(g_oof, g_test)                    # ~minutes
    print(f'\nTotal: {(time.time()-t0)/3600:.2f} h')

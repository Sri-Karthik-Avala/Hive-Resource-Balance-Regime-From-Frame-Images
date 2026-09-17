# made by - Karthik
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

START = time.time()
HARD = 5000.0
RESERVE = 200.0
PLAN_FOLDS = 3
N_SPLITS = 6
MAX_EPOCHS = 24
MIN_EPOCHS = 4
BATCH = 32
LR = 3e-4
WD = 1e-4
SMOOTH = 0.05
IMG_H = 224
IMG_W = 320
SML_H = 176
SML_W = 256
BIG_FRAC = 0.55
N_SNAP = 3
SCALE_LO = 0.45
SCALE_HI = 0.95
TRANS_F = 0.7
TTA_SC = (1.0, 0.72)
SEED = 1234
CLASSES = ["brood_dominant", "forage_dominant", "mixed_resource", "sparse_uncertain"]
CRIT = [0, 3]

torch.manual_seed(SEED)
np.random.seed(SEED)

NCPU = 8
try:
    import os as _os

    NCPU = max(1, (_os.cpu_count() or 8))
except Exception:
    NCPU = 8
try:
    torch.set_num_threads(max(1, min(NCPU, 10)))
except Exception:
    pass

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def elapsed():
    return time.time() - START


def macro_f1(y_true, y_pred, labels):
    fs = []
    for c in labels:
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        d = 2 * tp + fp + fn
        fs.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(fs))


def cwm_f1(y_true, y_pred):
    allc = [0, 1, 2, 3]
    of1 = macro_f1(y_true, y_pred, allc)
    cf1 = macro_f1(y_true, y_pred, CRIT)
    recs = []
    for c in allc:
        m = y_true == c
        recs.append(1.0 if m.sum() == 0 else float(np.sum(y_pred[m] == c)) / float(m.sum()))
    return 0.55 * of1 + 0.25 * cf1 + 0.20 * float(np.mean(recs))


def load_images(paths, root):
    n = len(paths)
    out = np.zeros((n, IMG_H, IMG_W, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(root / p).convert("RGB")
        if im.size != (IMG_W, IMG_H):
            im = im.resize((IMG_W, IMG_H), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.uint8)
    return out


def style_feats(arr):
    a = arr.astype(np.float32) / 255.0
    h, w = a.shape[1], a.shape[2]
    bh = max(2, h // 10)
    bw = max(2, w // 12)
    fs = []
    for p in (a[:, :bh, :, :], a[:, -bh:, :, :], a[:, :, :bw, :], a[:, :, -bw:, :]):
        fs.append(p.mean(axis=(1, 2)))
        fs.append(p.std(axis=(1, 2)))
    fs.append(a.mean(axis=(1, 2)))
    fs.append(a.std(axis=(1, 2)))
    mx = a.max(axis=3)
    mn = a.min(axis=3)
    sat = (mx - mn) / (mx + 1e-5)
    fs.append(sat.mean(axis=(1, 2))[:, None])
    fs.append(sat.std(axis=(1, 2))[:, None])
    f = np.concatenate(fs, axis=1)
    return (f - f.mean(0, keepdims=True)) / (f.std(0, keepdims=True) + 1e-6)


def make_groups(arr, n_clusters):
    f = style_feats(arr)
    try:
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=n_clusters, n_init=6, random_state=SEED).fit_predict(f).astype(np.int64)
    except Exception:
        rng = np.random.RandomState(SEED)
        c = f[rng.choice(len(f), n_clusters, replace=False)]
        lab = np.zeros(len(f), dtype=np.int64)
        for _ in range(12):
            lab = ((f[:, None, :] - c[None, :, :]) ** 2).sum(2).argmin(1)
            for k in range(n_clusters):
                m = lab == k
                if m.sum() > 0:
                    c[k] = f[m].mean(0)
        return lab


def make_folds(y, g, n_splits):
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        folds = np.zeros(len(y), dtype=np.int64)
        for i, (_, va) in enumerate(sgk.split(np.zeros(len(y)), y, g)):
            folds[va] = i
        return folds
    except Exception:
        ug = np.unique(g)
        rng = np.random.RandomState(SEED)
        rng.shuffle(ug)
        m = {gg: i % n_splits for i, gg in enumerate(ug)}
        return np.array([m[gg] for gg in g], dtype=np.int64)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_g = torch.tensor([1.0, 2.0, 1.0])
GK = (_g[:, None] * _g[None, :])
GK = (GK / GK.sum()).view(1, 1, 3, 3).repeat(3, 1, 1, 1)


def geom_aug(x, oh, ow):
    n = x.shape[0]
    d = x.device
    sx = torch.empty(n, device=d).uniform_(SCALE_LO, SCALE_HI)
    sy = torch.empty(n, device=d).uniform_(SCALE_LO, SCALE_HI)
    ang = torch.empty(n, device=d).uniform_(-0.14, 0.14)
    fx = torch.where(torch.rand(n, device=d) < 0.5, -1.0, 1.0)
    cx = (torch.rand(n, device=d) * 2 - 1) * (1 - sx) * TRANS_F
    cy = (torch.rand(n, device=d) * 2 - 1) * (1 - sy) * TRANS_F
    ca = torch.cos(ang)
    sa = torch.sin(ang)
    th = torch.zeros(n, 2, 3, device=d)
    th[:, 0, 0] = sx * ca * fx
    th[:, 0, 1] = -sy * sa
    th[:, 0, 2] = cx
    th[:, 1, 0] = sx * sa * fx
    th[:, 1, 1] = sy * ca
    th[:, 1, 2] = cy
    grid = F.affine_grid(th, (n, 3, oh, ow), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def zoom_view(x, sc):
    n = x.shape[0]
    th = torch.zeros(n, 2, 3, device=x.device)
    th[:, 0, 0] = sc
    th[:, 1, 1] = sc
    grid = F.affine_grid(th, (n, 3, IMG_H, IMG_W), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def color_aug(x):
    n = x.shape[0]
    d = x.device
    x = x * torch.exp(torch.empty(n, 3, 1, 1, device=d).uniform_(-0.28, 0.28))
    x = x + torch.empty(n, 1, 1, 1, device=d).uniform_(-0.12, 0.12)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = m + (x - m) * torch.empty(n, 1, 1, 1, device=d).uniform_(0.72, 1.35)
    lum = (x * torch.tensor([0.299, 0.587, 0.114], device=d).view(1, 3, 1, 1)).sum(1, keepdim=True)
    x = lum + (x - lum) * torch.empty(n, 1, 1, 1, device=d).uniform_(0.55, 1.35)
    gm = (torch.rand(n, 1, 1, 1, device=d) < 0.12).float()
    x = x * (1 - gm) + lum.repeat(1, 3, 1, 1) * gm
    bm = (torch.rand(n, 1, 1, 1, device=d) < 0.18).float()
    xb = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), GK.to(d), groups=3)
    x = x * (1 - bm) + xb * bm
    x = x + torch.randn_like(x) * torch.empty(n, 1, 1, 1, device=d).uniform_(0.0, 0.035)
    return x.clamp(0.0, 1.0)


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class Net(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.fc = nn.Linear(512, len(CLASSES))
        self.cell = nn.Sequential(
            nn.Conv2d(768, 160, 1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 4, 1),
        )
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        za = self.fc(F.adaptive_avg_pool2d(f4, 1).flatten(1))
        u4 = F.interpolate(f4, size=f3.shape[2:], mode="nearest")
        q = torch.softmax(self.cell(torch.cat([f3, u4], 1)), 1).mean(dim=(2, 3))
        t = q[:, :3].sum(1, keepdim=True)
        s = (q[:, :3] + 1e-3) / (t + 3e-3)
        b, fo, e = s[:, 0], s[:, 1], s[:, 2]
        room = 0.6 - e
        z_b = torch.minimum(b - 0.5, room)
        z_f = torch.minimum(fo - 0.5, room)
        z_m = torch.minimum(torch.minimum(0.5 - b, 0.5 - fo), room)
        z_s = e - 0.6
        zc = torch.stack([z_b, z_f, z_m, z_s], 1) * self.scale.clamp(1.0, 60.0)
        return za, zc


def build_model():
    import torchvision

    src = "scratch"
    try:
        from torchvision.models import ResNet18_Weights

        bb = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        src = "torchvision-imagenet"
    except Exception:
        try:
            bb = torchvision.models.resnet18(pretrained=True)
            src = "torchvision-imagenet-legacy"
        except Exception:
            bb = torchvision.models.resnet18(weights=None)
            src = "scratch"
    m = Net(bb)
    for blk in (m.stem, m.layer1):
        for p in blk.parameters():
            p.requires_grad = False
    return m, src


def predict(model, arr, bs=48):
    model.eval()
    A = []
    C = []
    with torch.no_grad():
        for i in range(0, len(arr), bs):
            x0 = torch.from_numpy(arr[i:i + bs]).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            pa = None
            pc = None
            k = 0
            for sc in TTA_SC:
                xs = x0 if sc >= 0.999 else zoom_view(x0, sc)
                for fl in (False, True):
                    xi = torch.flip(xs, dims=[3]) if fl else xs
                    za, zc = model(norm(xi))
                    sa = torch.softmax(za, 1)
                    scp = torch.softmax(zc, 1)
                    pa = sa if pa is None else pa + sa
                    pc = scp if pc is None else pc + scp
                    k += 1
            A.append((pa / k).float().cpu().numpy())
            C.append((pc / k).float().cpu().numpy())
    return np.concatenate(A, 0), np.concatenate(C, 0)


def tune(pa, pc, y):
    ws = np.arange(0.0, 1.001, 0.1)
    vs = np.arange(-1.2, 1.201, 0.08)

    def fit(idx):
        ya = y[idx]
        best = (0.5, np.zeros(4), -1.0)
        for w in ws:
            lp = np.log(w * pa[idx] + (1.0 - w) * pc[idx] + 1e-8)
            off = np.zeros(4)
            s = cwm_f1(ya, lp.argmax(1))
            for _ in range(3):
                for c in range(4):
                    for v in vs:
                        t = off.copy()
                        t[c] = v
                        s2 = cwm_f1(ya, (lp + t).argmax(1))
                        if s2 > s + 1e-9:
                            s = s2
                            off = t
            if s > best[2]:
                best = (w, off, s)
        return best[0], best[1]

    def sc(idx, w, off):
        return cwm_f1(y[idx], (np.log(w * pa[idx] + (1.0 - w) * pc[idx] + 1e-8) + off).argmax(1))

    allidx = np.arange(len(y))
    base = sc(allidx, 0.5, np.zeros(4))
    if len(y) < 60:
        return 0.5, np.zeros(4), base, base, 0.0
    chunks = np.array_split(np.random.RandomState(SEED).permutation(allidx), 4)
    gains = []
    for k in range(4):
        va = chunks[k]
        trn = np.concatenate([chunks[j] for j in range(4) if j != k])
        if len(va) < 8:
            continue
        w, off = fit(trn)
        gains.append(sc(va, w, off) - sc(va, 0.5, np.zeros(4)))
    mg = float(np.mean(gains)) if gains else 0.0
    if mg <= 0.002:
        return 0.5, np.zeros(4), base, base, mg
    w, off = fit(allidx)
    return w, off, base, sc(allidx, w, off), mg


def main():
    if len(sys.argv) >= 3:
        public_dir = Path(sys.argv[1])
        sub_out = Path(sys.argv[2])
    else:
        public_dir = Path(".")
        sub_out = Path("working/submission.csv")
    sub_out.parent.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(public_dir / "train.csv")
    te = pd.read_csv(public_dir / "test.csv")
    cls2i = {c: i for i, c in enumerate(CLASSES)}
    y = tr["target"].map(cls2i).values.astype(np.int64)

    pd.DataFrame({"id": te["id"].values, "target": CLASSES[0]}).to_csv(sub_out, index=False)
    print("device", DEV, "cpu", NCPU, "train", len(tr), "test", len(te), flush=True)

    Xtr = load_images(tr["image_path"].tolist(), public_dir)
    Xte = load_images(te["image_path"].tolist(), public_dir)
    print("loaded", round(elapsed(), 1), flush=True)

    g = make_groups(Xtr, min(14, max(4, len(tr) // 70)))
    folds = make_folds(y, g, N_SPLITS)
    print("groups", len(np.unique(g)), "folds", np.bincount(folds, minlength=N_SPLITS), flush=True)

    cnt = np.bincount(y, minlength=4).astype(np.float64)
    samp_w = (1.0 / cnt)[y]
    samp_w = samp_w / samp_w.sum()

    oof_a = np.zeros((len(tr), 4))
    oof_c = np.zeros((len(tr), 4))
    oof_n = np.zeros(len(tr))
    te_a = np.zeros((len(te), 4))
    te_c = np.zeros((len(te), 4))
    te_n = 0.0
    fold_costs = []

    for f in range(N_SPLITS):
        now = elapsed()
        if now > HARD - RESERVE - 120:
            print("stop: budget", flush=True)
            break
        if fold_costs and now + 0.75 * float(np.mean(fold_costs)) > HARD - RESERVE:
            print("stop: no room", flush=True)
            break
        va_idx = np.where(folds == f)[0]
        tr_idx = np.where(folds != f)[0]
        if len(va_idx) < 5 or len(tr_idx) < 20:
            continue
        f_start = time.time()
        remaining = HARD - RESERVE - now
        left_folds = PLAN_FOLDS if not fold_costs else max(1, int(remaining / max(np.mean(fold_costs), 1.0)))
        share = 0.74 * remaining / max(1, left_folds)

        model, src = build_model()
        model.to(DEV)
        if f == 0:
            print("weights:", src, flush=True)
        hid = set()
        for mod in (model.fc, model.cell):
            for p in mod.parameters():
                hid.add(id(p))
        hid.add(id(model.scale))
        gh = [p for p in model.parameters() if p.requires_grad and id(p) in hid]
        gb = [p for p in model.parameters() if p.requires_grad and id(p) not in hid]
        opt = torch.optim.AdamW([{"params": gb, "lr": LR}, {"params": gh, "lr": LR * 4.0}], lr=LR, weight_decay=WD)
        base_lrs = [LR, LR * 4.0]
        crit = nn.CrossEntropyLoss(label_smoothing=SMOOTH)

        w = samp_w[tr_idx]
        w = w / w.sum()
        steps = max(8, int(math.ceil(len(tr_idx) / BATCH)))
        rng = np.random.RandomState(SEED + f)

        n_ep = MAX_EPOCHS
        ep = 0
        sn_oa = []
        sn_oc = []
        sn_ta = []
        sn_tc = []
        while ep < n_ep:
            big = ep >= int(BIG_FRAC * n_ep)
            oh = IMG_H if big else SML_H
            ow = IMG_W if big else SML_W
            t0 = time.time()
            model.train()
            tot = 0.0
            for _ in range(steps):
                bi = rng.choice(tr_idx, size=BATCH, replace=True, p=w)
                xb = torch.from_numpy(Xtr[bi]).to(DEV).permute(0, 3, 1, 2).float() / 255.0
                yb = torch.from_numpy(y[bi]).to(DEV)
                xb = color_aug(geom_aug(xb, oh, ow))
                za, zc = model(norm(xb))
                loss = crit(za, yb) + crit(zc, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.detach())
            et = time.time() - t0
            if ep == 0:
                n_ep = int(min(MAX_EPOCHS, max(MIN_EPOCHS, int(share / max(1.27 * et, 1e-3)))))
                print("fold", f, "et", round(et, 1), "share", round(share), "n_ep", n_ep, flush=True)
            prog = min(1.0, (ep + 1) / max(1, n_ep - 1))
            cf = 0.5 * (1 + math.cos(math.pi * prog))
            for pg, bl in zip(opt.param_groups, base_lrs):
                pg["lr"] = max(bl * cf, bl * 0.02)
            if ep >= n_ep - N_SNAP:
                a, c = predict(model, Xtr[va_idx])
                sn_oa.append(a)
                sn_oc.append(c)
                a, c = predict(model, Xte)
                sn_ta.append(a)
                sn_tc.append(c)
            print("  f", f, "ep", ep, "sz", oh, "loss", round(tot / steps, 4), "t", round(elapsed()), flush=True)
            ep += 1
            if ep < n_ep - N_SNAP and elapsed() + et * 1.7 + 230 > HARD - RESERVE:
                n_ep = ep + N_SNAP

        if not sn_oa:
            a, c = predict(model, Xtr[va_idx])
            sn_oa.append(a)
            sn_oc.append(c)
            a, c = predict(model, Xte)
            sn_ta.append(a)
            sn_tc.append(c)
        oof_a[va_idx] += np.mean(np.stack(sn_oa, 0), 0)
        oof_c[va_idx] += np.mean(np.stack(sn_oc, 0), 0)
        oof_n[va_idx] += 1
        te_a += np.mean(np.stack(sn_ta, 0), 0)
        te_c += np.mean(np.stack(sn_tc, 0), 0)
        te_n += 1
        fold_costs.append(time.time() - f_start)
        po = 0.5 * np.mean(np.stack(sn_oa, 0), 0) + 0.5 * np.mean(np.stack(sn_oc, 0), 0)
        print("FOLD", f, "cwm", round(cwm_f1(y[va_idx], po.argmax(1)), 4), "cost", round(fold_costs[-1]), "t", round(elapsed()), flush=True)

        cur = 0.5 * (te_a / te_n) + 0.5 * (te_c / te_n)
        pd.DataFrame({"id": te["id"].values, "target": [CLASSES[i] for i in cur.argmax(1)]}).to_csv(sub_out, index=False)
        del model
        if DEV.type == "cuda":
            torch.cuda.empty_cache()

    have = oof_n > 0
    if have.sum() > 0 and te_n > 0:
        pa = np.nan_to_num(oof_a[have] / oof_n[have][:, None], nan=0.25)
        pc = np.nan_to_num(oof_c[have] / oof_n[have][:, None], nan=0.25)
        wgt, off, base, tuned, mg = tune(pa, pc, y[have])
        print("OOF base", round(base, 4), "tuned", round(tuned, 4), "w", round(wgt, 2), "gain", round(mg, 4), "off", np.round(off, 3), flush=True)
        cm = np.zeros((4, 4), dtype=int)
        pr = (np.log(wgt * pa + (1.0 - wgt) * pc + 1e-8) + off).argmax(1)
        for aa, bb in zip(y[have], pr):
            cm[aa, bb] += 1
        print("confusion\n", cm, flush=True)
        ta = np.nan_to_num(te_a / te_n, nan=0.25)
        tc = np.nan_to_num(te_c / te_n, nan=0.25)
        fin = (np.log(wgt * ta + (1.0 - wgt) * tc + 1e-8) + off).argmax(1)
    else:
        fin = np.zeros(len(te), dtype=np.int64)

    pd.DataFrame({"id": te["id"].values, "target": [CLASSES[i] for i in fin]}).to_csv(sub_out, index=False)
    print("wrote", sub_out, "folds", int(te_n), "elapsed", round(elapsed()), flush=True)


main()

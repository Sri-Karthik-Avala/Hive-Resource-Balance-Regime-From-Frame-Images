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
TOTAL_BUDGET = 5000.0
TRAIN_BUDGET = 4150.0
MAX_EPOCHS = 16
MIN_EPOCHS = 3
N_FOLDS = 5
BATCH = 32
LR = 3e-4
WD = 1e-4
SMOOTH = 0.05
IMG_H = 224
IMG_W = 320
N_SNAP = 2
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
    ba = float(np.mean(recs))
    return 0.55 * of1 + 0.25 * cf1 + 0.20 * ba


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
    parts = [a[:, :bh, :, :], a[:, -bh:, :, :], a[:, :, :bw, :], a[:, :, -bw:, :]]
    fs = []
    for p in parts:
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
    f = (f - f.mean(0, keepdims=True)) / (f.std(0, keepdims=True) + 1e-6)
    return f


def make_groups(arr, n_clusters):
    f = style_feats(arr)
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, n_init=6, random_state=SEED)
        return km.fit_predict(f).astype(np.int64)
    except Exception:
        rng = np.random.RandomState(SEED)
        c = f[rng.choice(len(f), n_clusters, replace=False)]
        lab = np.zeros(len(f), dtype=np.int64)
        for _ in range(12):
            d = ((f[:, None, :] - c[None, :, :]) ** 2).sum(2)
            lab = d.argmin(1)
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
GK = torch.tensor([1.0, 2.0, 1.0])
GK = (GK[:, None] * GK[None, :])
GK = (GK / GK.sum()).view(1, 1, 3, 3).repeat(3, 1, 1, 1)


def geom_aug(x):
    n = x.shape[0]
    d = x.device
    sx = torch.empty(n, device=d).uniform_(0.68, 1.0)
    sy = torch.empty(n, device=d).uniform_(0.68, 1.0)
    ang = torch.empty(n, device=d).uniform_(-0.14, 0.14)
    fx = torch.where(torch.rand(n, device=d) < 0.5, -1.0, 1.0)
    cx = (torch.rand(n, device=d) * 2 - 1) * (1 - sx)
    cy = (torch.rand(n, device=d) * 2 - 1) * (1 - sy)
    ca = torch.cos(ang)
    sa = torch.sin(ang)
    th = torch.zeros(n, 2, 3, device=d)
    th[:, 0, 0] = sx * ca * fx
    th[:, 0, 1] = -sy * sa
    th[:, 0, 2] = cx
    th[:, 1, 0] = sx * sa * fx
    th[:, 1, 1] = sy * ca
    th[:, 1, 2] = cy
    grid = F.affine_grid(th, (n, 3, IMG_H, IMG_W), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def color_aug(x):
    n = x.shape[0]
    d = x.device
    g = torch.exp(torch.empty(n, 3, 1, 1, device=d).uniform_(-0.28, 0.28))
    x = x * g
    x = x + torch.empty(n, 1, 1, 1, device=d).uniform_(-0.12, 0.12)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    c = torch.empty(n, 1, 1, 1, device=d).uniform_(0.72, 1.35)
    x = m + (x - m) * c
    lum = (x * torch.tensor([0.299, 0.587, 0.114], device=d).view(1, 3, 1, 1)).sum(1, keepdim=True)
    sfac = torch.empty(n, 1, 1, 1, device=d).uniform_(0.55, 1.35)
    x = lum + (x - lum) * sfac
    gm = (torch.rand(n, 1, 1, 1, device=d) < 0.12).float()
    x = x * (1 - gm) + lum.repeat(1, 3, 1, 1) * gm
    bm = (torch.rand(n, 1, 1, 1, device=d) < 0.18).float()
    xb = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), GK.to(d), groups=3)
    x = x * (1 - bm) + xb * bm
    x = x + torch.randn_like(x) * torch.empty(n, 1, 1, 1, device=d).uniform_(0.0, 0.035)
    return x.clamp(0.0, 1.0)


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def build_model():
    import torchvision

    src = "scratch"
    try:
        from torchvision.models import ResNet18_Weights

        m = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        src = "torchvision-imagenet"
    except Exception:
        try:
            m = torchvision.models.resnet18(pretrained=True)
            src = "torchvision-imagenet-legacy"
        except Exception:
            m = torchvision.models.resnet18(weights=None)
            src = "scratch"
    m.fc = nn.Linear(512, len(CLASSES))
    for p in m.conv1.parameters():
        p.requires_grad = False
    for p in m.bn1.parameters():
        p.requires_grad = False
    for p in m.layer1.parameters():
        p.requires_grad = False
    return m, src


def predict(model, arr, bs=64):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(arr), bs):
            xb = torch.from_numpy(arr[i:i + bs]).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            p = 0.0
            for fl in (False, True):
                xi = torch.flip(xb, dims=[3]) if fl else xb
                p = p + torch.softmax(model(norm(xi)), 1)
            outs.append((p / 2.0).float().cpu().numpy())
    return np.concatenate(outs, 0)


def tune_offsets(prob, y):
    base = cwm_f1(y, prob.argmax(1))
    lp = np.log(prob + 1e-8)

    def fit(idx):
        off = np.zeros(4)
        best = cwm_f1(y[idx], (lp[idx] + off).argmax(1))
        for _ in range(4):
            for c in range(4):
                for v in np.arange(-1.2, 1.21, 0.06):
                    t = off.copy()
                    t[c] = v
                    s = cwm_f1(y[idx], (lp[idx] + t).argmax(1))
                    if s > best + 1e-9:
                        best = s
                        off = t
        return off

    allidx = np.arange(len(y))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(allidx)
    chunks = np.array_split(perm, 4)
    gain = []
    for k in range(4):
        va = chunks[k]
        tr = np.concatenate([chunks[j] for j in range(4) if j != k])
        o = fit(tr)
        gain.append(cwm_f1(y[va], (lp[va] + o).argmax(1)) - cwm_f1(y[va], lp[va].argmax(1)))
    mg = float(np.mean(gain))
    if mg <= 0.002:
        return np.zeros(4), base, base, mg
    off = fit(allidx)
    return off, base, cwm_f1(y, (lp + off).argmax(1)), mg


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
    print("loaded images", elapsed(), flush=True)

    g = make_groups(Xtr, min(14, max(4, len(tr) // 70)))
    folds = make_folds(y, g, N_FOLDS)
    print("groups", len(np.unique(g)), "fold sizes", np.bincount(folds), flush=True)

    cnt = np.bincount(y, minlength=4).astype(np.float64)
    samp_w = (1.0 / cnt)[y]
    samp_w = samp_w / samp_w.sum()

    oof = np.zeros((len(tr), 4), dtype=np.float64)
    oof_n = np.zeros(len(tr), dtype=np.float64)
    test_p = np.zeros((len(te), 4), dtype=np.float64)
    test_n = 0.0

    train_end = START + TRAIN_BUDGET
    done_folds = 0
    for f in range(N_FOLDS):
        if elapsed() > TRAIN_BUDGET - 120:
            print("stop: budget", flush=True)
            break
        fold_deadline = START + (TRAIN_BUDGET) * (f + 1) / N_FOLDS
        va_idx = np.where(folds == f)[0]
        tr_idx = np.where(folds != f)[0]
        if len(va_idx) == 0 or len(tr_idx) == 0:
            continue

        model, src = build_model()
        model.to(DEV)
        if f == 0:
            print("weights:", src, flush=True)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
        crit = nn.CrossEntropyLoss(label_smoothing=SMOOTH)

        w = samp_w[tr_idx]
        w = w / w.sum()
        steps = max(8, int(math.ceil(len(tr_idx) / BATCH)))
        rng = np.random.RandomState(SEED + f)

        n_ep = MAX_EPOCHS
        ep = 0
        snaps_oof = []
        snaps_te = []
        while ep < n_ep:
            t0 = time.time()
            model.train()
            tot = 0.0
            for _ in range(steps):
                bi = rng.choice(tr_idx, size=BATCH, replace=True, p=w)
                xb = torch.from_numpy(Xtr[bi]).to(DEV).permute(0, 3, 1, 2).float() / 255.0
                yb = torch.from_numpy(y[bi]).to(DEV)
                xb = geom_aug(xb)
                xb = color_aug(xb)
                out = model(norm(xb))
                loss = crit(out, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.detach())
            et = time.time() - t0
            if ep == 0:
                left = fold_deadline - time.time()
                est = int(left / max(et, 1e-3))
                n_ep = int(min(MAX_EPOCHS, max(MIN_EPOCHS, est + 1)))
                print("fold", f, "epoch_time", round(et, 1), "n_epochs", n_ep, flush=True)
            prog = min(1.0, (ep + 1) / max(1, n_ep - 1))
            lr = LR * 0.5 * (1 + math.cos(math.pi * prog))
            for pg in opt.param_groups:
                pg["lr"] = max(lr, LR * 0.02)
            if ep >= n_ep - N_SNAP:
                snaps_oof.append(predict(model, Xtr[va_idx]))
                snaps_te.append(predict(model, Xte))
            print("  f", f, "ep", ep, "loss", round(tot / steps, 4), "t", round(elapsed()), flush=True)
            ep += 1
            if time.time() + et * 1.1 > fold_deadline and ep < n_ep - N_SNAP:
                n_ep = ep + N_SNAP

        if not snaps_oof:
            snaps_oof.append(predict(model, Xtr[va_idx]))
            snaps_te.append(predict(model, Xte))
        po = np.mean(np.stack(snaps_oof, 0), 0)
        pt = np.mean(np.stack(snaps_te, 0), 0)
        oof[va_idx] += po
        oof_n[va_idx] += 1
        test_p += pt
        test_n += 1
        done_folds += 1
        fs = cwm_f1(y[va_idx], po.argmax(1))
        print("FOLD", f, "cwm", round(fs, 4), "elapsed", round(elapsed()), flush=True)

        cur = test_p / max(test_n, 1.0)
        pd.DataFrame({"id": te["id"].values, "target": [CLASSES[i] for i in cur.argmax(1)]}).to_csv(sub_out, index=False)
        del model
        if DEV.type == "cuda":
            torch.cuda.empty_cache()

    have = oof_n > 0
    if have.sum() > 0 and test_n > 0:
        oo = oof.copy()
        oo[have] = oo[have] / oof_n[have][:, None]
        off, base, tuned, mg = tune_offsets(oo[have], y[have])
        print("OOF plain", round(base, 4), "tuned", round(tuned, 4), "cvgain", round(mg, 4), "off", np.round(off, 3), flush=True)
        cm = np.zeros((4, 4), dtype=int)
        pr = (np.log(oo[have] + 1e-8) + off).argmax(1)
        for a, b in zip(y[have], pr):
            cm[a, b] += 1
        print("confusion\n", cm, flush=True)
        cur = test_p / test_n
        fin = (np.log(cur + 1e-8) + off).argmax(1)
    else:
        fin = np.zeros(len(te), dtype=np.int64)

    pd.DataFrame({"id": te["id"].values, "target": [CLASSES[i] for i in fin]}).to_csv(sub_out, index=False)
    print("wrote", sub_out, "folds", done_folds, "elapsed", round(elapsed()), flush=True)


main()

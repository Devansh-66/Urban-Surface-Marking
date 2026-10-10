import os, sys, json, math, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F

def parse_paths():
    if len(sys.argv) > 2 and not sys.argv[1].startswith('-f'):
        pub, out = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        candidates = ['/kaggle/input/datasets/devansh66/urban-surface-marking',
                      '/kaggle/input/urban-surface-marking',
                      '/kaggle/input', './dataset/public', '.']
        pub = next((Path(c) for c in candidates if (Path(c) / 'train.csv').exists()), Path('./dataset/public'))
        out = Path('./working/submission.csv')
    os.makedirs(out.parent, exist_ok=True)
    return pub, out

PUB, OUT = parse_paths()
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = DEV == 'cuda'
torch.backends.cudnn.benchmark = True
R, C = 9, 12
IH, IW = 288, 192
CH, CW = IH//R, IW//C
SEEDS = [42, 1337, 2024, 7]
FOLDS, EPOCHS, BS = 5, 50, 40
NCNT, NAREA, NHIST, NVB = 31, 9, 10, 10
NCX, NCY, NW, NH = 8, 8, 7, 7
SM = ('small', 'medium', 'large')
CELLB = [6, 12, 20, 30, 45]
S = 4
RF, CF = R*S, C*S
LSE_R = 4.0
DICE_W = 0.5
EDGE_W = 0.35
TTA_SHIFTS = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))
try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:
    _lsa = None

def tok(r, c): return "r%02d_c%02d" % (r, c)

try:
    from scipy.ndimage import label as _sclabel
    def cc(g):
        lab, k = _sclabel(g)
        return lab-1, k
except Exception:
    def cc(g):
        lab = -np.ones(g.shape, int); k = 0
        H_, W_ = g.shape
        for i in range(H_):
            for j in range(W_):
                if g[i, j] and lab[i, j] < 0:
                    st = [(i, j)]; lab[i, j] = k
                    while st:
                        a, b = st.pop()
                        for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                            x, y = a+da, b+db
                            if 0 <= x < H_ and 0 <= y < W_ and g[x, y] and lab[x, y] < 0:
                                lab[x, y] = k; st.append((x, y))
                    k += 1
        return lab, k

def f1(p, t):
    P, T = set(p), set(t)
    if not P and not T: return 1.0
    if not P or not T: return 0.0
    i = len(P & T)
    return 0.0 if i == 0 else 2*i/(len(P)+len(T))

def dupf1(p, t):
    P = Counter(tuple(sorted(d.items())) for d in p); T = Counter(tuple(sorted(d.items())) for d in t)
    if not P and not T: return 1.0
    if not P or not T: return 0.0
    i = sum((P & T).values())
    return 0.0 if i == 0 else 2*i/(sum(P.values())+sum(T.values()))

def parts(p, t):
    return dict(
        count=max(0, 1-abs(p['marking_count']-t['marking_count'])/8),
        occ=f1(p['occupied_cells'], t['occupied_cells']),
        edge=f1(p['edge_touch_cells'], t['edge_touch_cells']),
        hist=sum(max(0, 1-abs(p['size_histogram'][k]-t['size_histogram'][k])/4) for k in SM)/3,
        vert=sum(max(0, 1-abs(a-b)/5) for a, b in zip(p['vertical_band_counts'], t['vertical_band_counts']))/9,
        area=max(0, 1-abs(p['area_bin']-t['area_bin'])/4),
        card=dupf1(p['region_cards'], t['region_cards']))

W = dict(count=.14, occ=.27, edge=.12, hist=.12, vert=.12, area=.08, card=.15)
def row_score(p, t):
    q = parts(p, t); return sum(W[k]*q[k] for k in W)

def lut_fit(pairs, maxk):
    d = defaultdict(Counter)
    for k, v in pairs: d[k][v] += 1
    lut = {k: d[k].most_common(1)[0][0] for k in d}
    ks = sorted(lut)
    return [lut[k] if k in lut else (lut[min(ks, key=lambda x: abs(x-k))] if ks else 0)
            for k in range(maxk+1)]

def lut_get(l, k):
    return int(l[min(max(int(k), 0), len(l)-1)])

def load_imgs(df):
    X = np.zeros((len(df), IH, IW, 3), np.uint8)
    for i, p in enumerate(df.image_path.values):
        fp = PUB / p
        if not fp.exists(): fp = PUB / 'images' / Path(p).name
        X[i] = np.asarray(Image.open(fp).convert('RGB').resize((IW, IH), Image.BILINEAR))
    return X

train = pd.read_csv(PUB / 'train.csv'); test = pd.read_csv(PUB / 'test.csv')
A = [json.loads(x) for x in train.answer_json]
N = len(A)
Y = np.zeros((N, R, C), np.float32); YE = np.zeros((N, R, C), np.float32)
Tcnt = np.zeros(N, np.int64); Tarea = np.zeros(N, np.int64)
Thist = np.zeros((N, 3), np.int64); Tvb = np.zeros((N, R), np.int64)
Tcard = np.zeros((N, 4), np.int64); Mcard = np.zeros(N, np.float32)
The = np.zeros(N, np.int64)
for i, a in enumerate(A):
    for c in a['occupied_cells']: Y[i, int(c[1:3]), int(c[5:7])] = 1
    for c in a['edge_touch_cells']: YE[i, int(c[1:3]), int(c[5:7])] = 1
    The[i] = 1 if a['edge_touch_cells'] else 0
    Tcnt[i] = min(a['marking_count'], NCNT-1); Tarea[i] = a['area_bin']
    Thist[i] = [min(a['size_histogram'][k], NHIST-1) for k in SM]
    Tvb[i] = [min(v, NVB-1) for v in a['vertical_band_counts']]
    if a['marking_count'] == 1 and len(a['region_cards']) == 1:
        d = a['region_cards'][0]; Mcard[i] = 1
        Tcard[i] = [min(d['cx_bin'], NCX-1), min(d['cy_bin'], NCY-1),
                    min(d['w_bin'], NW-1), min(d['h_bin'], NH-1)]
Xtr = load_imgs(train); Xte = load_imgs(test)
print('images', Xtr.shape, Xte.shape, flush=True)

def block(ci, co, s=2):
    return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

HEADS = [('cnt', NCNT), ('area', NAREA), ('h0', NHIST), ('h1', NHIST), ('h2', NHIST),
         ('cx', NCX), ('cy', NCY), ('w', NW), ('h', NH), ('he', 2)] + [('v%d' % i, NVB) for i in range(R)]

def lse_pool(fine):
    B, K = fine.shape[0], fine.shape[1]
    b = fine.reshape(B, K, R, S, C, S).permute(0, 1, 2, 4, 3, 5).reshape(B, K, R, C, S*S)
    return (torch.logsumexp(LSE_R*b, -1) - math.log(S*S)) / LSE_R

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = block(3, 32); self.b2 = block(32, 64)
        self.b3 = block(64, 128); self.b4 = block(128, 192)
        self.ctx = nn.Sequential(nn.Conv2d(192, 192, 3, 1, 2, dilation=2, bias=False),
                                 nn.BatchNorm2d(192), nn.ReLU(inplace=True), nn.Dropout2d(0.2))
        self.lat = nn.Conv2d(64, 64, 1)
        self.fuse = nn.Sequential(nn.Conv2d(192+64, 128, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fine = nn.Conv2d(128, 2, 1)
        self.trunk = nn.Sequential(nn.Linear(192*2, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.hs = nn.ModuleDict({k: nn.Linear(256, n) for k, n in HEADS})

    def forward(self, x):
        x2 = self.b2(self.b1(x))
        f = self.ctx(self.b4(self.b3(x2)))
        lo = F.adaptive_avg_pool2d(self.lat(x2), (RF, CF))
        hi = F.interpolate(f, size=(RF, CF), mode='bilinear', align_corners=False)
        fu = self.fuse(torch.cat([hi, lo], 1))                       # B,128,RF,CF
        fine = self.fine(fu)                                         # B,2,RF,CF
        o = lse_pool(fine)                                           # B,2,R,C
        d = f.detach()
        g = torch.cat([F.adaptive_avg_pool2d(d, 1).flatten(1), F.adaptive_max_pool2d(d, 1).flatten(1)], 1)
        g = self.trunk(g)
        return o[:, 0], o[:, 1], fine[:, 0], {k: self.hs[k](g) for k, _ in HEADS}

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
def norm(u8):
    return (u8.permute(0, 3, 1, 2).float()/255. - MEAN) / STD

FLIP_CX = torch.tensor([NCX-1-i for i in range(NCX)], device=DEV)

def predict(model, X, bs=32):
    """Returns (flip-only, flip+shift) variants; the decoder picks between them on OOF."""
    model.eval()
    n = len(X)
    acc = {k: [np.zeros((n,)+sh, np.float64) for _ in range(2)]
           for k, sh in (('o', (R, C)), ('e', (R, C)), ('f', (RF, CF)))}
    cnt = {k: [np.zeros((n,)+sh, np.float64) for _ in range(2)]
           for k, sh in (('o', (R, C)), ('e', (R, C)), ('f', (RF, CF)))}
    H = {k: [] for k, _ in HEADS}
    with torch.no_grad(), torch.autocast('cuda', enabled=AMP):
        for i in range(0, n, bs):
            xb0 = norm(torch.from_numpy(X[i:i+bs]).to(DEV)).contiguous(memory_format=torch.channels_last)
            for si, (dy, dx) in enumerate(TTA_SHIFTS):
                xs = xb0 if (dy == 0 and dx == 0) else torch.roll(xb0, (dy*CH, dx*CW), (2, 3))
                o1, e1, f1_, h1 = model(xs)
                o2, e2, f2_, h2 = model(torch.flip(xs, [3]))
                out = {'o': (torch.sigmoid(o1.float()) + torch.flip(torch.sigmoid(o2.float()), [2]))/2,
                       'e': (torch.sigmoid(e1.float()) + torch.flip(torch.sigmoid(e2.float()), [2]))/2,
                       'f': (torch.sigmoid(f1_.float()) + torch.flip(torch.sigmoid(f2_.float()), [2]))/2}
                for k, v in out.items():
                    sc = S if k == 'f' else 1
                    v = torch.roll(v, (-dy*sc, -dx*sc), (-2, -1))
                    m = torch.ones_like(v)
                    if dy > 0: m[:, :dy*sc, :] = 0
                    elif dy < 0: m[:, dy*sc:, :] = 0
                    if dx > 0: m[:, :, :dx*sc] = 0
                    elif dx < 0: m[:, :, dx*sc:] = 0
                    m = torch.roll(m, (-dy*sc, -dx*sc), (-2, -1))
                    vn = (v*m).cpu().numpy(); mn = m.cpu().numpy()
                    acc[k][1][i:i+bs] += vn; cnt[k][1][i:i+bs] += mn
                    if si == 0:
                        acc[k][0][i:i+bs] += vn; cnt[k][0][i:i+bs] += mn
                if si == 0:
                    for k, _ in HEADS:
                        p1 = F.softmax(h1[k].float(), 1); p2 = F.softmax(h2[k].float(), 1)
                        if k == 'cx': p2 = p2[:, FLIP_CX]
                        H[k].append(((p1+p2)/2).cpu().numpy())

    out = []
    for v in range(2):
        d = {k: (acc[k][v]/np.maximum(cnt[k][v], 1e-6)).astype(np.float32) for k in acc}
        out.append(d)
    hh = {k: np.concatenate(v) for k, v in H.items()}
    return out[0], out[1], hh

def wmed(p):
    return int(np.argmax(np.cumsum(p) >= 0.5))

idx = np.arange(N); rng = np.random.RandomState(SEED); rng.shuffle(idx)
folds = np.array_split(idx, FOLDS)
VAR = 2                                  # 0 = flip-only TTA, 1 = flip + cell shifts
oof_o = [np.zeros((N, R, C), np.float32) for _ in range(VAR)]
oof_e = [np.zeros((N, R, C), np.float32) for _ in range(VAR)]
oof_f = [np.zeros((N, RF, CF), np.float32) for _ in range(VAR)]
te_o = [np.zeros((len(test), R, C), np.float32) for _ in range(VAR)]
te_e = [np.zeros((len(test), R, C), np.float32) for _ in range(VAR)]
te_f = [np.zeros((len(test), RF, CF), np.float32) for _ in range(VAR)]
oof_h = {k: np.zeros((N, n), np.float32) for k, n in HEADS}
te_h = {k: np.zeros((len(test), n), np.float32) for k, n in HEADS}
pw = torch.tensor((1-Y.mean())/Y.mean()*0.6).to(DEV)
pwe = torch.tensor(min(30.0, (1-YE.mean())/max(YE.mean(), 1e-6)*0.6)).to(DEV)
HW = {'cnt': .3, 'area': .2, 'h0': .15, 'h1': .15, 'h2': .15,
      'cx': .2, 'cy': .2, 'w': .2, 'h': .2, 'he': .25}
for i in range(R): HW['v%d' % i] = .04

Tg = {'cnt': Tcnt, 'area': Tarea, 'h0': Thist[:, 0], 'h1': Thist[:, 1], 'h2': Thist[:, 2],
      'cx': Tcard[:, 0], 'cy': Tcard[:, 1], 'w': Tcard[:, 2], 'h': Tcard[:, 3], 'he': The}
for i in range(R): Tg['v%d' % i] = Tvb[:, i]

def shift_cells(xb, yb, ye, dy, dx):
    xb = torch.roll(xb, (dy*CH, dx*CW), (2, 3))
    yb = torch.roll(yb, (dy, dx), (1, 2)); ye = torch.roll(ye, (dy, dx), (1, 2))
    m = torch.ones_like(yb)
    if dy > 0: m[:, :dy, :] = 0
    elif dy < 0: m[:, dy:, :] = 0
    if dx > 0: m[:, :, :dx] = 0
    elif dx < 0: m[:, :, dx:] = 0
    return xb, yb, ye, m

def dice(logit, y, m=None):
    p = torch.sigmoid(logit)
    if m is not None: p = p*m; y = y*m
    num = 2*(p*y).sum((1, 2)) + 1.0
    den = p.sum((1, 2)) + y.sum((1, 2)) + 1.0
    return 1 - (num/den).mean()

RUNS = len(SEEDS)*FOLDS
for seed in SEEDS:
  random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
  for f in range(FOLDS):
    va = folds[f]; tr = np.concatenate([folds[j] for j in range(FOLDS) if j != f])
    model = Net().to(DEV).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = EPOCHS * max(1, math.ceil(len(tr)/BS))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    Xf = torch.from_numpy(Xtr[tr]).to(DEV)
    Yf = torch.from_numpy(Y[tr]).to(DEV); Ef = torch.from_numpy(YE[tr]).to(DEV)
    Tf = {k: torch.from_numpy(np.asarray(v)[tr]).to(DEV) for k, v in Tg.items()}
    Mf = torch.from_numpy(Mcard[tr]).to(DEV)
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(tr), device=DEV)
        for i in range(0, len(tr), BS):
            b = perm[i:i+BS]
            if b.numel() < 2: continue
            xb = norm(Xf[b]); yb = Yf[b].clone(); eb = Ef[b].clone(); flipped = False
            if random.random() < 0.5:
                xb = torch.flip(xb, [3]); yb = torch.flip(yb, [2]); eb = torch.flip(eb, [2]); flipped = True
            wm = None
            if random.random() < 0.6:
                dy, dx = random.randint(-1, 1), random.randint(-2, 2)
                if dy or dx: xb, yb, eb, wm = shift_cells(xb, yb, eb, dy, dx)
            if random.random() < 0.8:
                r = lambda: torch.rand(xb.size(0), 1, 1, 1, device=DEV)
                xb = xb*(0.8 + 0.4*r()) + 0.3*(r()-0.5)
            if random.random() < 0.4:
                xb = xb*(0.9 + 0.2*torch.rand(xb.size(0), 3, 1, 1, device=DEV))
            if random.random() < 0.3:
                xb = xb + 0.05*torch.randn_like(xb)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', enabled=AMP):
                o, oe, _fine, hd = model(xb)
                if wm is None:
                    occ_loss = F.binary_cross_entropy_with_logits(o, yb, pos_weight=pw) + DICE_W*dice(o, yb)
                else:
                    e = F.binary_cross_entropy_with_logits(o, yb, pos_weight=pw, reduction='none')
                    occ_loss = (e*wm).sum()/wm.sum().clamp(min=1) + DICE_W*dice(o, yb, wm)
                hloss = torch.zeros((), device=DEV)
                if wm is None:
                    # a cell shift moves the image boundary -> edge labels invalid
                    hloss = hloss + EDGE_W*(F.binary_cross_entropy_with_logits(oe, eb, pos_weight=pwe)
                                            + DICE_W*dice(oe, eb))
                    for k, n in HEADS:
                        t = Tf[k][b]
                        if k == 'cx' and flipped: t = (NCX-1) - t
                        ce = F.cross_entropy(hd[k], t, reduction='none', label_smoothing=0.05)
                        if k in ('cx', 'cy', 'w', 'h'):
                            m = Mf[b]; ce = (ce*m).sum()/m.sum().clamp(min=1)
                        else:
                            ce = ce.mean()
                        hloss = hloss + HW[k]*ce
                loss = occ_loss + hloss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    d0, d1, h = predict(model, Xtr[va])
    for v, d in enumerate((d0, d1)):
        oof_o[v][va] += d['o']/len(SEEDS); oof_e[v][va] += d['e']/len(SEEDS); oof_f[v][va] += d['f']/len(SEEDS)
    for k in h: oof_h[k][va] += h[k]/len(SEEDS)
    d0, d1, h = predict(model, Xte)
    for v, d in enumerate((d0, d1)):
        te_o[v] += d['o']/RUNS; te_e[v] += d['e']/RUNS; te_f[v] += d['f']/RUNS
    for k in h: te_h[k] += h[k]/RUNS
    print('seed %d fold %d  occ_loss %.4f' % (seed, f, occ_loss.item()), flush=True)

try:
    np.savez_compressed(OUT.parent/'oof.npz', o=oof_o[0], o1=oof_o[1], e=oof_e[0],
                        f=oof_f[0], f1=oof_f[1],
                        **{'h_'+k: v for k, v in oof_h.items()})
except Exception as _e:
    print('oof save skipped', _e)

# ---------------- decoding ----------------
def grid_th(prob, th, kmin=3):
    g = prob >= np.float32(th)
    if g.sum() < kmin:
        g = np.zeros(R*C, bool)
        g[np.argpartition(-prob.ravel(), kmin-1)[:kmin]] = True
        g = g.reshape(R, C)
    return g

def occ_grid(prob, cfg, kmin=3):
    """A single global cut is wrong when confidence varies image to image: measured on
    OOF, cutting at a fraction of each image's own high quantile scores 0.6315 against
    0.5954 for the best global threshold. Falls back to the global rule if OOF says so."""
    if cfg.get('rel') is None:
        return grid_th(prob, cfg['th'], kmin)
    c, q = cfg['rel']
    ref = prob.max() if q >= 1 else np.quantile(prob, q)
    g = prob >= np.float32(c*ref)
    if g.sum() < kmin:
        g = np.zeros(R*C, bool)
        g[np.argpartition(-prob.ravel(), kmin-1)[:kmin]] = True
        g = g.reshape(R, C)
    return g

def edge_set(pe, occ_cells, qhas, cfg):
    """80% of rows have no edge cells and F1(empty, empty)=1, so an over-eager guess is
    expensive. Gate on the has_edge head first, then cut the per-cell edge map restricted
    to cells already called occupied (edge_touch_cells must be a subset of occupied)."""
    if not cfg['edge'] or not occ_cells or qhas < cfg['qcut']: return []
    p = np.array([pe[int(c[1:3]), int(c[5:7])] for c in occ_cells], np.float64)
    k = np.where(p >= cfg['ecut'])[0]
    if not len(k): k = [int(np.argmax(p))]
    return [occ_cells[i] for i in k]

class Cal:
    pass
cal = Cal()

def comp_info(g, fp, cfg):
    lab, k = cc(g)
    big = np.repeat(np.repeat(g, S, 0), S, 1)
    fm = (fp >= cfg['fth']) & big if fp is not None else None
    out = []
    for t in range(k):
        rs, cs = np.where(lab == t)
        d = dict(size=int(rs.size), r0=int(rs.min()), r1=int(rs.max()),
                 c0=int(cs.min()), c1=int(cs.max()), box=None, mass=0)
        if fp is not None:
            cm = np.repeat(np.repeat(lab == t, S, 0), S, 1)
            m = fm & cm
            if not m.any(): m = cm
            if cfg['fmode'] == 'largest':
                fl, fk = cc(m)
                if fk > 1:
                    sz = np.bincount(fl[fl >= 0].ravel(), minlength=fk)
                    m = fl == int(np.argmax(sz))
            rs2, cs2 = np.where(m)
            d['box'] = (int(rs2.min()), int(rs2.max()), int(cs2.min()), int(cs2.max()))
            d['mass'] = int(m.sum())
        out.append(d)
    return out, (int(fm.sum()) if fm is not None else 0)

def build(prob, pe, fp, h, i, cfg):
    g = occ_grid(prob[i], cfg)
    occ = [tok(r, c) for r, c in zip(*np.where(g))]
    comps, tmass = comp_info(g, fp[i] if fp is not None else None, cfg)
    ncomp = len(comps)
    hist = {'small': 0, 'medium': 0, 'large': 0}; vb = [0]*R
    for d in comps:
        if cfg['szfine'] and d['box'] is not None:
            hist[SM[lut_get(cal.szf, round(d['mass']**0.5))]] += 1
        else:
            hist[SM[lut_get(cal.sz, d['size'])]] += 1
        if cfg['vbfine'] and d['box'] is not None:
            vb[min(R-1, int((d['box'][0]+d['box'][1])/(2.0*S)))] += 1
        else:
            vb[int(round((d['r0']+d['r1'])/2))] += 1
    comps.sort(key=lambda d: -(d['mass'] if cfg['cardfine'] and d['box'] is not None else d['size']))
    cards = []
    for d in comps[:12]:
        if cfg['cardfine'] and d['box'] is not None:
            r0, r1, c0, c1 = d['box']
            cards.append({'cx_bin': lut_get(cal.fcx, c0+c1), 'cy_bin': lut_get(cal.fcy, r0+r1),
                          'w_bin': lut_get(cal.fw, c1-c0+1), 'h_bin': lut_get(cal.fh, r1-r0+1)})
        else:
            cards.append({'cx_bin': lut_get(cal.cx, d['c0']+d['c1']), 'cy_bin': lut_get(cal.cy, d['r0']+d['r1']),
                          'w_bin': lut_get(cal.wb, d['c1']-d['c0']+1), 'h_bin': lut_get(cal.hb, d['r1']-d['r0']+1)})
    cnt = ncomp
    if cal.cnt is not None:
        cnt = cal.cnt.get((min(ncomp, 20), int(np.digitize(len(occ), CELLB))), max(1, ncomp))
    ab = lut_get(cal.areaf, round(tmass**0.5)) if (cfg['afine'] and fp is not None) else lut_get(cal.area, len(occ))
    b = {'marking_count': int(min(max(cnt, 1), 30)), 'occupied_cells': occ,
         'edge_touch_cells': edge_set(pe[i], occ, float(h['he'][i][1]), cfg),
         'size_histogram': {k: int(min(v, 9)) for k, v in hist.items()},
         'vertical_band_counts': [int(min(v, 9)) for v in vb],
         'area_bin': int(ab), 'region_cards': cards}
    if cfg['hcnt']: b['marking_count'] = max(1, wmed(h['cnt'][i]))
    if cfg['harea']: b['area_bin'] = wmed(h['area'][i])
    if cfg['hhist']: b['size_histogram'] = {k: wmed(h['h%d' % j][i]) for j, k in enumerate(SM)}
    if cfg['hvb']: b['vertical_band_counts'] = [wmed(h['v%d' % j][i]) for j in range(R)]
    if len(cards) == 1:                      # per-bin choice between the LUT and the head
        for key, hk in (('cx_bin', 'cx'), ('cy_bin', 'cy'), ('w_bin', 'w'), ('h_bin', 'h')):
            if cfg['h_'+hk]: b['region_cards'][0][key] = int(h[hk][i].argmax())
    return b

CFG = dict(th=0.38, fth=0.5, fmode='largest', var=0,
           cardfine=False, szfine=False, vbfine=False, afine=False, edge=False,
           qcut=0.5, ecut=0.4, hcnt=False, harea=False, hhist=False, hvb=False,
           h_cx=False, h_cy=False, h_w=False, h_h=False,
           rel=None)

def ev(cfg, detail=False):
    v = cfg['var']
    B = [build(oof_o[v], oof_e[v], oof_f[v], oof_h, i, cfg) for i in range(N)]
    Q = [parts(b, a) for b, a in zip(B, A)]
    s = float(np.mean([sum(W[k]*q[k] for k in W) for q in Q]))
    if detail: return s, {k: float(np.mean([q[k] for q in Q])) for k in W}, B
    return s

def fit_luts(cfg):
    """All LUTs are geometric maps (grid/fine bbox -> bin) fitted on single-region rows,
    where the region <-> card correspondence is unambiguous, then applied to every region."""
    cal.cnt = None
    pa, paf, ps, psf = [], [], [], []
    px, py, pw_, ph = [], [], [], []
    fx, fy, fw, fh = [], [], [], []
    for n, a in enumerate(A):
        g = occ_grid(oof_o[cfg['var']][n], cfg)
        comps, tmass = comp_info(g, oof_f[cfg['var']][n], cfg)
        occ = int(g.sum())
        if not occ: continue
        pa.append((occ, a['area_bin'])); paf.append((int(round(tmass**0.5)), a['area_bin']))
        if a['marking_count'] == 1 and len(comps) == 1:
            k = [k for k, v in a['size_histogram'].items() if v == 1]
            if k:
                ps.append((comps[0]['size'], SM.index(k[0])))
                psf.append((int(round(comps[0]['mass']**0.5)), SM.index(k[0])))
            if len(a['region_cards']) == 1:
                d = comps[0]; cd = a['region_cards'][0]
                px.append((d['c0']+d['c1'], cd['cx_bin'])); py.append((d['r0']+d['r1'], cd['cy_bin']))
                pw_.append((d['c1']-d['c0']+1, cd['w_bin'])); ph.append((d['r1']-d['r0']+1, cd['h_bin']))
                r0, r1, c0, c1 = d['box']
                fx.append((c0+c1, cd['cx_bin'])); fy.append((r0+r1, cd['cy_bin']))
                fw.append((c1-c0+1, cd['w_bin'])); fh.append((r1-r0+1, cd['h_bin']))
    cal.area = lut_fit(pa, R*C); cal.areaf = lut_fit(paf, 64)
    cal.sz = lut_fit(ps, R*C); cal.szf = lut_fit(psf, 64)
    cal.cx = lut_fit(px, 2*C); cal.cy = lut_fit(py, 2*R)
    cal.wb = lut_fit(pw_, C); cal.hb = lut_fit(ph, R)
    cal.fcx = lut_fit(fx, 2*CF); cal.fcy = lut_fit(fy, 2*RF)
    cal.fw = lut_fit(fw, CF); cal.fh = lut_fit(fh, RF)

def fit_cnt(cfg):
    nc, cb = [], []
    for n in range(N):
        g = occ_grid(oof_o[cfg['var']][n], cfg)
        _l, k = cc(g)
        nc.append(k); cb.append(int(np.digitize(int(g.sum()), CELLB)))
    nc = np.array(nc); cb = np.array(cb); tc = np.array([a['marking_count'] for a in A])
    lut = {}
    for k in range(nc.max()+1):
        for j in range(len(CELLB)+1):
            m = (nc == k) & (cb == j)
            if m.sum() >= 8: lut[(min(k, 20), j)] = int(np.median(tc[m]))
    return lut

CFG['_cnt'] = False
fit_luts(CFG); cal.cnt = None
best = ev(CFG)
print('\nbaseline (th, derived)  OOF %.4f' % best, flush=True)

for th in np.arange(.20, .80, .02):
    c = dict(CFG); c['th'] = th
    fit_luts(c); s = ev(c)
    if s > best: best, CFG['th'] = s, float(th)
fit_luts(CFG); print('threshold swept  %.4f @ th=%.2f' % (best, CFG['th']), flush=True)

print('\n-- TTA variant --', flush=True)
bv, bvs = 0, best
for v in (0, 1):
    c = dict(CFG); c['var'] = v
    bb, bt = -1, CFG['th']
    for th in np.arange(.24, .60, .02):
        c['th'] = th; fit_luts(c); s_ = ev(c)
        if s_ > bb: bb, bt = s_, float(th)
    print('  var %d (%s)  %.4f @ th=%.2f' % (v, ('flip', 'flip+shift')[v], bb, bt), flush=True)
    if bb > bvs + 1e-5: bv, bvs, bvt = v, bb, bt
if bv:
    CFG['var'] = 1; CFG['th'] = bvt; best = bvs
    print('  shift-TTA ON  %.4f' % best, flush=True)
else:
    print('  flip-only TTA kept', flush=True)
fit_luts(CFG)

cal.cnt = fit_cnt(CFG); s = ev(CFG)
if s > best + 1e-5:
    print('count remap ON  %.4f (+%.4f)' % (s, s-best), flush=True); best = s; CFG['_cnt'] = True
else:
    print('count remap off %.4f' % s, flush=True); cal.cnt = None

def try_cfg(name, **kw):
    global best
    c = dict(CFG); c.update(kw)
    fit_luts(c)
    if CFG['_cnt']: cal.cnt = fit_cnt(c)
    s = ev(c)
    ok = s > best + 1e-5
    print('  %-24s %s %.4f (%+.4f)' % (name, 'ON ' if ok else 'off', s, s-best), flush=True)
    if ok:
        CFG.update(kw); best = s
    fit_luts(CFG)
    if CFG['_cnt']: cal.cnt = fit_cnt(CFG)
    return ok

print('\n-- per-image relative threshold --', flush=True)
bs, bk = best, None
for q in (0.99, 1.0):
    row = []
    for c_ in (0.30, 0.34, 0.38, 0.42, 0.46):
        cc_ = dict(CFG); cc_['rel'] = (c_, q)
        fit_luts(cc_)
        if CFG['_cnt']: cal.cnt = fit_cnt(cc_)
        s_ = ev(cc_); row.append('%.4f' % s_)
        if s_ > bs: bs, bk = s_, (c_, q)
    print('  q=%.2f  c[.30 .34 .38 .42 .46] = %s' % (q, ' '.join(row)), flush=True)
if bk is not None:
    CFG['rel'] = bk; best = bs
    print('  relative threshold ON  %.4f  c=%.2f q=%.2f' % (best, bk[0], bk[1]), flush=True)
else:
    print('  global threshold kept', flush=True)
fit_luts(CFG)
if CFG['_cnt']: cal.cnt = fit_cnt(CFG)

print('\n-- fine-map decoding --', flush=True)
bs, bk = best, None
for fth in (0.35, 0.5, 0.65):
    for fm in ('largest', 'union'):
        c = dict(CFG); c.update(fth=fth, fmode=fm, cardfine=True)
        fit_luts(c)
        if CFG['_cnt']: cal.cnt = fit_cnt(c)
        s = ev(c)
        print('  cards fth=%.2f %-8s %.4f' % (fth, fm, s), flush=True)
        if s > bs: bs, bk = s, (fth, fm)
if bk is not None:
    CFG.update(fth=bk[0], fmode=bk[1], cardfine=True); best = bs
    print('  fine cards ON  %.4f  fth=%.2f %s' % (best, bk[0], bk[1]), flush=True)
fit_luts(CFG)
if CFG['_cnt']: cal.cnt = fit_cnt(CFG)

try_cfg('area from fine mass', afine=True)
try_cfg('hist from fine mass', szfine=True)
try_cfg('vbands from fine box', vbfine=True)

print('\n-- edge_touch_cells (baseline = always empty) --', flush=True)
bs, bk = best, None
ECUTS = (0.20, 0.30, 0.40, 0.55)
for qc in (0.35, 0.45, 0.55, 0.65, 0.75):
    row = []
    for ec in ECUTS:
        c = dict(CFG); c.update(edge=True, qcut=qc, ecut=ec)
        fit_luts(c)
        if CFG['_cnt']: cal.cnt = fit_cnt(c)
        s = ev(c); row.append('%.4f' % s)
        if s > bs: bs, bk = s, (qc, ec)
    print('  qcut=%.2f  ecut%s = %s' % (qc, ECUTS, ' '.join(row)), flush=True)
if bk is not None:
    CFG.update(edge=True, qcut=bk[0], ecut=bk[1]); best = bs
    print('  edge ON  %.4f  qcut=%.2f ecut=%.2f' % (best, bk[0], bk[1]), flush=True)
else:
    print('  edge off (staying empty)', flush=True)
fit_luts(CFG)
if CFG['_cnt']: cal.cnt = fit_cnt(CFG)

print('\n-- global heads --', flush=True)
for k in ('hcnt', 'harea', 'hhist', 'hvb', 'h_cx', 'h_cy', 'h_w', 'h_h'):
    try_cfg(k, **{k: True})

fit_luts(CFG)
if CFG['_cnt']: cal.cnt = fit_cnt(CFG)

if CFG.get('rel') is None:
    for th in np.arange(.20, .80, .02):
        c = dict(CFG); c['th'] = th
        fit_luts(c)
        if CFG['_cnt']: cal.cnt = fit_cnt(c)
        s = ev(c)
        if s > best: best, CFG['th'] = s, float(th)
    fit_luts(CFG)
    if CFG['_cnt']: cal.cnt = fit_cnt(CFG)

s, det, B = ev(CFG, detail=True)
print('\nFINAL OOF %.4f' % s, flush=True)
print({k: v for k, v in CFG.items() if v not in (False,)}, flush=True)

print('\n%-6s %8s %8s %10s' % ('field', 'weight', 'score', 'pts'))
for k in ('occ', 'card', 'count', 'edge', 'hist', 'vert', 'area'):
    print('%-6s %8.2f %8.4f %10.4f' % (k, W[k], det[k], W[k]*det[k]))

def dist(name, fn, pred, truth):
    a = np.array([fn(p) for p in pred]); b = np.array([fn(p) for p in truth])
    print('%-9s pred mean %6.2f med %3.0f max %3.0f | true mean %6.2f med %3.0f max %3.0f'
          % (name, a.mean(), np.median(a), a.max(), b.mean(), np.median(b), b.max()))
print('\ncalibration (OOF vs train truth):')
dist('ncells', lambda p: len(p['occupied_cells']), B, A)
dist('nedge', lambda p: len(p['edge_touch_cells']), B, A)
dist('count', lambda p: p['marking_count'], B, A)
dist('area_bin', lambda p: p['area_bin'], B, A)
dist('ncards', lambda p: len(p['region_cards']), B, A)
print('empty grids (must be 0):', sum(1 for b in B if not b['occupied_cells']))

# te_* averages len(SEEDS)*FOLDS models, oof_* only len(SEEDS) -> te_o is smoother, so the
# OOF-tuned cut selects a different number of cells there. Match the OOF cell rate instead.
TCFG = dict(CFG)
V = CFG['var']
if CFG.get('rel') is not None:
    print('\nrelative threshold is per-image; no OOF->test recalibration needed', flush=True)
    print('cell rate: OOF %.4f | test %.4f'
          % (np.mean([occ_grid(oof_o[V][i], CFG).mean() for i in range(N)]),
             np.mean([occ_grid(te_o[V][i], CFG).mean() for i in range(len(test))])), flush=True)
else:
    oof_rate = np.mean([occ_grid(oof_o[V][i], CFG).mean() for i in range(N)])
    tth = float(np.quantile(te_o[V], 1-oof_rate))
    cur = np.mean([occ_grid(te_o[V][i], CFG).mean() for i in range(len(test))])
    print('\ncell-rate: OOF %.4f | test %.4f -> th %.3f' % (oof_rate, cur, tth), flush=True)
    if abs(cur - oof_rate) >= 0.005: TCFG['th'] = tth

rows = []
for i, rid in enumerate(test.id.values):
    b = build(te_o[V], te_e[V], te_f[V], te_h, i, TCFG)
    rows.append({'id': rid, 'answer_json': json.dumps(b, separators=(',', ':'))})
sub = pd.DataFrame(rows)
assert len(sub) == len(test) and sub.id.duplicated().sum() == 0
for x in sub.answer_json:
    d = json.loads(x)
    assert d['occupied_cells'], 'empty grid'
    assert set(d['edge_touch_cells']) <= set(d['occupied_cells']), 'edge not subset'
    assert len(d['vertical_band_counts']) == 9 and len(d['region_cards']) <= 12
sub.to_csv(OUT, index=False)
print('\nwrote', OUT, len(sub))

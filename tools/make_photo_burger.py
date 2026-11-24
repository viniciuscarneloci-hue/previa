"""
Gera o hambúrguer da demonstração em RA a partir de uma foto em ângulo (não vista de cima).

O hambúrguer é quase uma peça de revolução: o contorno da foto vira o perfil (lathe). A foto é
"desenrolada" numa textura cilíndrica (ângulo x altura) projetando cada ponto do perfil de volta na foto
com uma câmera ortográfica inclinada. A faixa da frente é repetida em volta com emenda suavizada,
então não há espelho nas laterais. As camadas ganham ondulação (alface, carne, queijo) para não
parecer um torno perfeito. Escala real em metros.
Uso: python3 tools/make_photo_burger.py <foto.jpg> models/smash-burger.glb
"""
import math, os, sys
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates, minimum_filter
sys.path.insert(0, os.path.dirname(__file__))
from make_photo_dishes import GLB16, Mesh, lathe  # noqa: E402

CX, CY0 = 490.0, 620.0          # centro do hambúrguer e linha da base na foto (px)
ELEV = math.radians(30)         # inclinação da câmera
K = 2767.0                      # px por metro (pão com 12 cm)
# contorno medido na foto: (linha y, meia largura em px)
PROFILE = [(308, 0), (311, 45), (316, 72), (320, 86), (330, 110), (340, 125), (350, 136), (360, 145), (370, 152),
           (390, 160), (410, 164), (432, 166), (442, 162), (448, 168), (452, 186), (466, 192), (484, 190),
           (496, 178), (510, 174), (530, 178), (550, 182), (575, 182), (595, 176), (604, 166), (612, 163), (620, 150)]
PLATE_R, PLATE_H = 0.099, 0.012
SEG, ROWS = 192, 150
TW, TH = 2048, 1024             # textura cilíndrica: largura = volta inteira
SQ = 0.72                       # usa a parte central da frente (as bordas da foto esticam)
DELTA = 0.35                    # largura da emenda suavizada (rad)

def dehaze(a, w=0.7):
    """Tira parte do vapor branco da foto (prior do canal escuro, leve)."""
    dc = minimum_filter(a.min(2), size=9)
    A = 0.92
    t = np.clip(1 - w * gaussian_filter(dc, 4) / A, 0.6, 1)[..., None]
    return np.clip((a - A) / t + A, 0, 1)

def bun_top(patch, S, size):
    """Topo do pão visto de cima: tons do pão da foto, dourado irregular e sementes de gergelim."""
    rng = np.random.default_rng(3)
    px = patch.reshape(-1, 3); lum = px.mean(1)
    bun = px[(lum > np.percentile(lum, 20)) & (lum < np.percentile(lum, 70))]
    c0, c1 = np.percentile(bun, 25, axis=0), np.percentile(bun, 80, axis=0)
    t = gaussian_filter(rng.random((S, S)), S * 0.04); t = (t - t.min()) / (t.max() - t.min())
    t2 = gaussian_filter(rng.random((S, S)), S * 0.008); t2 = (t2 - t2.min()) / (t2.max() - t2.min())
    img = c0 + (c1 - c0) * (0.35 * t + 0.65 * t2 * 0.6 + 0.2)[..., None]
    img = img * (1 + 0.04 * (rng.random((S, S, 1)) - 0.5))
    seed_c = np.array([0.93, 0.87, 0.74]); yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    mpp = size / S
    cen = S / 2; Rmax = 0.05 / mpp
    shade = np.zeros((S, S), np.float32); seedm = np.zeros((S, S), np.float32)
    for _ in range(170):
        while True:
            x, y = rng.uniform(-Rmax, Rmax, 2)
            if x * x + y * y < Rmax * Rmax: break
        x += cen; y += cen; ang = rng.uniform(0, math.pi)
        L, W = 0.0034 / mpp / 2, 0.0018 / mpp / 2
        x0, x1, y0, y1 = int(x - L - 4), int(x + L + 4), int(y - L - 4), int(y + L + 4)
        sx_, sy_ = xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y
        u = sx_ * math.cos(ang) + sy_ * math.sin(ang); v = -sx_ * math.sin(ang) + sy_ * math.cos(ang)
        d = (u / L) ** 2 + (v / W) ** 2
        seedm[y0:y1, x0:x1] = np.maximum(seedm[y0:y1, x0:x1], np.clip((1.15 - d) * 4, 0, 1))
        d2 = ((u - 1.5) / (L * 1.2)) ** 2 + ((v - 1.5) / (W * 1.4)) ** 2
        shade[y0:y1, x0:x1] = np.maximum(shade[y0:y1, x0:x1], np.clip((1.2 - d2) * 2, 0, 1))
    img = img * (1 - 0.18 * shade * (1 - seedm))[..., None]
    sv = seed_c * (0.92 + 0.08 * rng.random((S, S, 1)))
    img = img * (1 - seedm[..., None]) + sv * seedm[..., None]
    return np.clip(img, 0, 1).astype(np.float32)

def build(photo):
    im = Image.open(photo).convert('RGB')
    a = np.asarray(im, np.float32) / 255
    box = (int(CX - 230), 280, int(CX + 230), 710)
    sub = a[box[1]:box[3], box[0]:box[2]]
    sub = dehaze(sub)
    a[box[1]:box[3], box[0]:box[2]] = sub
    ce, se = math.cos(ELEV), math.sin(ELEV)

    # perfil em metros, de baixo para cima
    ys = np.array([p[0] for p in PROFILE], np.float32)[::-1]
    hw = np.array([p[1] for p in PROFILE], np.float32)[::-1]
    Y = (CY0 - ys) / (K * ce); R = hw / K
    Yf = np.linspace(Y[0], Y[-1], ROWS)
    Rf = gaussian_filter(np.interp(Yf, Y, R), 1.0); Rf[-1] = 0
    prof_r = np.concatenate([[0, Rf[0] * 0.85], Rf])
    prof_yp = np.concatenate([[Yf[0], Yf[0]], Yf])          # altura na foto
    prof_y = np.concatenate([[0, 0], Yf - Yf[0] + 0.006])    # altura no modelo
    N = len(prof_r)

    # textura cilíndrica desenrolada da foto
    P_ = math.pi - DELTA
    v = np.linspace(0, N - 1, TH)
    rr = np.interp(v, np.arange(N), prof_r)[:, None]
    yy = np.interp(v, np.arange(N), prof_yp)[:, None]
    phi = (np.arange(TW) + 0.5) / TW * 2 * math.pi
    psi = (phi / (2 * math.pi) * 2 * P_ + DELTA / 2) % P_   # período P_, frente (phi=pi/2) no centro da foto
    def sample(ps):
        q = math.pi / 2 + (ps - math.pi / 2) * SQ
        sx = CX + K * rr * 0.95 * np.cos(q)
        sy = CY0 - K * yy * ce + K * rr * 0.95 * np.sin(q) * se
        sy = np.broadcast_to(sy, (TH, TW)); sx = np.broadcast_to(sx, (TH, TW))
        return np.dstack([map_coordinates(a[..., c], [sy, sx], order=1, mode='nearest') for c in range(3)])
    A1 = sample(psi[None, :])
    wgt = np.clip(psi / DELTA, 0, 1)
    A2 = sample((psi + P_)[None, :])                         # continuação da borda direita
    w = (wgt * wgt * (3 - 2 * wgt))[None, :, None]
    tex = A1 * w + A2 * (1 - w)
    # topo do pão: textura plana gerada (cor do pão da foto + gergelim espalhado), sem repetição nem redemoinho no ápice
    cap_img = bun_top(a[322:372, 420:570], 1024, 0.13)
    Xc, Zc = rr * np.cos(phi)[None, :], rr * np.sin(phi)[None, :]
    tu = (Xc / 0.13 + 0.5) * 1023; tv = (Zc / 0.13 + 0.5) * 1023
    cap = np.dstack([map_coordinates(cap_img[..., c], [tv, tu], order=1, mode='nearest') for c in range(3)])
    rowsy = (CY0 - yy * K * ce)[:, 0]
    wc = np.clip((352 - rowsy) / 18, 0, 1); wc = (wc * wc * (3 - 2 * wc))[:, None, None]
    tex = tex * (1 - wc) + cap * wc
    timg = Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=1))
    # normal map dos detalhes (gergelim, crosta da carne, queijo)
    f = np.asarray(timg, np.float32).mean(2) / 255
    h = gaussian_filter(f, 0.8) - gaussian_filter(f, 6)
    gy, gx = np.gradient(gaussian_filter(h, 0.6))
    nn = np.dstack([-gx * 6, gy * 6, np.ones_like(h)]); nn /= np.linalg.norm(nn, axis=2, keepdims=True)
    nimg = Image.fromarray(((nn * 0.5 + 0.5) * 255).astype(np.uint8)).resize((1024, 512), Image.LANCZOS)

    # malha: lathe com ondulação por camada
    rng = np.random.default_rng(7)
    def noise(ph, k):
        out = 0
        for f_, amp in ((3, 1), (5, .7), (8, .5), (13, .35), (21, .2)):
            out = out + amp * np.sin(f_ * ph + rng.uniform(0, 6.3) + k)
        return out / 2.2
    base = PLATE_H - 0.002
    P, UV, I = [], [], []
    phs = 2 * math.pi * np.arange(SEG + 1) / SEG
    nz = {key: noise(phs, i) for i, key in enumerate(('alface', 'meio', 'carne', 'pao'))}
    for i in range(N):
        sy = CY0 - prof_yp[i] * K * ce                       # linha da foto desta camada
        amp = {'alface': 0.075, 'meio': 0.035, 'carne': 0.028, 'pao': 0.006}
        key = 'alface' if 446 <= sy <= 478 else 'meio' if 478 < sy <= 512 else 'carne' if 512 < sy <= 606 else 'pao'
        rw = prof_r[i] * (1 + amp[key] * nz[key]) if 0 < i < N - 1 else np.full(SEG + 1, prof_r[i])
        for j in range(SEG + 1):
            P.append([rw[j] * math.cos(phs[j]), base + prof_y[i], rw[j] * math.sin(phs[j])])
            UV.append([j / SEG, i / (N - 1)])
    n = SEG + 1
    for i in range(N - 1):
        for j in range(SEG):
            a0, b0, c0, d0 = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            I += [a0, c0, b0, b0, c0, d0]
    body = Mesh().add(P, UV, I)

    pc = np.array([0xE6, 0xE1, 0xD8]) / 255   # cerâmica clara da foto
    plate = lathe([(0, 0.0), (PLATE_R * 0.6, 0.0), (PLATE_R * 0.63, 0.003), (PLATE_R * 0.8, 0.004), (PLATE_R * 0.985, PLATE_H - 0.002),
                   (PLATE_R, PLATE_H + 0.002), (PLATE_R * 0.975, PLATE_H + 0.004), (PLATE_R * 0.88, base + 0.001), (0, base)], seg=SEG)
    g = GLB16()
    g.mesh(body, g.material('hamburguer', base_tex=g.texture(timg), normal_tex=g.texture(nimg), rough=0.5))
    g.mesh(plate, g.material('prato', color=[float(c) ** 2.2 for c in pc], rough=0.3))
    timg.save('/tmp/claude-0/pd/burger-unwrap.jpg')
    return g

if __name__ == '__main__':
    build(sys.argv[1]).save(sys.argv[2], 'smash-burger')

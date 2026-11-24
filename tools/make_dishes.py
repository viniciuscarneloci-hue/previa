"""
Gera os pratos de demonstração do Oak AR como .glb em escala real (metros).

Os modelos são procedurais (geometria + texturas geradas por ruído) e servem só para
a demonstração. Em produção, cada prato vem da fotogrametria feita no painel do dono.

Uso: python3 tools/make_dishes.py public/models
Depende só de numpy e Pillow.
"""
import io, json, math, struct, sys
import numpy as np
from PIL import Image, ImageFilter

RNG = np.random.default_rng(7)

# ---------------------------------------------------------------- texturas

def fbm(size, base=4, octaves=5, seed=0, persistence=0.5):
    """Ruído fractal suave em [0,1], tamanho size x size."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size), np.float32)
    amp, tot, cells = 1.0, 0.0, base
    for _ in range(octaves):
        g = rng.random((cells, cells)).astype(np.float32)
        img = Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        out += amp * (np.asarray(img, np.float32) / 255)
        tot += amp; amp *= persistence; cells *= 2
    out /= tot
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    return out

def mix(a, b, t):
    t = np.clip(t, 0, 1)[..., None]
    return a * (1 - t) + b * t

def col(hexs):
    hexs = hexs.lstrip('#')
    return np.array([int(hexs[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)

def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

def to_img(rgb):
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

def normal_map(h, strength=4.0):
    gy, gx = np.gradient(h.astype(np.float32))
    n = np.dstack([-gx * strength * h.shape[0] / 64, -gy * strength * h.shape[0] / 64, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    img = Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8))
    return img.resize((min(512, img.width),) * 2, Image.LANCZOS)

def polar_grid(size):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    x = x / (size - 1) * 2 - 1; y = y / (size - 1) * 2 - 1
    return x, y, np.sqrt(x * x + y * y)

# ---------------------------------------------------------------- geometria

class Mesh:
    def __init__(self):
        self.P, self.UV, self.I = [], [], []

    def add(self, P, UV, I):
        k = len(self.P)
        self.P += [list(map(float, p)) for p in P]
        self.UV += [list(map(float, u)) for u in UV]
        self.I += [k + i for i in I]
        return self

    def transformed(self, fn):
        m = Mesh(); m.P = [fn(np.array(p)) for p in self.P]; m.UV = list(self.UV); m.I = list(self.I); return m

    def normals(self):
        P = np.array(self.P, np.float64); I = np.array(self.I).reshape(-1, 3)
        fn = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
        # soma por posição (solda as costuras do torno sem fundir os UVs)
        key = {tuple(np.round(p, 6)): None for p in P}
        idx = {k: i for i, k in enumerate(key)}
        vk = np.array([idx[tuple(np.round(p, 6))] for p in P])
        acc = np.zeros((len(idx), 3))
        for c in range(3):
            np.add.at(acc, vk[I[:, c]], fn)
        N = acc[vk]
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
        return N

def lathe(profile, seg=64, u_rep=1.0, wobble=None):
    """Superfície de revolução. O perfil (r, y) vai do centro de baixo para fora e para cima:
    assim as normais apontam para fora."""
    prof = np.array(profile, np.float64)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
    L /= L[-1] if L[-1] > 0 else 1
    P, UV, I = [], [], []
    for i, (r, y) in enumerate(prof):
        for j in range(seg + 1):
            t = 2 * math.pi * j / seg
            rr, yy = r, y
            if wobble: rr, yy = wobble(r, y, t, L[i])
            P.append([rr * math.cos(t), yy, rr * math.sin(t)]); UV.append([u_rep * j / seg, L[i]])
    w = seg + 1
    for i in range(len(prof) - 1):
        for j in range(seg):
            a, b, c, d = i * w + j, (i + 1) * w + j, i * w + j + 1, (i + 1) * w + j + 1
            I += [a, b, c, b, d, c]
    return Mesh().add(P, UV, I)

def height_disc(R, h, rings=40, seg=96, uv_r=None):
    """Disco voltado para cima com altura h(x, z). UV planar (uv_r = raio que cobre a textura)."""
    uv_r = uv_r or R
    P, UV, I = [[0, h(0, 0), 0]], [[0.5, 0.5]], []
    for i in range(1, rings + 1):
        r = R * i / rings
        for j in range(seg):
            t = 2 * math.pi * j / seg
            x, z = r * math.cos(t), r * math.sin(t)
            P.append([x, h(x, z), z]); UV.append([0.5 + x / (2 * uv_r), 0.5 + z / (2 * uv_r)])
    for j in range(seg):
        I += [0, 1 + (j + 1) % seg, 1 + j]
    for i in range(1, rings):
        o0, o1 = 1 + (i - 1) * seg, 1 + i * seg
        for j in range(seg):
            j2 = (j + 1) % seg
            a, b, c, d = o0 + j, o1 + j, o0 + j2, o1 + j2
            I += [a, c, b, b, c, d]
    return Mesh().add(P, UV, I)

def grid_sheet(w, d, nx, nz, h):
    """Folha retangular (queijo, folha) com altura h(x, z)."""
    P, UV, I = [], [], []
    for i in range(nz + 1):
        for j in range(nx + 1):
            x = -w / 2 + w * j / nx; z = -d / 2 + d * i / nz
            P.append([x, h(x, z), z]); UV.append([j / nx, i / nz])
    for i in range(nz):
        for j in range(nx):
            a = i * (nx + 1) + j; b = a + 1; c = a + nx + 1; d2 = c + 1
            I += [a, c, b, b, c, d2]
    m = Mesh().add(P, UV, I)
    # lado de baixo (mesma malha, invertida e um pouco abaixo)
    k = len(P)
    m.P += [[p[0], p[1] - 0.0012, p[2]] for p in P]; m.UV += UV
    m.I += [k + i for tri in zip(I[0::3], I[2::3], I[1::3]) for i in tri]
    return m

def box(sx, sy, sz):
    x, y, z = sx / 2, sy / 2, sz / 2
    F = [([1, 0, 0], [[x, -y, -z], [x, y, -z], [x, y, z], [x, -y, z]]),
         ([-1, 0, 0], [[-x, -y, z], [-x, y, z], [-x, y, -z], [-x, -y, -z]]),
         ([0, 1, 0], [[-x, y, -z], [-x, y, z], [x, y, z], [x, y, -z]]),
         ([0, -1, 0], [[-x, -y, z], [-x, -y, -z], [x, -y, -z], [x, -y, z]]),
         ([0, 0, 1], [[x, -y, z], [x, y, z], [-x, y, z], [-x, -y, z]]),
         ([0, 0, -1], [[-x, -y, -z], [-x, y, -z], [x, y, -z], [x, -y, -z]])]
    m = Mesh()
    for n, q in F:
        m.add(q, [[0, 0], [0, 1], [1, 1], [1, 0]], [0, 1, 2, 0, 2, 3])
    return m

def rounded_box(sx, sy, sz, r=0.25, seg=24):
    """Cubo arredondado (superelipsoide) — bom para cubos de peixe, batatas, croutons."""
    prof = []
    for k in range(seg + 1):
        a = -math.pi / 2 + math.pi * k / seg
        prof.append((max(math.cos(a), 0) * 1.0, math.sin(a)))
    m = lathe(prof, seg=seg * 2)
    e = 2 / max(r, 0.05)
    def sq(p):
        x, y, z = p
        n = (abs(x) ** e + abs(y) ** e + abs(z) ** e) ** (1 / e) + 1e-9
        return [x / n * sx / 2, y / n * sy / 2, z / n * sz / 2]
    return m.transformed(lambda p: sq(p))

def ellipsoid(rx, ry, rz, seg=32, bump=None):
    prof = [(math.sin(math.pi * k / seg), -math.cos(math.pi * k / seg)) for k in range(seg + 1)]
    m = lathe(prof, seg=seg * 2)
    def f(p):
        x, y, z = p
        s = 1 + (bump(x, y, z) if bump else 0)
        return [x * rx * s, y * ry * s, z * rz * s]
    return m.transformed(f)

def place(m, pos=(0, 0, 0), rot_y=0.0, rot_x=0.0, rot_z=0.0, scale=1.0):
    cy, sy = math.cos(rot_y), math.sin(rot_y)
    cx, sx = math.cos(rot_x), math.sin(rot_x)
    cz, sz = math.cos(rot_z), math.sin(rot_z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Ry @ Rz @ Rx
    s = np.array(scale if isinstance(scale, (tuple, list)) else [scale] * 3)
    return m.transformed(lambda p: list(R @ (p * s) + np.array(pos)))

def merge(*ms):
    out = Mesh()
    for m in ms: out.add(m.P, m.UV, [i for i in m.I])
    return out

# ---------------------------------------------------------------- GLB

class GLB:
    def __init__(self):
        self.bin = bytearray(); self.views = []; self.acc = []; self.images = []; self.tex = []; self.mats = []; self.prims = []

    def _view(self, data, target=None):
        while len(self.bin) % 4: self.bin += b'\0'
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target: v["target"] = target
        self.bin += data; self.views.append(v); return len(self.views) - 1

    def _acc(self, arr, ctype, typ, target, minmax=False):
        a = {"bufferView": self._view(arr.tobytes(), target), "componentType": ctype, "count": len(arr), "type": typ}
        if minmax: a["min"] = arr.min(0).tolist(); a["max"] = arr.max(0).tolist()
        self.acc.append(a); return len(self.acc) - 1

    def texture(self, img, fmt='JPEG'):
        buf = io.BytesIO()
        if fmt == 'JPEG': img.convert('RGB').save(buf, 'JPEG', quality=86, optimize=True)
        else: img.save(buf, 'PNG', optimize=True)
        v = self._view(buf.getvalue())
        self.images.append({"bufferView": v, "mimeType": "image/jpeg" if fmt == 'JPEG' else "image/png"})
        self.tex.append({"source": len(self.images) - 1, "sampler": 0})
        return len(self.tex) - 1

    def material(self, name, color=None, base_tex=None, normal_tex=None, rough=0.7, metal=0.0, normal_scale=1.0, double=False):
        pbr = {"metallicFactor": metal, "roughnessFactor": rough}
        if color is not None: pbr["baseColorFactor"] = [float(c) for c in color] + [1.0]
        if base_tex is not None: pbr["baseColorTexture"] = {"index": base_tex}
        m = {"name": name, "pbrMetallicRoughness": pbr}
        if normal_tex is not None: m["normalTexture"] = {"index": normal_tex, "scale": normal_scale}
        if double: m["doubleSided"] = True
        self.mats.append(m); return len(self.mats) - 1

    def mesh(self, m, mat):
        P = np.array(m.P, np.float32); N = m.normals().astype(np.float32); UV = np.array(m.UV, np.float32)
        I = np.array(m.I, np.uint32)
        self.prims.append({"attributes": {
            "POSITION": self._acc(P, 5126, "VEC3", 34962, True),
            "NORMAL": self._acc(N, 5126, "VEC3", 34962),
            "TEXCOORD_0": self._acc(UV, 5126, "VEC2", 34962)},
            "indices": self._acc(I, 5125, "SCALAR", 34963), "material": mat})

    def save(self, path, name):
        gl = {"asset": {"version": "2.0", "generator": "oak-ar make_dishes.py"}, "scene": 0,
              "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
              "meshes": [{"name": name, "primitives": self.prims}], "materials": self.mats,
              "accessors": self.acc, "bufferViews": self.views, "buffers": [{"byteLength": len(self.bin)}],
              "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]}
        if self.images: gl["images"] = self.images; gl["textures"] = self.tex
        j = json.dumps(gl, separators=(',', ':')).encode()
        while len(j) % 4: j += b' '
        while len(self.bin) % 4: self.bin += b'\0'
        out = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(j) + 8 + len(self.bin))
        out += struct.pack('<II', len(j), 0x4E4F534A) + j + struct.pack('<II', len(self.bin), 0x004E4942) + bytes(self.bin)
        open(path, 'wb').write(out)
        tris = sum(self.acc[p["indices"]]["count"] for p in self.prims) // 3
        print(f"{path}: {len(out) / 1024:.0f} KB, {tris} triângulos")

# ---------------------------------------------------------------- peças comuns

def plate(g, R=0.135, rim=0.03, h=0.022, color='#F4F1EA', seed=1):
    """Prato de cerâmica com aba e pé."""
    prof = [(0, 0.004), (R - rim - 0.025, 0.004), (R - rim - 0.02, 0.0), (R - rim - 0.012, 0.0), (R - rim, 0.006),
            (R - 0.004, h - 0.002), (R, h), (R + 0.0005, h + 0.0015), (R - 0.004, h + 0.003),
            (R - rim + 0.002, 0.0095), (R - rim - 0.01, 0.0075), (0, 0.0075)]
    S = 512
    n = fbm(S, 8, 4, seed) * 0.04
    tex = to_img(np.dstack([np.full((S, S), c) for c in col(color)]) - n[..., None] * 0.5)
    mat = g.material("prato", base_tex=g.texture(tex), rough=0.18)
    g.mesh(lathe(prof, seg=96), mat)
    return 0.0075  # altura do centro do prato

def board(g, R=0.17, h=0.018, seed=3):
    """Tábua de madeira redonda."""
    S = 1024
    x, y, _ = polar_grid(S)
    warp = fbm(S, 3, 4, seed) * 2.5
    rings = np.sin((y * 18 + warp * 6 + fbm(S, 40, 2, seed + 1) * 0.6) * math.pi) * 0.5 + 0.5
    grain = fbm(S, 64, 3, seed + 2)
    t = rings * 0.55 + grain * 0.25 + fbm(S, 6, 4, seed + 3) * 0.2
    rgb = mix(col('#8A5A33')[None, None], col('#C8955E')[None, None], t)
    base = g.texture(to_img(rgb)); nrm = g.texture(normal_map(t, 1.2))
    mat = g.material("madeira", base_tex=base, normal_tex=nrm, rough=0.62)
    prof = [(0, 0), (R - 0.004, 0), (R, 0.003), (R, h - 0.003), (R - 0.004, h), (0, h)]
    m = lathe(prof, seg=96)
    # UV planar no topo/fundo para os veios ficarem retos
    m.UV = [[0.5 + p[0] / (2 * R), 0.5 + p[2] / (2 * R)] for p in m.P]
    g.mesh(m, mat)
    return h

# ---------------------------------------------------------------- pratos

def pizza(path):
    g = GLB(); top = board(g)
    R, Rin = 0.152, 0.128
    S = 1024
    x, y, r = polar_grid(S)
    # massa da borda: dourada com pintas queimadas (leopard)
    n1 = fbm(S, 6, 5, 11); n2 = fbm(S, 48, 3, 12)
    char = smooth(0.80, 0.9, fbm(S, 24, 3, 13) * 0.7 + n2 * 0.3)
    crust = mix(col('#E8B871')[None, None], col('#B9702F')[None, None], n1 * 0.9)
    crust = mix(crust, col('#3A2416')[None, None], char * 0.85)
    flour = smooth(0.72, 0.95, fbm(S, 30, 3, 14)) * 0.35
    crust = mix(crust, col('#F3E6D0')[None, None], flour)
    crust_h = n1 * 0.6 + n2 * 0.4
    mc = g.material("massa", base_tex=g.texture(to_img(crust)), normal_tex=g.texture(normal_map(crust_h, 2.5)), rough=0.82)
    def bump(r_, y_, t, L):
        k = 1 + 0.06 * math.sin(t * 7 + 1.3) + 0.04 * math.sin(t * 13 + 0.4)
        return r_ * (1 + 0.012 * math.sin(t * 9)), top + (y_ - top) * k
    prof = [(Rin - 0.004, top + 0.006), (Rin + 0.002, top + 0.014), (Rin + 0.008, top + 0.022), (R - 0.008, top + 0.024),
            (R - 0.002, top + 0.018), (R, top + 0.010), (R - 0.004, top + 0.002), (R - 0.012, top), (Rin - 0.004, top + 0.0005)]
    # o perfil acima vai de dentro para fora por cima: invertemos para manter normais para fora
    prof = prof[::-1]
    g.mesh(lathe(prof, seg=128, u_rep=4, wobble=bump), mc)
    # cobertura: molho + muçarela derretida + pontos tostados
    sauce = mix(col('#B8261A')[None, None], col('#E0552C')[None, None], fbm(S, 10, 4, 21))
    sauce = mix(sauce, col('#7A1A12')[None, None], smooth(0.65, 0.9, fbm(S, 40, 3, 22)) * 0.5)
    blobs = fbm(S, 5, 4, 23) * 0.65 + fbm(S, 18, 3, 24) * 0.35
    cheese_m = smooth(0.50, 0.56, blobs)
    cheese = mix(col('#FFF4D8')[None, None], col('#F2D9A0')[None, None], fbm(S, 30, 3, 25))
    brown = smooth(0.78, 0.92, fbm(S, 26, 3, 26)) * cheese_m
    cheese = mix(cheese, col('#B07A3A')[None, None], brown * 0.7)
    rgb = mix(sauce, cheese, cheese_m)
    oil = smooth(0.8, 0.95, fbm(S, 14, 3, 27)) * (1 - cheese_m) * 0.25
    rgb = mix(rgb, col('#E8862E')[None, None], oil)
    h_tex = cheese_m * 0.7 + fbm(S, 50, 2, 28) * 0.3
    mt = g.material("cobertura", base_tex=g.texture(to_img(rgb)), normal_tex=g.texture(normal_map(h_tex, 3)), rough=0.42)
    cm = cheese_m
    def hfun(px, pz):
        u = int((0.5 + px / (2 * Rin)) * (S - 1)); v = int((0.5 + pz / (2 * Rin)) * (S - 1))
        u = min(max(u, 0), S - 1); v = min(max(v, 0), S - 1)
        return top + 0.006 + cm[v, u] * 0.0035
    g.mesh(height_disc(Rin + 0.001, hfun, rings=60, seg=160, uv_r=Rin), mt)
    # manjericão
    S2 = 256
    lx, ly, _ = polar_grid(S2)
    vein = np.exp(-(lx * 12) ** 2) * 0.5
    leaf = mix(col('#2F6B1F')[None, None], col('#4E9A2E')[None, None], fbm(S2, 4, 3, 31) * 0.6 + vein)
    ml = g.material("manjericao", base_tex=g.texture(to_img(leaf)), rough=0.35, double=True)
    def leaf_mesh(L=0.034, W=0.02):
        def lh(x_, z_):
            return 0.004 * (1 - (x_ / (W / 2)) ** 2) * 0.6 + 0.003 * math.sin(z_ / L * math.pi)
        m = grid_sheet(W, L, 6, 10, lh)
        def shape(p):
            x_, y_, z_ = p
            t = (z_ / L) + 0.5
            w = math.sin(math.pi * min(max(t, 0), 1)) ** 0.8
            return [x_ * w, y_, z_]
        return m.transformed(shape)
    for k, (a, rr, rot) in enumerate([(0.3, 0.05, 0.5), (2.2, 0.07, 2.0), (4.1, 0.045, 3.6), (5.3, 0.085, 1.1), (1.2, 0.0, 0.2)]):
        g.mesh(place(leaf_mesh(), (rr * math.cos(a), top + 0.0115, rr * math.sin(a)), rot_y=rot, rot_x=0.08), ml)
    g.save(path, "pizza-margherita")

def burger(path):
    g = GLB(); top = plate(g, R=0.13)
    S = 1024
    # pão: dourado brilhante, gergelim
    n = fbm(S, 5, 5, 41)
    bun = mix(col('#C9772E')[None, None], col('#8E4A17')[None, None], n)
    bun_h = fbm(S, 40, 3, 42)
    mb = g.material("pao", base_tex=g.texture(to_img(bun)), normal_tex=g.texture(normal_map(bun_h, 1.2)), rough=0.38)
    crumb = mix(col('#F1D9A8')[None, None], col('#DDB878')[None, None], fbm(S, 30, 3, 43))
    mcr = g.material("miolo", base_tex=g.texture(to_img(crumb)), rough=0.85)
    y = top
    # pão de baixo: casca (lateral) e miolo (topo)
    Rb = 0.056
    g.mesh(lathe([(0, y), (Rb - 0.006, y), (Rb, y + 0.006), (Rb + 0.001, y + 0.014), (Rb - 0.002, y + 0.018)], seg=96), mb)
    g.mesh(height_disc(Rb - 0.002, lambda a, b: y + 0.018, rings=8, seg=96), mcr)
    y += 0.018
    # carne
    pm = fbm(S, 40, 4, 44)
    patty = mix(col('#3B2215')[None, None], col('#6B3E24')[None, None], pm)
    patty = mix(patty, col('#1E120B')[None, None], smooth(0.7, 0.9, fbm(S, 20, 3, 45)) * 0.7)
    mp = g.material("carne", base_tex=g.texture(to_img(patty)), normal_tex=g.texture(normal_map(pm, 4)), rough=0.55)
    Rp = 0.06
    def pw(r_, y_, t, L):
        return r_ * (1 + 0.02 * math.sin(t * 6) + 0.015 * math.sin(t * 15 + 1)), y_
    g.mesh(lathe([(0, y), (Rp - 0.005, y), (Rp, y + 0.005), (Rp + 0.001, y + 0.012), (Rp - 0.004, y + 0.018), (0, y + 0.019)], seg=96, u_rep=3, wobble=pw), mp)
    y += 0.019
    # queijo cheddar derretendo nas pontas
    cheese = mix(col('#F5A623')[None, None], col('#F7C04A')[None, None], fbm(256, 6, 3, 46))
    mq = g.material("queijo", base_tex=g.texture(to_img(cheese)), rough=0.3, double=True)
    def ch(x_, z_):
        d = max(abs(x_), abs(z_)); rr = math.hypot(x_, z_)
        droop = max(0, rr - 0.05) * 1.1 + max(0, d - 0.045) * 0.4
        return y + 0.001 - droop
    g.mesh(place(grid_sheet(0.105, 0.105, 24, 24, ch), rot_y=0.4), mq)
    y += 0.003
    # tomate
    tom = mix(col('#D2231D')[None, None], col('#F05A3C')[None, None], fbm(256, 6, 3, 47))
    mt = g.material("tomate", base_tex=g.texture(to_img(tom)), rough=0.2)
    g.mesh(lathe([(0, y), (0.035, y), (0.036, y + 0.006), (0, y + 0.006)], seg=48), mt)
    g.mesh(lathe([(0, y), (0.034, y), (0.035, y + 0.006), (0, y + 0.006)], seg=48).transformed(lambda p: [p[0] + 0.02, p[1] + 0.0005, p[2] - 0.012]), mt)
    y += 0.006
    # alface crespa
    lt = mix(col('#5FA83A')[None, None], col('#B9DB7A')[None, None], fbm(256, 8, 3, 48))
    ml = g.material("alface", base_tex=g.texture(to_img(lt)), rough=0.4, double=True)
    def lw(x_, z_):
        rr = math.hypot(x_, z_); t = math.atan2(z_, x_)
        return y + 0.002 + max(0, rr - 0.048) * (0.25 * math.sin(t * 9) + 0.12 * math.sin(t * 19 + 1)) - max(0, rr - 0.05) * 0.35
    g.mesh(height_disc(0.066, lw, rings=20, seg=120), ml)
    # o disco só tem a face de cima: duplicamos invertido para ver por baixo
    y += 0.004
    # pão de cima com gergelim
    Rt = 0.058; H = 0.045
    prof = [(0, y), (Rt - 0.004, y)]
    for k in range(1, 15):
        a = math.pi / 2 * k / 14
        prof.append((Rt * math.cos(a) * (1 - 0.04 * math.sin(a)), y + 0.006 + (H - 0.006) * math.sin(a) ** 0.9))
    prof[-1] = (0, y + H)
    g.mesh(lathe(prof, seg=96, u_rep=2), mb)
    seed_m = g.material("gergelim", color=col('#F6EBCB'), rough=0.5)
    seeds = []
    rng = np.random.default_rng(49)
    for _ in range(46):
        a = rng.random() * 2 * math.pi; e = math.acos(rng.uniform(0.25, 1.0))
        rr = Rt * math.sin(e); hh = y + 0.006 + (H - 0.006) * math.cos(e) ** 0.9
        rr = Rt * math.cos(math.pi / 2 - e)
        seeds.append(place(ellipsoid(0.0026, 0.0008, 0.0014, seg=6), (rr * math.cos(a), hh + 0.0004, rr * math.sin(a)), rot_y=rng.random() * 6.28, rot_x=-(math.pi / 2 - e) * math.sin(a) * 0.5, rot_z=(math.pi / 2 - e) * math.cos(a) * 0.5))
    g.mesh(merge(*seeds), seed_m)
    # batatas fritas ao lado
    fr = mix(col('#E9B040')[None, None], col('#F6D477')[None, None], fbm(128, 6, 3, 50))
    mf = g.material("batata", base_tex=g.texture(to_img(fr)), rough=0.55)
    fries = []
    for k in range(9):
        a = 0.2 + k * 0.11 + rng.uniform(-0.04, 0.04)
        fries.append(place(rounded_box(0.075, 0.009, 0.009, r=0.35, seg=8), (0.085 + rng.uniform(-0.01, 0.01), top + 0.006 + (k % 3) * 0.008, -0.015 + k * 0.006), rot_y=math.pi / 2 + rng.uniform(-0.5, 0.5), rot_z=rng.uniform(-0.15, 0.15)))
    g.mesh(merge(*fries), mf)
    g.save(path, "hamburguer-artesanal")

def poke(path):
    g = GLB()
    # tigela de cerâmica
    S = 512
    glaze = mix(col('#24303F')[None, None], col('#3B4D63')[None, None], fbm(S, 8, 4, 61))
    mb = g.material("tigela", base_tex=g.texture(to_img(glaze)), rough=0.25)
    Rt, H = 0.095, 0.07
    prof = [(0, 0.0), (0.04, 0.0), (0.043, 0.006), (0.05, 0.008)]
    for k in range(1, 13):
        a = math.pi / 2 * k / 12
        prof.append((0.05 + (Rt - 0.05) * math.sin(a), 0.008 + (H - 0.008) * (1 - math.cos(a))))
    prof += [(Rt + 0.002, H + 0.002), (Rt - 0.005, H + 0.003)]
    for k in range(12, -1, -1):
        a = math.pi / 2 * k / 12
        prof.append((0.045 + (Rt - 0.05) * math.sin(a), 0.014 + (H - 0.012) * (1 - math.cos(a))))
    prof.append((0, 0.014))
    g.mesh(lathe(prof, seg=96), mb)
    lvl = 0.055
    # arroz
    S = 1024
    grains = fbm(S, 120, 2, 62)
    rice = mix(col('#EFEADF')[None, None], col('#FFFFFF')[None, None], grains)
    rice = mix(rice, col('#D8CFBE')[None, None], smooth(0.7, 0.9, fbm(S, 60, 2, 63)) * 0.5)
    mr = g.material("arroz", base_tex=g.texture(to_img(rice)), normal_tex=g.texture(normal_map(grains, 6)), rough=0.6)
    g.mesh(height_disc(0.088, lambda x_, z_: lvl + 0.006 * (1 - (x_ * x_ + z_ * z_) / 0.0078) + 0.0015 * math.sin(x_ * 200) * math.sin(z_ * 180), rings=30, seg=96), mr)
    rng = np.random.default_rng(64)
    def surf(x_, z_): return lvl + 0.006 * (1 - (x_ * x_ + z_ * z_) / 0.0078)
    # salmão em cubos (laranja com listras de gordura)
    S2 = 256
    _, yy, _ = polar_grid(S2)
    stripes = smooth(0.75, 0.95, np.abs(np.sin((yy * 5 + fbm(S2, 4, 3, 65) * 0.8) * math.pi)))
    sal = mix(col('#F26A3A')[None, None], col('#FF9A6B')[None, None], fbm(S2, 8, 3, 66))
    sal = mix(sal, col('#FFE1CF')[None, None], stripes * 0.8)
    ms = g.material("salmao", base_tex=g.texture(to_img(sal)), rough=0.25)
    cubes = []
    for k in range(9):
        a = -0.9 + rng.uniform(-0.35, 0.35) + (k % 3) * 0.35; rr = 0.02 + (k // 3) * 0.018
        px, pz = rr * math.cos(a), rr * math.sin(a)
        cb_ = rounded_box(0.019, 0.016, 0.018, r=0.3, seg=10)
        cb_.UV = [[0.5 + p[0] * 20 + p[2] * 8, 0.5 + p[1] * 25] for p in cb_.P]
        cubes.append(place(cb_, (px, surf(px, pz) + 0.006, pz), rot_y=rng.random() * 3, rot_x=rng.uniform(-0.3, 0.3)))
    g.mesh(merge(*cubes), ms)
    # abacate fatiado em leque
    av = np.zeros((S2, S2, 3), np.float32)
    xx, yy2, _ = polar_grid(S2)
    t = (yy2 + 1) / 2
    av = mix(col('#D7E27A')[None, None], col('#7FAE35')[None, None], smooth(0.55, 0.95, t))
    av = mix(av, col('#2F5A1A')[None, None], smooth(0.93, 1.0, t))
    ma = g.material("abacate", base_tex=g.texture(to_img(av)), rough=0.35)
    slices = []
    for k in range(6):
        a = 1.4 + k * 0.17
        m = rounded_box(0.05, 0.0045, 0.014, r=0.4, seg=8)
        m = m.transformed(lambda p: [p[0], p[1] + 30 * (p[0]) ** 2 * 0.25, p[2]])
        px, pz = 0.045 * math.cos(a), 0.045 * math.sin(a)
        slices.append(place(m, (px, surf(px, pz) + 0.004 + k * 0.0012, pz), rot_y=-a + math.pi / 2, rot_x=0.25))
    g.mesh(merge(*slices), ma)
    # edamame
    me = g.material("edamame", color=col('#6DB33F'), rough=0.4)
    beans = []
    for k in range(16):
        a = 3.3 + rng.uniform(-0.4, 0.4); rr = rng.uniform(0.02, 0.07)
        px, pz = rr * math.cos(a), rr * math.sin(a)
        beans.append(place(ellipsoid(0.0055, 0.0042, 0.0045, seg=10), (px, surf(px, pz) + 0.004, pz), rot_y=rng.random() * 6))
    g.mesh(merge(*beans), me)
    # pepino em rodelas
    cu = np.ones((S2, S2, 3), np.float32) * col('#DDEBB5')[None, None]
    _, _, rr2 = polar_grid(S2)
    cu = mix(cu, col('#2E6B2A')[None, None], smooth(0.85, 0.92, rr2))
    cu = mix(cu, col('#C7DB8E')[None, None], np.exp(-((rr2 - 0.45) * 8) ** 2) * 0.5)
    mcu = g.material("pepino", base_tex=g.texture(to_img(cu)), rough=0.3)
    cuc = []
    for k in range(6):
        a = 5.0 + rng.uniform(-0.3, 0.3) + (k % 2) * 0.25; rr = 0.025 + (k // 2) * 0.017
        px, pz = rr * math.cos(a), rr * math.sin(a)
        d = height_disc(0.0125, lambda x_, z_: 0.0, rings=2, seg=24)
        cuc.append(place(d, (px, surf(px, pz) + 0.004, pz), rot_x=rng.uniform(-0.3, 0.3), rot_z=rng.uniform(-0.3, 0.3)))
    g.mesh(merge(*cuc), mcu)
    # gergelim e cebolinha
    msd = g.material("gergelim", color=col('#F3E9CE'), rough=0.5)
    sd = []
    for k in range(70):
        a = rng.random() * 6.28; rr = math.sqrt(rng.random()) * 0.075
        px, pz = rr * math.cos(a), rr * math.sin(a)
        sd.append(place(ellipsoid(0.0018, 0.0006, 0.001, seg=4), (px, surf(px, pz) + 0.0125, pz), rot_y=rng.random() * 6))
    g.mesh(merge(*sd), msd)
    mcb = g.material("cebolinha", color=col('#3E9B3A'), rough=0.4)
    cb = []
    for k in range(18):
        a = rng.random() * 6.28; rr = math.sqrt(rng.random()) * 0.06
        px, pz = rr * math.cos(a), rr * math.sin(a)
        cb.append(place(lathe([(0, 0), (0.0022, 0), (0.0022, 0.002), (0, 0.002)], seg=10), (px, surf(px, pz) + 0.0135, pz), rot_x=rng.uniform(-0.5, 0.5)))
    g.mesh(merge(*cb), mcb)
    g.save(path, "poke-de-salmao")

def dessert(path):
    g = GLB(); top = plate(g, R=0.12, color='#FBFAF7', seed=71)
    S = 1024
    # calda de chocolate no prato
    x, y, r = polar_grid(S)
    swirl = np.abs(np.sin((r * 9 + np.arctan2(y, x) * 0.6) * math.pi)) < 0.12
    swirl = swirl & (r > 0.45) & (r < 0.85) & (np.arctan2(y, x) > -0.6) & (np.arctan2(y, x) < 1.8)
    sauce_mask = np.asarray(Image.fromarray((swirl * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
    dots = np.zeros((S, S), np.float32)
    for k in range(5):
        a = -2.4 + k * 0.32; cx, cy = 0.62 * math.cos(a), 0.62 * math.sin(a)
        dots = np.maximum(dots, smooth(0.035 - k * 0.004, 0.025 - k * 0.004, np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))
    sm = np.maximum(smooth(0.3, 0.6, sauce_mask), dots)
    rgb = mix(np.ones((S, S, 3), np.float32) * col('#FBFAF7')[None, None], col('#3A1C10')[None, None], sm)
    alpha_like = sm
    mc = g.material("calda", base_tex=g.texture(to_img(rgb)), rough=0.15)
    g.mesh(height_disc(0.088, lambda a, b: top + 0.0004, rings=10, seg=96), mc)
    # petit gâteau
    cake_n = fbm(S, 30, 4, 72)
    cake = mix(col('#3A1D12')[None, None], col('#5A2E1C')[None, None], cake_n)
    dust = smooth(0.55, 0.8, fbm(S, 60, 3, 73)) * smooth(0.6, 0.2, r)
    cake_top = mix(cake, col('#F5EFE6')[None, None], dust * 0.85)
    mk = g.material("bolo", base_tex=g.texture(to_img(cake)), normal_tex=g.texture(normal_map(cake_n, 3)), rough=0.7)
    mkt = g.material("bolo_topo", base_tex=g.texture(to_img(cake_top)), normal_tex=g.texture(normal_map(cake_n, 3)), rough=0.8)
    cx = -0.02
    R = 0.037; H = 0.042
    side = lathe([(0, top), (R - 0.003, top), (R, top + 0.003), (R * 0.97, top + H - 0.004), (R * 0.93, top + H)], seg=96, u_rep=2)
    g.mesh(place(side, (cx, 0, 0)), mk)
    g.mesh(place(height_disc(R * 0.93, lambda a, b: top + H + 0.004 * (1 - (a * a + b * b) / (R * R)), rings=12, seg=96), (cx, 0, 0)), mkt)
    # sorvete de creme
    ice_n = fbm(512, 10, 4, 74)
    ice = mix(col('#FFF8E6')[None, None], col('#F6E7C1')[None, None], ice_n)
    beans_ = smooth(0.92, 0.97, fbm(512, 120, 2, 75))
    ice = mix(ice, col('#2B1B10')[None, None], beans_ * 0.7)
    mi = g.material("sorvete", base_tex=g.texture(to_img(ice)), normal_tex=g.texture(normal_map(ice_n, 2)), rough=0.5)
    def ib(x_, y_, z_):
        t = math.atan2(z_, x_)
        return 0.06 * math.sin(t * 5 + y_ * 6) * max(0, 0.3 - y_) + 0.03 * math.sin(t * 11)
    sc = ellipsoid(0.026, 0.022, 0.026, seg=24, bump=ib)
    g.mesh(place(sc, (0.045, top + 0.016, 0.03)), mi)
    # morangos
    st = mix(col('#C4161C')[None, None], col('#E6362F')[None, None], fbm(256, 6, 3, 76))
    st = mix(st, col('#F7E36B')[None, None], smooth(0.9, 0.95, fbm(256, 40, 2, 77)) * 0.9)
    msb = g.material("morango", base_tex=g.texture(to_img(st)), rough=0.25)
    def berry(s=1.0):
        prof = [(0, 0)]
        for k in range(1, 12):
            t = k / 11
            prof.append((0.012 * s * math.sin(math.pi * t) ** 0.7 * (0.6 + 0.4 * t), 0.024 * s * t))
        prof.append((0, 0.024 * s))
        return lathe(prof, seg=24)
    g.mesh(merge(place(berry(), (0.04, top + 0.009, -0.035), rot_z=math.pi / 2.2, rot_y=0.6),
                 place(berry(0.85), (0.062, top + 0.007, -0.01), rot_z=math.pi / 2.1, rot_y=-0.4)), msb)
    # hortelã
    mm = g.material("hortela", color=col('#2F8A3B'), rough=0.35, double=True)
    def mint(L=0.022, W=0.013):
        m = grid_sheet(W, L, 6, 8, lambda x_, z_: 0.004 * (x_ / W) ** 2 * 4)
        return m.transformed(lambda p: [p[0] * math.sin(math.pi * min(max(p[2] / L + 0.5, 0), 1)) ** 0.7, p[1], p[2]])
    g.mesh(merge(place(mint(), (0.045, top + 0.038, 0.028), rot_y=0.6, rot_x=-0.4),
                 place(mint(0.018, 0.011), (0.052, top + 0.036, 0.036), rot_y=2.3, rot_x=-0.3)), mm)
    g.save(path, "petit-gateau")

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'public/models'
    import os; os.makedirs(out, exist_ok=True)
    pizza(f'{out}/pizza-margherita.glb')
    burger(f'{out}/hamburguer-artesanal.glb')
    poke(f'{out}/poke-de-salmao.glb')
    dessert(f'{out}/petit-gateau.glb')

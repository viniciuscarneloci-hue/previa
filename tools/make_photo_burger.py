"""
Gera o hambúrguer da demonstração em RA a partir de uma foto em ângulo (não vista de cima).

O hambúrguer é quase uma peça de revolução: o contorno da foto vira o perfil (lathe) e cada vértice
é projetado de volta na foto com uma câmera ortográfica inclinada, então a própria foto vira a textura.
A metade de trás espelha a da frente. Escala real em metros.
Uso: python3 tools/make_photo_burger.py <foto.jpg> models/smash-burger.glb
"""
import math, os, sys
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
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

def build(photo):
    im = Image.open(photo).convert('RGB')
    W, H = im.size
    ce, se = math.cos(ELEV), math.sin(ELEV)
    # perfil em metros (r, y) de baixo para cima, reamostrado fino
    ys = np.array([p[0] for p in PROFILE], np.float32)[::-1]
    hw = np.array([p[1] for p in PROFILE], np.float32)[::-1]
    Y = (CY0 - ys) / (K * ce)
    R = hw / K
    Yf = np.linspace(Y[0], Y[-1], 150)
    Rf = gaussian_filter(np.interp(Yf, Y, R), 1.0)
    Rf[-1] = 0
    base = PLATE_H - 0.002
    # fundo: fecha a base do pão
    prof_r = np.concatenate([[0, Rf[0] * 0.85], Rf])
    prof_y = np.concatenate([[0, 0], Yf - Yf[0] + 0.006])
    seg = 192
    P, UV, I = [], [], []
    # textura: recorte do hambúrguer, ampliado e com nitidez
    box = (CX - 215, 290, CX + 215, 700)
    T = 1024
    tex = im.crop(box).resize((T, T), Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=2, percent=70, threshold=1))
    for i, (r, y) in enumerate(zip(prof_r, prof_y)):
        yy = y - 0.006 + Yf[0]  # altura usada na projeção (Y da foto)
        for j in range(seg + 1):
            ph = 2 * math.pi * j / seg
            X, Z = r * math.cos(ph), r * math.sin(ph)
            P.append([X, base + y, Z])
            # ângulo espelhado para a frente (0..pi) e comprimido para fugir das bordas esticadas da foto
            fm = math.atan2(abs(math.sin(ph)), math.cos(ph))
            ps = math.pi / 2 + (fm - math.pi / 2) * 0.72
            sx = CX + K * r * math.cos(ps) * 0.965
            sy = CY0 - K * max(yy, 0) * ce + K * r * math.sin(ps) * 0.965 * se
            UV.append([(sx - box[0]) / (box[2] - box[0]), (sy - box[1]) / (box[3] - box[1])])
    n = seg + 1
    for i in range(len(prof_r) - 1):
        for j in range(seg):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            I += [a, c, b, b, c, d]
    top = Mesh().add(P, UV, I)
    # normal map dos detalhes (gergelim, crosta da carne, queijo)
    f = np.asarray(tex, np.float32).mean(2) / 255
    h = gaussian_filter(f, 0.8) - gaussian_filter(f, 6)
    gy, gx = np.gradient(gaussian_filter(h, 0.6))
    nn = np.dstack([-gx * 6, gy * 6, np.ones_like(h)]); nn /= np.linalg.norm(nn, axis=2, keepdims=True)
    nimg = Image.fromarray(((nn * 0.5 + 0.5) * 255).astype(np.uint8))
    # prato raso na cor do prato da foto
    pc = np.array([0xE6, 0xE1, 0xD8]) / 255   # cerâmica clara da foto
    plate = lathe([(0, 0.0), (PLATE_R * 0.6, 0.0), (PLATE_R * 0.63, 0.003), (PLATE_R * 0.8, 0.004), (PLATE_R * 0.985, PLATE_H - 0.002),
                   (PLATE_R, PLATE_H + 0.002), (PLATE_R * 0.975, PLATE_H + 0.004), (PLATE_R * 0.88, base + 0.001), (0, base)], seg=seg)
    g = GLB16()
    g.mesh(top, g.material('hamburguer', base_tex=g.texture(tex), normal_tex=g.texture(nimg), rough=0.55))
    g.mesh(plate, g.material('prato', color=[float(c) ** 2.2 for c in pc], rough=0.3))
    return g

if __name__ == '__main__':
    build(sys.argv[1]).save(sys.argv[2], 'smash-burger')

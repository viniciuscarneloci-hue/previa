# Previa

Site da Previa: cardápio em realidade aumentada, site da casa, plaquinhas NFC e painel do restaurante.

- Página única e estática: `index.html`, com imagens e vídeo em `img/` e pratos 3D (GLB, escala real) em `models/`.
- Deploy na Vercel: Framework Preset "Other", sem build command e sem output directory.
- `tools/make_photo_dishes.py` gera os pratos 3D a partir de fotos vistas de cima (requer numpy, scipy e Pillow).

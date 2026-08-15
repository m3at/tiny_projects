# Bundled typefaces

The title face is **IM FELL English SC** and the body face is **Inter**. Both are
distributed under the SIL Open Font License 1.1; the corresponding license texts
are kept beside the font files. They come from the canonical Google Fonts
repository so desktop builds do not need a network connection at runtime. CMake copies this directory
beside the desktop executable; the renderer checks that executable-relative location before its source
tree fallback.

- `IMFeENsc28P.ttf` — SHA-256 `102324fb5434bb5da7963533426b0ad44c85bbc9e7755067535c9d11464a176b`
- `Inter-Variable.ttf` — SHA-256 `29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031`

Sources:

- `https://github.com/google/fonts/tree/main/ofl/imfellenglishsc`
- `https://github.com/google/fonts/tree/main/ofl/inter`

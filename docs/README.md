# API documentation

The API reference for `uniqat` is auto-generated from in-source NumPy-style
docstrings using [pdoc](https://pdoc.dev/) and deployed to GitHub Pages on
every push to `main` via `.github/workflows/docs.yml`.

Hosted docs: <https://open-AIMS.github.io/UNIQAT/>

The UNIQAT repository is private to the `open-AIMS` organisation. The hosted
documentation is therefore visible only to collaborators with access to the
repository. See the project README for how to request access.

## Build the docs locally

```bash
pip install -e .[deep,web,dev]
pdoc -o site -d numpy uniqat
python docs/postprocess.py site
python -m http.server -d site
```

Open <http://localhost:8000/> in your browser.

The `[deep,web,dev]` extras are required for the local build because the
`uniqat` package lazily imports the deep learning model classes and the
Gradio web interface; installing those extras lets pdoc follow the full
public API surface.

## Layout

The `site/` directory produced by pdoc contains a single entry point
(`site/uniqat.html`) that indexes the submodules. All public names are
re-exported from the package root; use the root page for discovery and
click through to the submodule pages for full signatures and notes.

## Logo

The GitHub Actions workflow references `docs/assets/logo.png`. The file is
optional; if not present, pdoc falls back to its default styling. A 128x128
pixel PNG at that path will brand the rendered documentation.

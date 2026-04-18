# API documentation

The API reference for `uniqat` is auto-generated from in-source NumPy-style
docstrings using [pdoc](https://pdoc.dev/). Build it locally:

```bash
pip install -e .[deep,web,dev]
pdoc -o site -d numpy uniqat
python docs/postprocess.py site
python -m http.server -d site
```

Open <http://localhost:8000/> in your browser.

## Hosted docs (requires a public repository or GitHub Enterprise)

`.github/workflows/docs.yml` contains a `build` job that renders the site
on every push to `main` as a smoke test, and a `deploy` job that publishes
it to GitHub Pages when manually triggered from the Actions tab. The
`deploy` job is gated by `if: github.event_name == 'workflow_dispatch'`
because GitHub Pages is only available on public repositories (or on
private repositories in a GitHub Enterprise organisation). The UNIQAT
repository is currently private in a free-tier organisation, so the
hosted path is unavailable until either of those constraints changes.

When the repository becomes public, or the organisation upgrades to
Enterprise:

1. Enable Pages at Settings -> Pages -> Source: "GitHub Actions".
2. Trigger the `docs` workflow once from the Actions tab.
3. Remove the `if:` guard on the `deploy` job so every push to `main`
   publishes automatically.

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

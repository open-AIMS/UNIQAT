# Contributing to UNIQAT

Thank you for considering a contribution. UNIQAT ships the research code
accompanying a *Methods in Ecology and Evolution* submission, so stability
and reproducibility of the version-1 results take precedence over
aggressive refactoring. Bug fixes, documentation improvements, new
examples, and carefully-scoped features are all welcome.

If you are unsure whether a change is in scope, open an issue first with
`[discussion]` in the title; we prefer the conversation before the pull
request.

---

## Development setup

Clone the repository and install in editable mode with all extras:

```bash
git clone https://github.com/open-AIMS/UNIQAT.git
cd UNIQAT
python -m venv .venv
source .venv/bin/activate
pip install -e .[deep,web,dev]
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The CPU torch wheel is sufficient for the test suite and the bundled
examples. For GPU training or bulk deep learning inference, swap the CPU
index URL for the matching CUDA-suffixed one from
<https://pytorch.org/get-started/locally/> (for example `cu121`).

---

## Running tests and lint

```bash
pytest tests/ -v
ruff check src/uniqat
```

Both commands are enforced by CI on Python 3.10 and 3.11. Pull requests
that break either check will not be merged.

The test suite takes roughly fifteen seconds on CPU. It includes:

- `test_public_api.py`: verifies every name in `uniqat.__all__` is
  importable and that the version string is valid semver.
- `test_metrics.py`: all 37 metrics on the bundled good and poor images
  return finite floats, and the good image out-scores the poor one.
- `test_assessor_smoke.py`: `UnderwaterImageAssessor` produces a fully
  populated `QualityAssessment` with no NaN or inf values, and the paper's
  Python API compatibility aliases (`feature_quality`, `colour_quality`,
  `blue_water_score`) behave.
- `test_video_assessor_smoke.py`: `VideoQualityAssessor` runs on a
  30-frame synthetic video generated inside the test.
- `test_cli.py`: every console-script entry point exits 0 on `--help`.
- `test_deep_models_shapes.py`: each deep learning architecture accepts
  a dummy CPU batch and returns a supported output type.

New features should ship with tests in the same style.

---

## Style

- The `ruff` configuration in `pyproject.toml` is authoritative. Run
  `ruff check src/uniqat --fix` before pushing.
- Australian English in docstrings and comments: modelling, behaviour,
  colour, centre, analyse, optimise, normalised, organisation.
- No em dashes anywhere (commas, semicolons, colons instead). This
  applies to docstrings, READMEs, comments, and commit messages.
- NumPy-style docstrings on every public function, class, and method.
  pdoc consumes these to generate the hosted API documentation.
- Type hints on public signatures where they clarify intent.

---

## Commit and pull-request guidance

- One logical change per commit. Conventional Commit prefixes (`feat:`,
  `fix:`, `docs:`, `test:`, `refactor:`, `ci:`) are encouraged but not
  required.
- Pull requests should describe what changed, why, and any follow-up
  work. Reference an open issue when applicable.
- If your change affects the public API, mention it in `CHANGELOG.md`
  under an `[Unreleased]` section.
- Do not modify the paper-reported scoring logic in
  `src/uniqat/core/metrics.py` or `src/uniqat/core/assessor.py` without
  a clear, opt-in path (for example, a new keyword argument with a
  default that preserves existing behaviour).

---

## Reporting a bug or requesting a feature

Use the issue templates at
<https://github.com/open-AIMS/UNIQAT/issues/new/choose>. There are three:

- **Bug report** for a reproducible failure.
- **Feature request** for a proposed new capability.
- **Reef data question** for questions about applying UNIQAT to a
  specific survey dataset, a non-reef underwater use case, or an unusual
  camera rig.

If you include code in a bug report, prefer the CPU examples in
`examples/single_image_assessment.py` or `examples/batch_directory.py`
as a minimal reproducer.

---

## Security

For security-sensitive issues, email the corresponding author rather
than opening a public issue: `alzayat.saleh@my.jcu.edu.au`.

---

## Licence

By contributing, you agree that your contributions will be licensed
under the MIT Licence, consistent with the rest of the project.

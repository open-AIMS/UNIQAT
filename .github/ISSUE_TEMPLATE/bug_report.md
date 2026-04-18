---
name: Bug report
about: Report a reproducible failure or incorrect behaviour
title: "[bug] "
labels: bug
---

**Describe the bug**
A clear and concise description of what went wrong.

**Minimal reproducer**
Prefer adapting `examples/single_image_assessment.py` or
`examples/batch_directory.py` to the smallest change that triggers the
bug. Paste the full error traceback.

```python
# your reproducer
```

**Expected behaviour**
What you expected to happen instead.

**Environment**

- OS:
- Python version: (output of `python --version`)
- UNIQAT version: (output of `python -c "import uniqat; print(uniqat.__version__)"`)
- torch version (if relevant): (output of `python -c "import torch; print(torch.__version__)"`)
- Installation method: (`pip install -e .[deep,web]`, `.[deep,web,dev]`, conda env, etc.)
- Hardware: CPU only, or GPU model and CUDA version

**Input image**
If the bug is image-specific, attach the image (or a representative
subset) if you are able, or describe its key properties: resolution,
format, colour space, turbidity, illumination.

**Additional context**
Anything else we should know: custom metrics, unusual environment,
partial reproductions.

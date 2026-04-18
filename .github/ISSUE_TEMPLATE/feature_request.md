---
name: Feature request
about: Propose a new capability or an enhancement
title: "[feature] "
labels: enhancement
---

**Problem**
What are you trying to do that UNIQAT does not currently support? Please
be concrete about the scientific or operational use case.

**Proposed change**
Describe the feature you would like to see. Include API sketches if
applicable.

**Alternatives considered**
What have you already tried or worked around? Why is this feature
preferable?

**Scope check**
The project prioritises bug fixes, documentation, and carefully scoped
additions that preserve reproducibility of the v1.0.0 paper results.
Changes that would alter paper-reported metric definitions or composite
scoring weights should be proposed as opt-in keyword arguments rather
than as defaults.

- [ ] This change preserves v1.0.0 composite score reproducibility.
- [ ] This change is opt-in for existing users (new keyword with
      backward-compatible default).
- [ ] I am willing to help implement and test this change.

**Additional context**
Links to related issues, papers, or external resources.

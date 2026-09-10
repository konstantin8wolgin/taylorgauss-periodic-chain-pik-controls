# Standalone Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a small, reproducible repository containing only the validated bounded periodic-chain Gaussian-shift controls and their scientific limits.

**Architecture:** One installable Python package exposes separate four-site and six-site authorities. Compact JSON evidence pins the untracked upstream files by hash, while focused tests validate observable behavior and isolate the opt-in dense `N_t=4` cost.

**Tech Stack:** Python 3.11+, NumPy, SciPy, optional PyTorch, pytest, Git, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-09-10-standalone-repository-design.md`

## Global Constraints

- Implement only `N=4,N_t<=4` and frozen `N=6,N_t=1` case B.
- State that N=4 is exact and bounded but exponentially dense, and N=6,N_t=1 is confirmed but substantially Trotter biased.
- Do not claim continuum control, scalable large-N behavior, sparse-event sampling, or a sign-problem solution.
- Do not include private paths, secrets, dense result arrays, or external local-state artifacts.
- Create and push the public GitHub repository only after independent review and fresh verification.

---

### Task 1: Provenance-first project skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `LICENSE`
- Create: `docs/provenance.md`
- Create: `evidence/source-lineage.json`
- Create: `evidence/validated-results.json`

**Interfaces:**
- Consumes: audited source hashes and numerical receipts from the controlling worktree
- Produces: package metadata, dependency contract, public scope statement, and machine-readable lineage

- [ ] **Step 1:** Initialize a Tweag-style `src/` Python package skeleton with runtime dependencies `numpy>=2.0` and `scipy>=1.14`, optional `sampler` dependency `torch>=2.6`, and pytest configuration that excludes `slow` by default.
- [ ] **Step 2:** Record all 16 audited file hashes, source base commit `4b2502f9cbd46f61d9b4b41cd1402ae2d072443f`, upstream URL, and untracked-source status in `evidence/source-lineage.json`.
- [ ] **Step 3:** Record the N=4 channel counts/phases/cost boundary and N=6 finite-slice/Hamiltonian values in `evidence/validated-results.json`, with explicit excluded claims.
- [ ] **Step 4:** Write the README and provenance narrative so an unfamiliar reader sees scope and limitations before reproduction commands.
- [ ] **Step 5:** Run `python3 -m json.tool` on both evidence files and `python3 -m build --sdist --wheel` after implementation exists.

### Task 2: Four-site authority and Gaussian shift

**Files:**
- Create: `tests/test_chain4.py`
- Create: `src/taylorgauss_pik/__init__.py`
- Create: `src/taylorgauss_pik/chain4.py`

**Interfaces:**
- Produces: `Chain4Case`, `physical_trace`, `determinant`, `channel_authority`, `dense_authority`, `GaussianShift`, `endpoint_observables`, and `nt4_negative_coefficient`

- [ ] **Step 1:** Write tests with literal physical observables for `N_t=1/2`, determinant reconstruction, Gaussian pullback, `N_t=3` dense reconstruction, the literal `-0.00637424864053005068` four-slice coefficient, and rejected out-of-scope cases.
- [ ] **Step 2:** Run `pytest tests/test_chain4.py -q` and verify collection fails because `taylorgauss_pik.chain4` does not exist.
- [ ] **Step 3:** Implement the minimal validated authority and map code, preserving exact enumeration and resource gates.
- [ ] **Step 4:** Run `pytest tests/test_chain4.py -q` and verify all default tests pass.
- [ ] **Step 5:** Add an opt-in `slow` test that builds the full 43,046,721-label `N_t=4` authority and compares it with the physical trace; do not run it in ordinary CI.

### Task 3: Six-site authority and lazy sampler

**Files:**
- Create: `tests/test_chain6.py`
- Create: `src/taylorgauss_pik/chain6.py`

**Interfaces:**
- Produces: `CaseB`, `spinful_authority`, `occupation_law`, `exact_preflight`, `validate_authority`, `GaussianHistorySampler`, `sample_seed`, and `analytic_controls`

- [ ] **Step 1:** Write tests with literal finite-slice and Hamiltonian observables, their nonzero differences, the 4096-state positive occupation law, independent authority validation, lazy-map pullback, and scope rejection.
- [ ] **Step 2:** Run `pytest tests/test_chain6.py -q` and verify collection fails because `taylorgauss_pik.chain6` does not exist.
- [ ] **Step 3:** Implement the independent spinful sector authority and frozen lazy-map sampler without any time-history or Fourier catalogue.
- [ ] **Step 4:** Run `pytest tests/test_chain6.py -q` and verify all CPU tests pass; PyTorch-only tests skip cleanly if the optional extra is absent.

### Task 4: Reproduction docs, CI, and release receipt

**Files:**
- Create: `docs/method.md`
- Create: `docs/results-and-limits.md`
- Create: `.github/workflows/test.yml`
- Create: `RELEASE_RECEIPT.json`

**Interfaces:**
- Consumes: package APIs and compact evidence
- Produces: public reproduction commands, CI contract, and release audit record

- [ ] **Step 1:** Document finite-slice targets, map identities, observables, and the finite-`N_t` versus Hamiltonian distinction with the validated N=6 numbers.
- [ ] **Step 2:** Add GitHub Actions for Python 3.11, 3.12, and 3.13 running the default CPU suite and package build.
- [ ] **Step 3:** Run `python3 -m pytest -ra`, `python3 -m compileall -q src tests`, JSON validation, and the package build from a clean environment.
- [ ] **Step 4:** Ask an independent agent to review the complete diff for mathematical scope, provenance, privacy, packaging, and test quality; fix every critical or important finding and request re-review.
- [ ] **Step 5:** Write `RELEASE_RECEIPT.json` with commands, results, versions, source lineage, review evidence, and limitations, then rerun the complete verification.
- [ ] **Step 6:** Commit locally, create the public repository with `gh-axi`, push the reviewed commit, create an annotated `v0.1.0` release, and verify the remote repository and release through `gh-axi`.

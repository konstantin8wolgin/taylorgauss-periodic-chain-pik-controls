# Taylor–Gauss periodic-chain PIK controls

Reproducible analytic Gaussian-shift controls for two **strictly bounded** periodic Hubbard-chain calculations:

- `N=4`, `N_t=1..4`: exact finite-dimensional finite-slice controls in floating-point arithmetic. The dense `N_t=4` construction is deliberately retained as negative evidence: 43,046,721 labels, about 1.25 GiB measured peak RSS, and an 81× storage multiplier per extra slice.
- `N=6`, `N_t=1`, `beta=kappa=mu=1`, `U=4`: a confirmed occupation-state/lazy-map sampler with an independent spinful authority. It has substantial Trotter bias relative to the Hamiltonian trace.

These are analytic-shift baselines, not scalable PIK algorithms or sign-problem solutions. This repository does **not** implement `N=6,N_t=2`, `N=7`, large `N`, `N_t≈50`, sparse-event sampling, a continuum extrapolation, or a general sign-problem cure.

## Install and test

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e '.[test,sampler]'
python3 -m pytest -ra
```

The default CPU suite excludes the intentionally expensive full `N=4,N_t=4` enumeration. Run that negative-scaling check explicitly only on a machine with more than 2 GiB free memory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m pytest -m slow tests/test_chain4.py -ra
```

The exact N=6 finite-slice/Hamiltonian authority requires only NumPy and SciPy. PyTorch is an optional dependency for actual lazy-map sampling; sampler tests skip if it is absent.

The supported execution platform is Linux. The copied bounded runners use Linux `resource` accounting for their measured RSS gates.

Small public API and proof-only runner examples:

```bash
python3 - <<'PY'
from taylorgauss_pik.chain6 import occupation_law, spinful_authority
print(spinful_authority()["finite_nt"])
print(len(occupation_law().probabilities))
PY

python3 -m taylorgauss_pik.chain6_run \
  --device cpu --output /tmp/taylorgauss-chain6-proof
```

The second command writes a small proof receipt and representative maps. Adding `--confirm` replays the full 218,103,808-draw campaign and is intentionally not part of ordinary CI.

## Key result: sampling accuracy is not continuum accuracy

For the frozen six-site case, the confirmed sampler targets the symmetric one-slice trace:

| Quantity | Finite `N_t=1` | Hamiltonian ED | Finite minus ED |
|---|---:|---:|---:|
| `log Z` | 14.0247420797 | 13.1678166982 | 0.8569253815 |
| Density | 1.1086464164 | 1.1415980552 | -0.0329516387 |
| Double occupancy | 0.1303521521 | 0.2010432284 | -0.0706910763 |

The discretization difference is much larger than the confirmed Monte Carlo uncertainty. A precise sampler of the finite-slice target does not make that target equal to the Hamiltonian trace.

## Documentation and evidence

- [Method](docs/method.md)
- [Validated results and hard limits](docs/results-and-limits.md)
- [Source and receipt provenance](docs/provenance.md)
- [Machine-readable source lineage](evidence/source-lineage.json)
- [Compact validated results](evidence/validated-results.json)

No private paths, external machine-state artifacts, dense coefficient arrays, full Gaussian streams, or secrets are included.

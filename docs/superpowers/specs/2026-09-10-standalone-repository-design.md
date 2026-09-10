# Standalone periodic-chain PIK controls design

## Purpose and scientific contract

This repository publishes the validated Taylor–Gauss analytic Gaussian-shift controls for a periodic Hubbard chain without carrying the surrounding private research workspace. It implements exactly two bounded authorities:

- `N=4`, `N_t=1..4`: exact finite-dimensional, finite-slice controls evaluated in floating point. The `N_t=4` dense enumeration is preserved as negative scaling evidence, not as a scalable algorithm.
- `N=6`, `N_t=1`, `beta=kappa=mu=1`, `U=4`: the confirmed lazy-map sampler and independent spinful authorities for both the symmetric one-slice target and Hamiltonian exact diagonalization. Their material difference is always reported as Trotter bias.

The repository makes no claim for `N=6,N_t=2`, `N=7`, large `N`, `N_t` near 50, sparse-event sampling, continuum accuracy, or a sign-problem solution.

## Architecture

The installable `taylorgauss_pik` package is split by scientific authority. Six byte-for-byte upstream authority modules preserve source hashes; `chain4.py` and `chain6.py` are small public facades over them. The four-site modules contain the physical trace, Fourier-channel constructions, dense streamed controls, and Gaussian shift. The six-site modules contain the frozen case-B occupation law, shared lazy map, sampler, and independent spinful finite-slice/Hamiltonian authority. Public interfaces validate scope before allocation.

The default Linux CPU test suite exercises the mathematical contracts against literal results and independent authorities. Full dense `N=4,N_t=4` reconstruction remains available as an explicitly selected slow test because it allocates roughly 1.25 GiB peak RSS; default CI instead checks its independently derived negative coefficient formula and the hard scaling boundary. Torch is an optional dependency used only by the sampled N=6 stream; exact N=6 authorities need NumPy and SciPy.

## Evidence and provenance

`docs/provenance.md` and `evidence/source-lineage.json` record the source repository URL, source worktree base commit, the fact that controlling files were untracked there, and SHA-256 hashes for every source, test, report, and receipt audited. `evidence/validated-results.json` carries only concise numerical results and scope limits; it excludes absolute local paths, dense arrays, complete sample streams, and private machine-state artifacts. Upstream receipt hashes preserve byte-level lineage.

The release receipt records the standalone commit, commands, dependency versions, test counts, independent review outcome, and remaining limitations. It is generated before the release commit and then finalized with the release commit SHA through an annotated GitHub release note, avoiding a self-referential file hash.

## Error handling and reproducibility

Invalid lattice sizes, time slices, physics parameters, seeds, draw counts, and unavailable CUDA requests fail closed before expensive work. Reproduction commands pin single-thread BLAS settings where dense numerical order matters. Reports distinguish deterministic authority tolerances from Monte Carlo diagnostics and never substitute sampling precision for discretization accuracy.

## Publication gate

No GitHub repository is created and nothing is pushed until the local package, tests, documentation, evidence manifests, and release receipt have passed fresh verification and an independent read-only code/science review. Critical and important review findings must be resolved and re-reviewed first.

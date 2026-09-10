# Validated results and hard limits

## Four sites through four slices

All reported N=4 cases use `U=4`, `kappa=1`. Cases A, B, C are `(beta,mu)=(1,0),(1,1),(4,1)`.

| `N_t` | Dense labels | Nonzero labels | Negative labels A/B/C | Integrated channel phase A/B/C |
|---:|---:|---:|---:|---:|
| 3 | 531,441 | 10,689 | 0 / 0 / 0 | 1 / 1 / 1 |
| 4 | 43,046,721 | 139,173 | 11,296 / 11,296 / 10,400 | .99678517 / .99659158 / .99999349 |

The N=4,N_t=4 negative coefficient for occupation masks `(3,5,6,10)` is independently

\[
e^{2\beta\mu}\left[{\sinh(2x)\over2}\right]^4
[\cosh^4(2x)-3],\qquad x={\beta\kappa\over4},
\]

which is `-0.00637424864053005068` at `beta=kappa=1`, `mu=0`. This is genuine signed-channel evidence. It is not evidence of a solved sign problem.

Dense N=4,N_t=4 construction retained two 43,046,721-element float arrays (688,747,536 bytes total) and measured 1.246–1.248 GB peak RSS. Another time slice multiplies label storage by 81. This route stops at `N_t=4` and is explicit negative evidence against scaling it toward `N_t≈50`.

## Six sites, one slice

The frozen case is exactly `N=6,N_t=1,beta=U/4=mu=kappa=1`. The source campaign confirmed 128 predeclared seeds with 1,703,936 IID draws each (218,103,808 draws total). All component-wise precision and authority gates passed.

| Quantity | Pooled sample | Pooled MCSE | Finite `N_t=1` | Hamiltonian ED | Finite minus ED |
|---|---:|---:|---:|---:|---:|
| Density | 1.1086323452 | 0.0000097815 | 1.1086464164 | 1.1415980552 | -0.0329516387 |
| Double occupancy | 0.1303309707 | 0.0000101789 | 0.1303521521 | 0.2010432284 | -0.0706910763 |

`log Z` is 14.0247420797 for the finite-slice authority and 13.1678166982 for Hamiltonian ED, a difference of 0.8569253815. The finite-time discretization bias dominates the sampling uncertainty.

The integrated history phase is structurally one at this positive one-slice occupation level and is not a precision observable. It does not imply favorable behavior for multiple slices or larger systems.

## What is not implemented

- N=6,N_t=2 or any temporal bridge
- N=7 or larger N
- N_t near 50
- sparse-event, compressed, or large-N sampling
- a controlled continuum limit
- an end-to-end variance/cost improvement theorem
- a general solution of the fermion sign problem

The N=6 preprocessing retains all 4096 spatial Fock states and therefore remains exponential in N. No extrapolation beyond the measured bounded cases is justified.

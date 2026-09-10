# Method

## Physical convention

For a periodic chain of `N` sites,

\[
H=T+V,\qquad
T=-\kappa\sum_{\langle xy\rangle,\sigma}
(c^\dagger_{x\sigma}c_{y\sigma}+c^\dagger_{y\sigma}c_{x\sigma}),
\]

\[
V={U\over2}\sum_x(n_{x\uparrow}+n_{x\downarrow}-1)^2-\mu N_e.
\]

The usual Hubbard chemical potential is therefore `mu+U/2`; this convention retains the constant `NU/2`. At `Delta tau=beta/N_t`, the implemented finite-slice target is the symmetric product

\[
Z_{N_t}=\operatorname{Tr}\left[
e^{-\Delta\tau V/2}e^{-\Delta\tau T}e^{-\Delta\tau V/2}
\right]^{N_t}.
\]

This equals neither `Tr exp(-beta H)` at finite `N_t` nor a continuum extrapolation. `physical_trace` evaluates the finite-slice trace in the full 256-state N=4 Fock space. `spinful_authority` evaluates both the N=6 finite-slice target and the Hamiltonian trace in all 49 `(N_up,N_down)` sectors, whose largest block has dimension 400.

## Fourier channels and Gaussian shift

For variance `v=beta U/N_t` and a finite channel label `n`, the auxiliary integrand has coefficient `c_n`. Completing the square gives

\[
c_n e^{-(z\cdot z)/(2v)+i n\cdot z}
=c_n e^{-v\|n\|^2/2}
e^{-[(z-ivn)\cdot(z-ivn)]/(2v)}.
\]

The analytic control is the translation

\[
F_{n,t}(x)=x+i t v n,
\]

with inverse `x=z-itvn` and oriented Jacobian one. Each channel keeps its full unnormalized mass `c_n exp(-v||n||^2/2)`. Channel integrals are summed; coordinate maps are never summed.

For N=4, `N_t=1/2` explicitly enumerate `3^(4N_t)` labels. `N_t=3/4` accumulate every closed-history contribution into the full dense ternary array and stream nonzero blocks without thresholding. Negative N=4,N_t=4 coefficients carry their sign; the Gaussian shift removes within-channel oscillation but cannot remove cancellation between signed channels.

For the frozen N=6,N_t=1 case, every one-slice closed history is an occupation state `h=(S,T)`. The 4096 positive masses are sampled directly, then histories sharing mode `n(h)` share the lazy map `x -> x+4in`. No aggregated Fourier catalogue or temporal-history catalogue is built. This reduces the one-slice representation, but its 4096-state preprocessing still scales exponentially with N.

## Observables

Let `d=N N_t`, `v=beta U/N_t`, and let `j=(t,x)` run over all time-slice and site coordinates. At a mapped endpoint `z`, density and double occupancy per site are evaluated as entire insertions:

\[
\rho(z)=1+{\sum_j z_j\over i d v},\qquad
D(z)={1\over2}\left[
{d v-\sum_j z_j^2\over d v^2}+\rho(z)-1
\right].
\]

Every quadratic form is holomorphic (`z\cdot z=\sum_j z_j^2`), with no complex conjugation. Imaginary components are zero-mean transport diagnostics with real sampling variance.

## Numerical authority

The tests compare channel thermodynamics with separately assembled spinful Fock traces, and channel Fourier sums with direct small determinant products. Physical absolute tolerance is `1e-10`; scaled determinant tolerance is `1e-8`. These are floating-point finite-dimensional checks, not interval certificates.

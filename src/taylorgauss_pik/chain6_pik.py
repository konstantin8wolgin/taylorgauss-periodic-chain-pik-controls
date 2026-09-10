"""N6 Nt1 case B only: occupation-state sampling and shared lazy Gaussian maps.

No Fourier-channel catalogue or complete time-history table is constructed.
All finite Fock sectors are retained; this is explicitly exponential in N.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class CaseB:
    n_sites: int = 6
    n_t: int = 1
    beta: float = 1.
    U: float = 4.
    mu: float = 1.
    kappa: float = 1.

    def __post_init__(self):
        if asdict(self) != dict(n_sites=6, n_t=1, beta=1., U=4., mu=1., kappa=1.):
            raise ValueError('approved sampler is frozen to N6 Nt1 case B')
        if type(self.n_sites) is not int or type(self.n_t) is not int:
            raise ValueError('lattice counts must be integers')


@dataclass(frozen=True)
class OccupationLaw:
    case: CaseB
    up_masks: np.ndarray
    hole_masks: np.ndarray
    sectors: np.ndarray
    log_coefficients: np.ndarray
    probabilities: np.ndarray
    log_auxiliary_partition: float

    def modes(self, indices):
        shifts = np.arange(6)
        up = (self.up_masks[np.asarray(indices), None] >> shifts) & 1
        holes = (self.hole_masks[np.asarray(indices), None] >> shifts) & 1
        return up-holes


def occupation_law(case=CaseB()):
    """Finite Fock-sector diagonal law for the one-state closed-loop control."""
    species = []
    for count in range(7):
        states = np.array([s for s in range(64) if s.bit_count() == count])
        index = {int(s): i for i, s in enumerate(states)}
        generator = np.zeros((len(states), len(states)))
        for column, state in enumerate(states):
            state = int(state)
            for src in range(6):
                for dest in ((src+1) % 6, (src-1) % 6):
                    if not (state >> src) & 1 or (state >> dest) & 1:
                        continue
                    lo, hi = sorted((src, dest))
                    between = sum((state >> j) & 1 for j in range(lo+1, hi))
                    generator[index[state ^ (1 << src) ^ (1 << dest)], column] = (-1.)**between
        plus = np.diag(expm(case.beta*case.kappa*generator))
        minus = np.diag(expm(-case.beta*case.kappa*generator))
        species.append((states, plus, minus))
    up_masks, holes, sectors, logc = [], [], [], []
    for p, (up, plus, _) in enumerate(species):
        for q, (hole, _, minus) in enumerate(species):
            up_masks.extend(np.repeat(up, len(hole)))
            holes.extend(np.tile(hole, len(up)))
            sectors.extend([7*p+q]*(len(up)*len(hole)))
            logc.extend((np.log(plus)[:, None]+np.log(minus)[None, :]
                         + case.beta*case.mu*(p-q)).ravel())
    up_masks, holes = np.asarray(up_masks), np.asarray(holes)
    logc = np.asarray(logc)
    modes = (((up_masks[:, None] >> np.arange(6)) & 1)
             - ((holes[:, None] >> np.arange(6)) & 1))
    logmass = logc-case.beta*case.U*np.square(modes).sum(axis=1)/2
    from scipy.special import logsumexp
    log_z = float(logsumexp(logmass))
    probabilities = np.exp(logmass-log_z)
    arrays = (up_masks, holes, np.asarray(sectors), logc, probabilities)
    for array in arrays:
        array.setflags(write=False)
    return OccupationLaw(case, *arrays, log_z)


def exact_preflight(law):
    """Total categorical-plus-Gaussian covariance, not stratified covariance."""
    modes = law.modes(np.arange(len(law.probabilities)))
    s = modes.sum(axis=1)
    r = np.square(modes).sum(axis=1)
    k = (r+s)/2
    p = law.probabilities
    d, v = 6, law.case.beta*law.case.U
    es, er, ek = float(p @ s), float(p @ r), float(p @ k)
    covariance = np.zeros((4, 4))
    covariance[0, 0] = float(p @ np.square(s-es))/d**2
    covariance[0, 1] = covariance[1, 0] = float(p @ ((s-es)*(k-ek)))/d**2
    covariance[1, 1] = float(p @ np.square(k-ek))/d**2 + 1/(2*v*v*d)
    covariance[2, 2] = 1/(v*d)
    covariance[2, 3] = covariance[3, 2] = (es+d/2)/(v*d*d)
    covariance[3, 3] = (er+es+d/4)/(v*d*d)
    rare = (law.up_masks == 63) & (law.hole_masks == 0)
    return dict(component_order=['density.real', 'double.real', 'density.imag', 'double.imag'],
                mean=np.array([1+es/d, ek/d, 0., 0.]),
                single_draw_covariance=covariance,
                minimum_draws=np.ceil(36*np.diag(covariance)/1e-6).astype(np.int64),
                rare_all_double_probability=float(p[rare].sum()),
                integrated_history_phase=1.,
                log_partition=law.log_auxiliary_partition+law.case.beta*law.case.mu*d,
                basis_states=len(p), sectors=49, max_sector_dimension=400)


class GaussianHistorySampler:
    """Nt1 closed-loop state draw followed by its shared lazy F_n Gaussian map."""

    def __init__(self, law, *, device='cpu'):
        import torch
        if device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        self.law, self.device = law, device
        cumulative = np.cumsum(law.probabilities)
        cumulative[-1] = 1.
        differences = np.diff(np.concatenate(([0.], cumulative)))
        if np.any(differences <= 0):
            raise ArithmeticError('floating CDF loses positive state support')
        self.cdf_max_probability_error = float(np.max(np.abs(differences-law.probabilities)))
        self.cdf = torch.tensor(cumulative, dtype=torch.float64, device=device)
        self.up = torch.tensor(law.up_masks.copy(), dtype=torch.int64, device=device)
        self.holes = torch.tensor(law.hole_masks.copy(), dtype=torch.int64, device=device)
        self.logc = torch.tensor(law.log_coefficients.copy(), dtype=torch.float64, device=device)
        self.bits = torch.arange(6, dtype=torch.int64, device=device)

    def generators(self, seed):
        import torch
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError('seed must be an integer in [0,2^31)')
        # CPU MT19937 initialization can alias seeds differing only in high
        # bits. Distinct low-32-bit keys also remain distinct on CUDA Philox.
        history = torch.Generator(device=self.device).manual_seed(2*seed)
        gaussian = torch.Generator(device=self.device).manual_seed(2*seed+1)
        return history, gaussian

    def batch(self, count, history_generator, gaussian_generator):
        import torch
        if type(count) is not int or not 1 <= count <= 65536:
            raise ValueError('batch count must be in [1,65536]')
        uniform = torch.rand(count, dtype=torch.float64, device=self.device, generator=history_generator)
        states = torch.searchsorted(self.cdf, uniform, right=True)
        modes = (((self.up[states, None] >> self.bits) & 1)
                 - ((self.holes[states, None] >> self.bits) & 1)).to(torch.float64)
        sources = torch.randn((count, 6), dtype=torch.float64, device=self.device,
                              generator=gaussian_generator)*2
        endpoints = torch.complex(sources, 4*modes)
        density = 1+endpoints.sum(dim=1)/(24j)
        double = ((24-endpoints.square().sum(dim=1))/96+density-1)/2
        components = torch.stack((density.real, double.real, density.imag, double.imag), dim=1)
        original = (self.logc[states]-endpoints.square().sum(dim=1)/8
                    + 1j*(endpoints*modes).sum(dim=1))
        expected = self.logc[states]-2*modes.square().sum(dim=1)-sources.square().sum(dim=1)/8
        residual = torch.max(torch.abs(original-expected))
        return dict(states=states, sources=sources, endpoints=endpoints, components=components,
                    pullback_residual=residual)

    def draws(self, count, *, seed):
        """Small public replay/control batch; production aggregates bounded batches."""
        batch = self.batch(count, *self.generators(seed))
        return {name: tensor.cpu().numpy() for name, tensor in batch.items()}


def validate_authority(law, authority):
    """Compare means, covariance and rare probability with independent Fock insertions."""
    p = exact_preflight(law)
    a, f = authority['finite_insertions'], authority['finite_nt']
    s, k = a['charge'], a['double_count']
    covariance = np.zeros((4, 4))
    covariance[0, 0] = (a['charge_squared']-s*s)/36
    covariance[0, 1] = covariance[1, 0] = (a['charge_double_count']-s*k)/36
    covariance[1, 1] = (a['double_count_squared']-k*k)/36+1/192
    covariance[2, 2] = 1/24
    covariance[2, 3] = covariance[3, 2] = (s+3)/144
    covariance[3, 3] = (2*k+1.5)/144
    errors = dict(mean_max_error=float(np.max(np.abs(p['mean']-np.array([
        f['density'], f['double_occupancy'], 0., 0.])))),
        log_partition_error=abs(p['log_partition']-f['log_partition']),
        covariance_max_error=float(np.max(np.abs(p['single_draw_covariance']-covariance))),
        rare_probability_error=abs(p['rare_all_double_probability']-a['all_double_probability']))
    return dict(**errors, passed=bool(all(x < 1e-11 for x in errors.values())))


def sample_seed(sampler, count, *, seed, batch_size=65536):
    """Bounded-memory IID stream with stable moments and a reproducible state digest.

    Replay requires the same backend, software, seed, count and batch size. Only
    tiny moment arrays and integer state indices cross the device boundary.
    """
    import hashlib
    import time
    import torch
    if type(count) is not int or count < 2:
        raise ValueError('count must be an integer >=2')
    if type(batch_size) is not int or not 1 <= batch_size <= 65536:
        raise ValueError('batch size must be in [1,65536]')
    if sampler.device == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    history, gaussian = sampler.generators(seed)
    total, mean, m2 = 0, np.zeros(4), np.zeros((4, 4))
    lag_product, previous, first = np.zeros(4), None, None
    counts, digest = np.zeros(4096, dtype=np.int64), hashlib.sha256()
    residual = 0.
    for offset in range(0, count, batch_size):
        n = min(batch_size, count-offset)
        batch = sampler.batch(n, history, gaussian)
        values = batch['components']
        block_mean_t = values.mean(dim=0)
        centered = values-block_mean_t
        block_mean = block_mean_t.cpu().numpy()
        block_m2 = (centered.T @ centered).cpu().numpy()
        batch_residual = float(batch['pullback_residual'].cpu())
        if not (math.isfinite(batch_residual) and np.all(np.isfinite(block_mean))
                and np.all(np.isfinite(block_m2))):
            raise ArithmeticError('nonfinite sampled weight residual or moments')
        delta = block_mean-mean
        m2 += block_m2+np.outer(delta, delta)*(total*n/(total+n))
        mean += delta*n/(total+n)
        total += n
        ends = values[[0, -1]].cpu().numpy()
        lag_product += (values[:-1]*values[1:]).sum(dim=0).cpu().numpy()
        if previous is not None:
            lag_product += previous*ends[0]
        else:
            first = ends[0]
        previous = ends[1]
        states = batch['states'].cpu().numpy().astype('<i8', copy=False)
        counts += np.bincount(states, minlength=4096)
        digest.update(states.tobytes())
        residual = max(residual, batch_residual)
    lag_centered = lag_product-(count+1)*mean*mean+mean*(first+previous)
    covariance = m2/(count-1)
    rare_indices = (sampler.law.up_masks == 63) & (sampler.law.hole_masks == 0)
    return dict(seed=seed, history_key=2*seed, gaussian_key=2*seed+1,
                draws=count, batch_size=batch_size, device=sampler.device,
                mean=mean, single_draw_covariance=covariance,
                mcse=np.sqrt(np.diag(covariance)/count),
                iid_ess=np.full(4, count, dtype=np.int64),
                lag_one_correlation=lag_centered/np.diag(m2), state_counts=counts,
                rare_all_double_hits=int(counts[rare_indices].sum()),
                state_sha256=digest.hexdigest(), pullback_max_abs_log_residual=residual,
                elapsed_seconds=time.perf_counter()-started)


def analytic_controls(sampler):
    """Small direct 6x6 determinant, homotopy and differential-map controls."""
    law = sampler.law
    batch = sampler.draws(64, seed=201)
    x, modes = batch['sources'], law.modes(batch['states'])
    logc = law.log_coefficients[batch['states']]
    path_residual = 0.
    for t in np.linspace(0, 1, 9):
        z = x+4j*t*modes
        lhs = (logc-(z*z).sum(axis=1)/8+1j*t*(z*modes).sum(axis=1)
               -2*(1-t*t)*(modes*modes).sum(axis=1))
        rhs = logc-(x*x).sum(axis=1)/8-2*(modes*modes).sum(axis=1)
        step_residual = float(np.max(np.abs(lhs-rhs)))
        if not math.isfinite(step_residual):
            raise ArithmeticError('nonfinite homotopy control')
        path_residual = max(path_residual, step_residual)
    epsilon = 1e-5
    jacobian = np.stack([((x+epsilon*np.eye(6)[j]+4j*modes)
                         -(x-epsilon*np.eye(6)[j]+4j*modes))/(2*epsilon)
                        for j in range(6)], axis=-1)
    jacobian_residual = float(np.max(np.abs(jacobian-np.eye(6))))
    adj = np.zeros((6, 6))
    for site in range(6):
        adj[site, (site+1) % 6] = adj[site, (site-1) % 6] = 1.
    plus, minus = expm(adj+np.eye(6)), expm(-adj-np.eye(6))
    basis_modes = law.modes(np.arange(4096))
    determinant_residual = 0.
    # Finite occupation trace only; no aggregated Fourier coefficient table.
    for z in x[:12]/4 + .1j*modes[:12]:
        terms = np.exp(law.log_coefficients+1j*(basis_modes @ z))
        direct = (np.linalg.det(np.eye(6)+plus @ np.diag(np.exp(1j*z)))
                  * np.linalg.det(np.eye(6)+minus @ np.diag(np.exp(-1j*z))))
        step_residual = float(abs(terms.sum()-direct)/np.abs(terms).sum())
        if not math.isfinite(step_residual):
            raise ArithmeticError('nonfinite determinant control')
        determinant_residual = max(determinant_residual, step_residual)
    inverse_residual = float(np.max(np.abs(batch['endpoints']-4j*modes-x)))
    return dict(determinant_relative_residual=determinant_residual,
                homotopy_normalization='fixed channel integral',
                homotopy_log_residual=path_residual, jacobian_residual=jacobian_residual,
                inverse_residual=inverse_residual,
                endpoint_log_residual=float(batch['pullback_residual']),
                jacobian_determinant_exact=1.,
                passed=bool(determinant_residual < 1e-11 and path_residual < 1e-11
                            and jacobian_residual < 1e-7 and inverse_residual < 1e-12
                            and float(batch['pullback_residual']) < 1e-11))

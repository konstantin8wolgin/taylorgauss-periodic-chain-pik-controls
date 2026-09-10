"""Dense ALL-label Nt<=4 authority with bounded streamed Gaussian map batches.

Explicit exponential enumeration is intentional and hard bounded. This module
does not implement a compressed representation or a scalable target algorithm.
"""
import argparse
from dataclasses import asdict, dataclass, replace
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

from .chain4_authority import Chain4Case, determinant, physical_trace
from .chain4_pik import endpoint_observables, validate_authority


MAX_RSS_BYTES = 2*1024**3


def resource_gate(started=None):
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > MAX_RSS_BYTES:
        raise ArithmeticError("bounded pilot exceeded 2 GiB peak RSS")
    if started is not None and time.perf_counter()-started > 300:
        raise ArithmeticError("bounded pilot exceeded 300 seconds")


def decode_modes(indices, dimension):
    indices = np.asarray(indices, dtype=np.int64)
    return ((indices[:, None] // (3**np.arange(dimension-1, -1, -1))) % 3 - 1).astype(np.int8)


def exterior_ring4(case, sign):
    """Exact four-ring polynomial: G³=4G, preserving structural zeros.

    This identity uses the four-ring spectrum; it is not a general lattice
    approximation or a compressed channel representation.
    """
    if sign not in (-1, 1):
        raise ValueError("branch sign must be -1 or 1")
    occup = np.array([[(s >> x) & 1 for x in range(4)] for s in range(16)], dtype=np.int64)
    generator = np.zeros((16, 16), dtype=np.int64)
    for state in range(16):
        for src in range(4):
            for dest in ((src+1) % 4, (src-1) % 4):
                if occup[state, src] and not occup[state, dest]:
                    lo, hi = sorted((src, dest))
                    generator[state ^ (1 << src) ^ (1 << dest), state] = (
                        -1 if occup[state, lo+1:hi].sum() % 2 else 1)
    dt = case.beta/case.n_t
    matrix = (np.eye(16) + sign*np.sinh(2*case.kappa*dt)/2*generator
              + np.sinh(case.kappa*dt)**2/2*(generator @ generator))
    matrix *= np.exp(sign*dt*case.mu*occup.sum(axis=1))[None, :]
    return matrix, occup


@dataclass
class DenseAuthority:
    case: Chain4Case
    coefficients: np.ndarray
    history_pair_count: int
    absolute_history_coefficients: np.ndarray | None = None

    def blocks(self, block_size=4096):
        """Visit every dense slot; only exact zeros omit arithmetic, not mass."""
        if type(block_size) is not int or not 1 <= block_size <= 4096:
            raise ValueError("block_size must be an integer in [1,4096]")
        for start in range(0, len(self.coefficients), block_size):
            resource_gate()
            block = self.coefficients[start:start+block_size]
            selected = np.flatnonzero(block != 0)
            if not len(selected):
                continue
            indices = selected + start
            modes = decode_modes(indices, self.case.dimension)
            yield indices, modes, block[selected]

    def statistics(self):
        totals = [[] for _ in range(5)]
        positive = negative = 0
        for _, modes, coefficients in self.blocks():
            mass = coefficients * np.exp(-self.case.variance*np.square(modes).sum(axis=1)/2)
            positive += int(np.count_nonzero(coefficients > 0))
            negative += int(np.count_nonzero(coefficients < 0))
            density = 1 + modes[:, :4].sum(axis=1)/4
            double = (modes == 1).mean(axis=1)
            for parts, value in zip(totals, (mass.sum(), np.abs(mass).sum(),
                                           mass @ mass, mass @ density, mass @ double)):
                parts.append(float(value))
        z, absolute, squares, density, double = map(math.fsum, totals)
        if (not np.isfinite([z, absolute, squares]).all() or absolute <= 0
                or z <= 128*np.finfo(float).eps*absolute):
            raise ArithmeticError("invalid signed integrated partition")
        history_mass = absolute
        cancelled = 0
        if self.absolute_history_coefficients is not None:
            parts = []
            for start in range(0, len(self.coefficients), 4096):
                values = self.absolute_history_coefficients[start:start+4096]
                selected = np.flatnonzero(values != 0)
                if not len(selected):
                    continue
                cancelled += int(np.count_nonzero(self.coefficients[start+selected] == 0))
                modes = decode_modes(start+selected, self.case.dimension)
                parts.append(float(values[selected] @ np.exp(
                    -self.case.variance*np.square(modes).sum(axis=1)/2)))
            history_mass = math.fsum(parts)
        # Conservative bound for coefficient additions only. It excludes the
        # error in elementary functions, products and subsequent weighted sums.
        factor = self.history_pair_count*np.finfo(float).eps
        addition_bound = factor/(1-factor)*history_mass
        return dict(z=z, absolute_mass=absolute, average_phase=z/absolute,
                    mass_concentration_ess=absolute**2/squares,
                    signed_mass_participation=z*z/squares,
                    positive_channels=positive, negative_channels=negative,
                    zero_channels=len(self.coefficients)-positive-negative,
                    density=density/z, double_occupancy=double/z,
                    history_absolute_mass=history_mass,
                    cancelled_to_zero_channels=cancelled,
                    accumulation_absolute_error_bound=addition_bound)

    def thermodynamics(self):
        stats = self.statistics()
        return dict(log_partition=math.log(stats['z'])+4*self.case.beta*self.case.mu,
                    density=stats['density'], double_occupancy=stats['double_occupancy'])

    def fourier(self, fields):
        fields = np.asarray(fields, dtype=complex)
        if fields.ndim != 2 or fields.shape[1] != self.case.dimension or not np.isfinite(fields).all():
            raise ValueError("invalid Fourier probes")
        value = np.zeros(len(fields), dtype=complex)
        scale = np.zeros(len(fields))
        for _, modes, coefficients in self.blocks():
            terms = np.exp(1j*fields @ modes.T)
            value += terms @ coefficients
            scale += np.abs(terms) @ np.abs(coefficients)
        return value, scale


def dense_authority(case: Chain4Case):
    """Accumulate signed closed histories into every dense ternary label slot."""
    a, occup = exterior_ring4(case, 1)
    b, _ = exterior_ring4(case, -1)
    histories = np.array([history for number in range(5)
                         for history in product([s for s in range(16) if s.bit_count() == number],
                                                repeat=case.n_t)], dtype=np.int8)
    next_states = np.roll(histories, -1, axis=1)
    weight_a = np.prod(a[next_states, histories], axis=1)
    weight_b = np.prod(b[next_states, histories], axis=1)
    history_modes = occup[histories].reshape(len(histories), case.dimension)
    powers = 3**np.arange(case.dimension-1, -1, -1)
    encoded = history_modes @ powers
    offset = int(powers.sum())
    coefficients = np.zeros(3**case.dimension, dtype=np.float64)
    absolute_histories = np.zeros_like(coefficients)
    resource_gate()
    for index, weight in enumerate(weight_a):
        labels = offset + encoded[index] - encoded
        np.add.at(coefficients, labels, weight*weight_b)
        np.add.at(absolute_histories, labels, np.abs(weight*weight_b))
    if not np.isfinite(coefficients).all():
        raise ArithmeticError("nonfinite dense channel coefficients")
    coefficients.setflags(write=False)
    absolute_histories.setflags(write=False)
    return DenseAuthority(case, coefficients, len(histories)**2, absolute_histories)


def stream_samples(authority, *, draws, seed, block_size=4096):
    """Sample independent complete signed sums, with bounded field allocation."""
    if type(draws) is not int or not 2 <= draws <= 128:
        raise ValueError("draws must be an integer in [2,128]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    stats = authority.statistics()
    case = authority.case
    v, d = case.variance, case.dimension
    rng = np.random.default_rng(seed)
    replicates = np.zeros((draws, 2), dtype=complex)
    analytic = np.zeros((4, 4))
    source_hash, endpoint_hash = hashlib.sha256(), hashlib.sha256()
    pullback = inverse = jacobian = homotopy = 0.0
    mapped_count = 0
    for _, modes, coefficients in authority.blocks(block_size):
        norm = np.square(modes).sum(axis=1)
        total = modes.sum(axis=1)
        mass = coefficients*np.exp(-v*norm/2)
        signed_weights = mass/stats['z']
        sources = rng.normal(scale=np.sqrt(v), size=(len(modes), draws, d))
        shift = 1j*v*modes[:, None, :]
        endpoints = sources + shift
        source_hash.update(sources.tobytes())
        endpoint_hash.update(endpoints.tobytes())
        values = endpoint_observables(case, endpoints)
        replicates += np.einsum('c,cmo->mo', signed_weights, values)
        phase = np.where(coefficients < 0, np.pi, 0)
        logc = np.log(np.abs(coefficients)) + 1j*phase
        expected = logc[:, None]-v*norm[:, None]/2-np.square(sources).sum(axis=-1)/(2*v)
        original = (logc[:, None]-np.square(endpoints).sum(axis=-1)/(2*v)
                    + 1j*np.einsum('cmd,cd->cm', endpoints, modes))
        pullback = max(pullback, float(np.max(np.abs(original-expected))))
        for flow_time in (0., .5, 1.):
            mapped = sources + flow_time*shift
            recovered = mapped-flow_time*shift
            inverse = max(inverse, float(np.max(np.abs(recovered-sources))))
            path_weight = logc[:, None]-v*norm[:, None]/2-np.square(recovered).sum(axis=-1)/(2*v)
            homotopy = max(homotopy, float(np.max(np.abs(path_weight-expected))))
        # Translation has oriented Jacobian +1 analytically; probe one entry
        # independently with a real finite difference in every nonempty batch.
        x = sources[0, 0, 0]
        step = 1e-5*max(1., abs(x))
        derivative = ((x+step+shift[0, 0, 0])-(x-step+shift[0, 0, 0]))/(2*step)
        jacobian = max(jacobian, float(abs(derivative-1)))
        p2 = signed_weights**2/draws
        analytic[1, 1] += p2.sum()/(2*v*v*d)
        analytic[2, 2] += p2.sum()/(v*d)
        analytic[3, 3] += p2 @ (norm+total+d/4)/(v*d*d)
        analytic[2, 3] += p2 @ (total+d/2)/(v*d*d)
        mapped_count += len(modes)
    analytic[3, 2] = analytic[2, 3]
    components = np.concatenate((replicates.real, replicates.imag), axis=1)
    covariance = np.cov(components, rowvar=False, ddof=1)/draws
    return dict(replicates=replicates, estimate=replicates.mean(axis=0),
                covariance_of_mean=covariance, analytic_covariance_of_mean=analytic,
                mapped_channel_count=mapped_count, original_pullback_residual=pullback,
                inverse_residual=inverse, jacobian_probe_residual=jacobian,
                homotopy_pullback_residual=homotopy,
                sources_sha256=source_hash.hexdigest(), endpoints_sha256=endpoint_hash.hexdigest())


def validate_dense(authority):
    case = authority.case
    physical = physical_trace(case)
    analytic = authority.thermodynamics()
    errors = {key: abs(physical[key]-analytic[key]) for key in physical}
    rng = np.random.default_rng(20260909)
    fields = rng.normal(size=(12, case.dimension)).astype(complex)
    fields[6:] += .15j*rng.normal(size=(6, case.dimension))
    fields[0] = np.pi
    expanded, scale = authority.fourier(fields)
    determinant_error = float(np.max(np.abs(expanded-determinant(case, fields))/scale))
    support = True
    for _, modes, _ in authority.blocks():
        charge = modes.reshape(-1, case.n_t, 4).sum(axis=2)
        support &= bool(np.all(charge == charge[:, :1]))
    control = stream_samples(authority, draws=2, seed=20260909)
    residuals = {key: control[key] for key in (
        'original_pullback_residual', 'inverse_residual', 'homotopy_pullback_residual',
        'jacobian_probe_residual')}
    passed = bool(all(np.isfinite(x) and x <= 1e-10 for x in errors.values())
                  and np.isfinite(determinant_error) and determinant_error <= 1e-8 and support
                  and all(np.isfinite(x) and x <= (1e-8 if 'jacobian' in key else 1e-10)
                          for key, x in residuals.items()))
    return dict(n_t=case.n_t, passed=passed, physical_reference=physical,
                channel_reference=analytic, physical_absolute_errors=errors,
                determinant_scaled_residual=determinant_error, conserved_charge_support=support,
                physical_tolerance=1e-10, determinant_scaled_tolerance=1e-8, **residuals)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_extended(case, output, *, draws=16, seed=20260909):
    if case.n_t not in (3, 4) or not case.allow_nt34:
        raise ValueError("extended pilot requires explicitly enabled Nt=3 or Nt=4")
    if type(draws) is not int or not 2 <= draws <= 128 or type(seed) is not int or seed < 0:
        raise ValueError("draws must be 2..128 and seed a nonnegative integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(schema_version=1, case=asdict(case), draws=draws, seed=seed, checks=[],
                  scope='N=4, Nt<=4; full dense channels, streamed analytic Gaussian shifts',
                  maximum_rss_bytes=MAX_RSS_BYTES, maximum_pilot_seconds=300,
                  numpy_version=np.__version__, rng='PCG64; ascending nonzero label, draw, coordinate')
    try:
        for nt in range(1, case.n_t+1):
            checked = replace(case, n_t=nt, allow_nt2=nt == 2, allow_nt34=nt >= 3)
            stage_start = time.perf_counter()
            if nt <= 2:
                check = validate_authority(checked)
            else:
                authority = dense_authority(checked)
                check = validate_dense(authority)
            check['stage_seconds'] = time.perf_counter()-stage_start
            report['checks'].append(check)
            resource_gate(started)
            if not check['passed']:
                raise ArithmeticError(f"Nt={nt} authority gate failed")
            if nt >= 3 and nt < case.n_t:
                del authority
        stats = authority.statistics()
        sample_started = time.perf_counter()
        samples = stream_samples(authority, draws=draws, seed=seed)
        report['sampling_seconds'] = time.perf_counter()-sample_started
        for key in ('original_pullback_residual', 'homotopy_pullback_residual',
                    'inverse_residual', 'jacobian_probe_residual'):
            report[key] = samples[key]
            if not np.isfinite(samples[key]) or samples[key] > (1e-8 if 'jacobian' in key else 1e-10):
                raise ArithmeticError(f"sample {key} gate failed")
        resource_gate(started)
        np.save(output/'coefficients.npy', authority.coefficients, allow_pickle=False)
        np.save(output/'absolute_history_coefficients.npy', authority.absolute_history_coefficients,
                allow_pickle=False)
        np.savez(output/'replicates.npz', replicates=samples['replicates'],
                 covariance_of_mean=samples['covariance_of_mean'],
                 analytic_covariance_of_mean=samples['analytic_covariance_of_mean'])
        reference = report['checks'][-1]['physical_reference']
        exact = np.array([reference['density'], reference['double_occupancy']])
        errors = np.stack((samples['estimate'].real-exact, samples['estimate'].imag), axis=1)
        mcse = np.sqrt(np.maximum(0, np.diag(samples['covariance_of_mean']))).reshape(2, 2).T
        analytic_mcse = np.sqrt(np.diag(samples['analytic_covariance_of_mean'])).reshape(2, 2).T
        diagnostic = bool(np.all(np.abs(errors) <= 6*mcse+1e-10))
        raw = np.random.default_rng(seed+1).normal(scale=np.sqrt(case.variance), size=(256, case.dimension))
        raw_weights = determinant(case, raw)
        report.update(status='validated_pilot' if diagnostic else 'failed_sampling_diagnostic',
                      deterministic_validation_passed=True,
                      channel_count=len(authority.coefficients), omitted_channel_count=0,
                      mapped_channel_count=samples['mapped_channel_count'],
                      history_pair_count=authority.history_pair_count, channel_statistics=stats,
                      dense_authority_bytes=authority.coefficients.nbytes*2,
                      estimate_real=samples['estimate'].real.tolist(), estimate_imag=samples['estimate'].imag.tolist(),
                      observable_order=['density', 'double_occupancy'],
                      covariance_component_order=['density.real', 'double.real', 'density.imag', 'double.imag'],
                      mcse_real_imag=mcse.tolist(), analytic_mcse_real_imag=analytic_mcse.tolist(),
                      iid_complete_sum_replicates=draws, mc_ess_replicates=draws,
                      ess_note='Independent complete-sum replicates, not categorical channel draws. Mass participation is not MC ESS.',
                      statistical_six_se_diagnostic=diagnostic,
                      six_se_within_1e_minus3=bool(np.all(6*mcse <= 1e-3)),
                      analytic_six_se_within_1e_minus3=bool(np.all(6*analytic_mcse <= 1e-3)),
                      integrated_channel_average_phase=stats['average_phase'],
                      baseline_phase_ratio_256_gaussian_draws=float(abs(raw_weights.mean())/np.abs(raw_weights).mean()),
                      baseline_note='Raw-Gaussian finite-sample diagnostic only; no variance/cost or sign-solution claim.',
                      sources_sha256=samples['sources_sha256'], endpoints_sha256=samples['endpoints_sha256'],
                      replay_note='Every nonzero channel sampled; exact deterministic stream hashes replace retaining all field arrays. Dense zero slots are retained; no threshold.',
                      files_sha256={name: file_sha256(output/name) for name in (
                          'coefficients.npy', 'absolute_history_coefficients.npy', 'replicates.npz')},
                      source_sha256={name: file_sha256(Path(__file__).with_name(name)) for name in (
                          'chain4_authority.py', 'chain4_pik.py', 'chain4_multislice.py')},
                      artifact_bytes=sum(p.stat().st_size for p in output.iterdir()),
                      limitations=['Floating-point finite-slice authority; not an interval or continuum-time certificate.',
                                   'Accumulation bound excludes elementary-function/product error; no coefficient threshold.',
                                   'Full exponential dense representation; no scalability or sign-problem solution.'])
        resource_gate(started)
    except ArithmeticError as error:
        report.update(status='failed_validation', failure_reason=str(error))
    report['elapsed_seconds'] = time.perf_counter()-started
    report['timing_scope'] = 'Includes all lower-slice gates, archive writes and hashes; excludes imports and final JSON.'
    report['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--nt', type=int, choices=(3, 4), required=True)
    parser.add_argument('--allow-nt34', action='store_true')
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--U', type=float, default=4)
    parser.add_argument('--mu', type=float, default=0)
    parser.add_argument('--kappa', type=float, default=1)
    parser.add_argument('--draws', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20260909)
    args = parser.parse_args(argv)
    try:
        case = Chain4Case(beta=args.beta, U=args.U, mu=args.mu, kappa=args.kappa,
                          n_t=args.nt, allow_nt34=args.allow_nt34)
        report = run_extended(case, args.output, draws=args.draws, seed=args.seed)
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(2, f'pilot rejected: {error}\n')
    print(json.dumps(dict(status=report['status'], output=args.output)))
    return 0 if report['status'] == 'validated_pilot' else 1


if __name__ == '__main__':
    raise SystemExit(main())

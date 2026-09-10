"""Frozen N6 Nt1 B confirmation. Run with python -m taylorgauss.chain6_run."""
import json
import math

import numpy as np
from scipy.stats import binom


def confirmation_contract(preflight):
    count = math.ceil(1.1*int(max(preflight['minimum_draws']))/65536)*65536
    return dict(schema=1, n_sites=6, n_t=1, beta=1., U=4., mu=1., kappa=1.,
                seeds=list(range(2026090900, 2026091028)), draws_per_seed=count,
                batch_size=65536, six_mcse_limit=.001, variance_headroom=1.1,
                normal_95_coverage_99_bounds=binom.ppf([.005, .995], 128, .95).astype(int).tolist(),
                rare_hit_99_bounds=binom.ppf([.005, .995], 128*count,
                    preflight['rare_all_double_probability']).astype(int).tolist(),
                integrated_phase_is_precision_target=False,
                wall_budget_seconds=600., process_rss_limit_bytes=2**31,
                gpu_reserved_limit_bytes=512*2**20)


def resource_gate(*, projected_seconds, rss_bytes, gpu_reserved_bytes):
    checks = dict(wall=0 <= projected_seconds <= 600,
                  rss=0 <= rss_bytes <= 2**31,
                  gpu=0 <= gpu_reserved_bytes <= 512*2**20)
    return dict(passed=all(checks.values()), checks=checks,
                projected_seconds=projected_seconds, rss_bytes=rss_bytes,
                gpu_reserved_bytes=gpu_reserved_bytes)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False)+'\n')


def evaluate_confirmation(rows, contract, preflight):
    complete = ([r['seed'] for r in rows] == contract['seeds'] and
                all(r['draws'] == contract['draws_per_seed'] for r in rows))
    if not complete:
        return dict(passed=False, checks=dict(complete=False), completed_seeds=len(rows))
    means = np.array([r['mean'] for r in rows])
    mcse = np.array([r['mcse'] for r in rows])
    error = means-preflight['mean']
    analytic_mcse = np.sqrt(np.diag(preflight['single_draw_covariance'])/contract['draws_per_seed'])
    coverage = (np.abs(error) <= 1.959963984540054*analytic_mcse).sum(axis=0)
    hits = sum(r['rare_all_double_hits'] for r in rows)
    ratio = mcse/analytic_mcse
    pooled_mean = means.mean(axis=0)
    pooled_mcse = np.sqrt(np.square(mcse).sum(axis=0))/len(rows)
    lo, hi = contract['normal_95_coverage_99_bounds']
    rare_lo, rare_hi = contract['rare_hit_99_bounds']
    checks = dict(complete=True, finite=bool(np.all(np.isfinite(means)) and np.all(np.isfinite(mcse))),
                  precision=bool(np.all(6*mcse <= contract['six_mcse_limit'])),
                  authority_agreement=bool(np.all(np.abs(error) <= 6*mcse+1e-11)),
                  pooled_authority_agreement=bool(np.all(np.abs(pooled_mean-preflight['mean']) <= 6*pooled_mcse+1e-11)),
                  variance_agreement=bool(np.all((ratio >= .8) & (ratio <= 1.25))),
                  normal_coverage=bool(lo <= coverage[2] <= hi),
                  rare_hits=bool(rare_lo <= hits <= rare_hi),
                  pullback=all(r['pullback_max_abs_log_residual'] <= 1e-11 for r in rows))
    return dict(passed=all(checks.values()), checks=checks, analytic_mcse=analytic_mcse,
                minimum_empirical_mcse=mcse.min(axis=0), maximum_empirical_mcse=mcse.max(axis=0),
                maximum_six_mcse=6*mcse.max(axis=0),
                coverage_95_counts=coverage, exact_normal_coverage_component='density.imag',
                other_coverage_status='nominal CLT diagnostics, not exact normal calibration',
                rare_all_double_hits=hits, rare_expected_hits=128*contract['draws_per_seed']*preflight['rare_all_double_probability'],
                maximum_absolute_error=np.abs(error).max(axis=0), pooled_mean=pooled_mean,
                pooled_mcse=pooled_mcse, pooled_error=pooled_mean-preflight['mean'],
                max_abs_lag_one=np.max(np.abs([r['lag_one_correlation'] for r in rows]), axis=0),
                exact_test_family_note='Two separate 99% diagnostics; joint false-rejection bound <=2%, not 1%.')


def run(output, *, device='auto', confirm=False):
    import hashlib
    import platform
    import resource
    import subprocess
    import time
    from pathlib import Path
    import torch
    from .chain6_fock import spinful_authority
    from .chain6_pik import (occupation_law, exact_preflight, validate_authority,
                            GaussianHistorySampler, sample_seed, analytic_controls)
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    law = occupation_law()
    preflight = exact_preflight(law)
    contract = confirmation_contract(preflight)
    write_json(output/'manifest.json', contract)
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ('chain6_run.py', 'chain6_pik.py', 'chain6_fock.py')}
    authority = spinful_authority()
    validation = validate_authority(law, authority)
    available = torch.cuda.is_available()
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    selected = ('cuda' if available else 'cpu') if device == 'auto' else device
    if selected == 'cuda' and not available:
        raise ValueError('explicit CUDA request unavailable; use auto for CPU fallback')
    hardware = dict(torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                    cpu=platform.processor(), cuda_available=available, selected_device=selected,
                    cpu_threads=torch.get_num_threads(), cuda_version=torch.version.cuda)
    if selected == 'cuda':
        hardware.update(gpu_name=torch.cuda.get_device_name(0),
                        gpu_free_total_bytes=torch.cuda.mem_get_info(),
                        nvidia_smi=subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used',
                            '--format=csv,noheader'], capture_output=True, text=True, check=False).stdout.strip())
        torch.cuda.reset_peak_memory_stats()
    benchmarks = {}
    for backend in (['cpu', 'cuda'] if selected == 'cuda' else ['cpu']):
        bench_sampler = GaussianHistorySampler(law, device=backend)
        sample_seed(bench_sampler, 65536, seed=101)
        trials = [sample_seed(bench_sampler, 262144, seed=102+i) for i in range(3)]
        benchmarks[backend] = dict(draws_per_trial=262144,
                                  elapsed_seconds=[r['elapsed_seconds'] for r in trials])
    sampler = GaussianHistorySampler(law, device=selected)
    controls = analytic_controls(sampler)
    proof = sample_seed(sampler, 65536, seed=202)
    np.savez_compressed(output/'proof_states.npz', state_counts=proof.pop('state_counts'))
    np.savez_compressed(output/'map_examples.npz', **sampler.draws(256, seed=203))
    projection = (time.perf_counter()-started + 2*max(benchmarks[selected]['elapsed_seconds'])
                  *128*contract['draws_per_seed']/262144)
    def resources(seconds):
        return resource_gate(projected_seconds=seconds,
                             rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                             gpu_reserved_bytes=torch.cuda.max_memory_reserved() if selected == 'cuda' else 0)
    budget = resources(projection)
    receipt = dict(status='proof_only_not_confirmed', contract=contract, exact_preflight=preflight,
                   authority=authority, authority_validation=validation, analytic_controls=controls,
                   hardware=hardware, source_sha256=source_hashes, benchmarks=benchmarks,
                   cost_preflight=budget, proof=proof,
                   floating_cdf_max_probability_error=sampler.cdf_max_probability_error,
                   integrated_phase=dict(value=1., variance=0., role='structural only; not a precision observable'),
                   normalizer=dict(method='exact finite occupation contraction', sampled=False,
                                   log_partition=preflight['log_partition']),
                   no_enumeration=dict(fourier_catalogue=False, time_history_catalogue=False,
                                       spatial_fock_states=4096, spatial_scaling='exponential in N; no scalable-N claim'),
                   replay=dict(requirements='same device backend, software, seed, batch size and count',
                               state_hash='SHA256 of little-endian int64 sampled occupation indices',
                               gaussian_examples='map_examples.npz; seeded regeneration of full Gaussian stream'),
                   finite_nt_minus_hamiltonian={k: authority['finite_nt'][k]-authority['hamiltonian'][k]
                                               for k in authority['finite_nt']})
    write_json(output/'proof.json', receipt)
    if not (validation['passed'] and controls['passed'] and budget['passed']):
        receipt['status'] = 'preflight_failed_confirmation_not_started'
    elif confirm:
        rows, state_counts = [], []
        for seed in contract['seeds']:
            row = sample_seed(sampler, contract['draws_per_seed'], seed=seed)
            state_counts.append(row.pop('state_counts'))
            rows.append(row)
            write_json(output/f'seed-{seed}.json', row)
            if not resources(time.perf_counter()-started)['passed']:
                break
        np.savez_compressed(output/'confirmation_states.npz', state_counts=np.array(state_counts))
        receipt['confirmation'] = evaluate_confirmation(rows, contract, preflight)
        receipt['confirmation_seed_rows'] = rows
        receipt['status'] = 'confirmation_passed' if receipt['confirmation']['passed'] else 'confirmation_failed'
    receipt['actual_resources'] = resources(time.perf_counter()-started)
    if not receipt['actual_resources']['passed']:
        receipt['status'] = 'resource_boundary_failed'
    if selected == 'cuda':
        receipt['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    write_json(output/'receipt.json', receipt)
    print(json.dumps(dict(output=str(output), status=receipt['status'],
                          elapsed_seconds=receipt['actual_resources']['projected_seconds'])), flush=True)
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='new directory; never overwrite a run')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--confirm', action='store_true', help='run all 128 frozen confirmation seeds after preflight')
    args = parser.parse_args()
    result = run(args.output, device=args.device, confirm=args.confirm)
    raise SystemExit(0 if result['status'] in ('proof_only_not_confirmed', 'confirmation_passed') else 1)

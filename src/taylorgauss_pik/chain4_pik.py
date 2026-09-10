"""Explicit all-channel Gaussian PIK control for the bounded four-site pilot."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.special import logsumexp

from .chain4_authority import (
    Chain4Case, ChannelAuthority, channel_authority, determinant, physical_trace,
)


class ChannelPIK:
    """F_n,s(x)=x+i*s*v*n; full unnormalized homotopy weight is retained.

    W_n,s(z)=w_n exp(-(z-i*s*v*n)^2/(2*v)), w_n=c_n exp(-v*n²/2).
    W_n,1 is the original Fourier channel. At intermediate times this is a
    moving normalized-shape homotopy, not the fixed original W_n.
    """

    def __init__(self, authority: ChannelAuthority):
        self.authority = authority

    def _fields(self, fields):
        fields = np.asarray(fields)
        expected = (len(self.authority.modes), self.authority.case.dimension)
        if fields.ndim != 3 or (fields.shape[0], fields.shape[2]) != expected:
            raise ValueError("fields must have shape (all_channels, draws, dimension)")
        if not np.isfinite(fields).all():
            raise ValueError("fields must be finite")
        return fields

    @staticmethod
    def _time(time):
        if not np.isscalar(time) or not np.isfinite(time) or not 0 <= time <= 1:
            raise ValueError("PIK time must be finite in [0,1]")
        return time

    def velocity(self):
        return 1j * self.authority.case.variance * self.authority.modes[:, None, :]

    def forward(self, sources, time=1.0):
        sources = self._fields(sources)
        if np.iscomplexobj(sources) and np.any(sources.imag != 0):
            raise ValueError("source coordinates must be real")
        return sources + self._time(time) * self.velocity()

    def inverse(self, endpoints, time=1.0):
        return self._fields(endpoints) - self._time(time) * self.velocity()

    def log_jacobian(self, fields):
        return np.zeros(self._fields(fields).shape[:2], dtype=np.complex128)

    def log_weight(self, fields, time=1.0):
        shifted = self.inverse(fields, time)
        value = (self.authority.log_masses[:, None]
                 - np.square(shifted).sum(axis=-1)/(2*self.authority.case.variance))
        return np.asarray(value, dtype=np.complex128)


def endpoint_observables(case: Chain4Case, fields):
    """Entire physical numerator estimators. Square z, never conjugate z."""
    fields = np.asarray(fields, dtype=np.complex128)
    if fields.shape[-1] != case.dimension or not np.isfinite(fields).all():
        raise ValueError("invalid endpoint fields")
    density = 1 + fields.sum(axis=-1)/(1j*case.beta*case.U*4)
    q2 = (case.dimension*case.variance - np.square(fields).sum(axis=-1))/(
        case.variance**2 * case.dimension
    )
    return np.stack((density, (q2+density-1)/2), axis=-1)


def sample_channels(pik: ChannelPIK, *, draws_per_channel: int, seed: int):
    """Independent Gaussian draws in every label, with no target truncation.

    Zero-coefficient labels are retained; their draws have zero estimator weight.
    This bounded all-label control does NOT implement a scalable target sampler.
    """
    if type(draws_per_channel) is not int or not 2 <= draws_per_channel <= 128:
        raise ValueError("draws_per_channel must be an integer in [2,128]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    authority = pik.authority
    shape = (len(authority.modes), draws_per_channel, authority.case.dimension)
    sources = np.random.default_rng(seed).normal(size=shape) * np.sqrt(authority.case.variance)
    endpoints = pik.forward(sources)
    values = endpoint_observables(authority.case, endpoints)
    # Each column is an independent replicate of the complete channel sum.
    replicates = np.einsum("c,cmo->mo", authority.probabilities, values)
    components = np.concatenate((replicates.real, replicates.imag), axis=1)
    covariance = np.cov(components, rowvar=False, ddof=1) / draws_per_channel
    standard_error = np.sqrt(np.maximum(0, np.diag(covariance))).reshape(2, 2).T
    # Independent analytic Gaussian covariance, not a replacement for the draws.
    d, v = authority.case.dimension, authority.case.variance
    r = np.square(authority.modes).sum(axis=1)
    s = authority.modes.sum(axis=1)
    p2 = authority.probabilities**2 / draws_per_channel
    analytic_covariance = np.zeros((4, 4))
    analytic_covariance[1, 1] = p2.sum() / (2*v*v*d)
    analytic_covariance[2, 2] = p2.sum() / (v*d)
    analytic_covariance[3, 3] = p2 @ (r+s+d/4) / (v*d*d)
    analytic_covariance[2, 3] = analytic_covariance[3, 2] = p2 @ (s+d/2) / (v*d*d)
    return {
        "sources": sources,
        "endpoints": endpoints,
        "replicates": replicates,
        "estimate": replicates.mean(axis=0),
        "standard_error": standard_error,
        "covariance_of_mean": covariance,
        "analytic_covariance_of_mean": analytic_covariance,
    }


def validate_authority(case: Chain4Case):
    """Deterministic finite-point/numerical checks, not an interval certificate."""
    channels = channel_authority(case)
    physical = physical_trace(case)
    analytic = channels.thermodynamics()
    errors = {name: abs(analytic[name] - physical[name]) for name in physical}
    rng = np.random.default_rng(20260908)
    fields = rng.normal(size=(12, case.dimension)).astype(complex)
    fields[6:] += 0.15j * rng.normal(size=(6, case.dimension))
    fields[0] = np.pi  # Includes a zero-prone point in the half-filled target.
    expanded = np.exp(1j * fields @ channels.modes.T) @ channels.coefficients
    scale = np.exp(-fields.imag @ channels.modes.T) @ channels.coefficients
    direct = determinant(case, fields)
    residual = float(np.max(np.abs(expanded-direct) / scale))
    charges = channels.modes.reshape(-1, case.n_t, 4).sum(axis=2)
    conservation = bool(np.all(channels.coefficients[np.any(charges != charges[:, :1], axis=1)] == 0))
    # Evaluate the original Fourier weight independently of log_weight(), so
    # this gate also validates the Nt=1 endpoint before any Nt=2 samples.
    sources = rng.normal(size=(len(channels.modes), 2, case.dimension)) * np.sqrt(case.variance)
    pik = ChannelPIK(channels)
    endpoints = pik.forward(sources)
    live = channels.coefficients > 0
    original = (np.log(channels.coefficients[live])[:, None]
                - np.square(endpoints[live]).sum(axis=-1)/(2*case.variance)
                + 1j*np.einsum("cmd,cd->cm", endpoints[live], channels.modes[live]))
    source_weight = (channels.log_masses[live, None]
                     - np.square(sources[live]).sum(axis=-1)/(2*case.variance))
    pullback = float(np.max(np.abs(
        original + pik.log_jacobian(endpoints)[live] - source_weight)))
    passed = bool(
        all(np.isfinite(e) and e <= 1e-10 for e in errors.values())
        and np.isfinite(residual) and residual <= 1e-8 and conservation
        and np.isfinite(pullback) and pullback <= 1e-10
    )
    return {
        "n_t": case.n_t, "passed": passed,
        "physical_reference": physical, "channel_reference": analytic,
        "physical_absolute_errors": errors,
        "determinant_scaled_residual": residual,
        "original_channel_pullback_residual": pullback,
        "original_channel_pullback_tolerance": 1e-10,
        "conserved_charge_support": conservation,
        "physical_tolerance": 1e-10, "determinant_scaled_tolerance": 1e-8,
        "authority_label": "explicit bounded four-site numerical authority",
    }


def _save_report(output, report):
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_pilot(case: Chain4Case, output, *, draws_per_channel=16, seed=20260908):
    """Create a fresh replay directory, validating before generating samples."""
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "case": asdict(case), "seed": seed,
        "draws_per_channel": draws_per_channel,
        "scope": "bounded N=4, Nt<=2 authority/pilot; explicit enumeration",
        "map": "F_n,s(x)=x+i*s*v*n; Gaussian shift control",
        "partition_convention": "Z_N=exp(beta*mu*N)*sum(c_n*exp(-v*n²/2))",
        "checks": [],
    }
    cases = ([replace(case, n_t=1, allow_nt2=False)] if case.n_t == 2 else []) + [case]
    for checked in cases:
        check = validate_authority(checked)
        report["checks"].append(check)
        if not check["passed"]:
            report["status"] = "failed_validation"
            _save_report(output, report)
            return report
    authority = channel_authority(case)
    pik = ChannelPIK(authority)
    batch = sample_channels(pik, draws_per_channel=draws_per_channel, seed=seed)
    sources = batch["sources"]
    live = authority.coefficients > 0
    residuals = []
    for flow_time in (0.0, 0.5, 1.0):
        mapped = pik.forward(sources, flow_time)
        actual = pik.log_weight(mapped, flow_time)[live] + pik.log_jacobian(mapped)[live]
        expected = authority.log_masses[live, None] - np.square(sources[live]).sum(axis=-1)/(2*case.variance)
        residuals.append(float(np.max(np.abs(actual-expected))))
    report["path_log_weight_residual"] = max(residuals)
    if not np.isfinite(residuals).all() or max(residuals) > 1e-10:
        report["status"] = "failed_validation"
        _save_report(output, report)
        return report
    archive = output / "samples.npz"
    np.savez_compressed(
        archive, modes=authority.modes, coefficients=authority.coefficients,
        log_masses=authority.log_masses, probabilities=authority.probabilities,
        sources=sources, endpoints=batch["endpoints"], midpoint=pik.forward(sources, 0.5),
        path_times=np.array([0., 0.5, 1.]),
        oriented_log_jacobian=pik.log_jacobian(sources),
        endpoint_log_weight=pik.log_weight(batch["endpoints"]),
        replicates=batch["replicates"], covariance_of_mean=batch["covariance_of_mean"],
        analytic_covariance_of_mean=batch["analytic_covariance_of_mean"],
    )
    exact = report["checks"][-1]["physical_reference"]
    reference = np.array([exact["density"], exact["double_occupancy"]])
    error = np.stack((batch["estimate"].real-reference, batch["estimate"].imag), axis=1)
    diagnostic = bool(np.all(np.abs(error) <= 6*batch["standard_error"] + 1e-10))
    # A bounded baseline diagnostic; it is not an absolute-weight sample or a
    # statistically certified average-sign estimate.
    raw = np.random.default_rng(seed+1).normal(
        scale=np.sqrt(case.variance), size=(256, case.dimension)
    )
    raw_weights = determinant(case, raw)
    baseline_phase = float(abs(raw_weights.mean()) / np.abs(raw_weights).mean())
    report.update({
        "status": "validated_pilot" if diagnostic else "failed_sampling_diagnostic",
        "deterministic_validation_passed": True,
        "channel_count": len(authority.modes),
        "nonzero_channel_count": int(live.sum()), "omitted_channel_count": 0,
        "coefficient_pair_count": 256 if case.n_t == 1 else 4900,
        "estimate_real": batch["estimate"].real.tolist(),
        "estimate_imag": batch["estimate"].imag.tolist(),
        "observable_order": ["density", "double_occupancy"],
        "standard_error_real_imag": batch["standard_error"].tolist(),
        "analytic_standard_error_real_imag": np.sqrt(
            np.diag(batch["analytic_covariance_of_mean"])).reshape(2, 2).T.tolist(),
        "covariance_component_order": ["density.real", "double.real", "density.imag", "double.imag"],
        "statistical_six_se_diagnostic": diagnostic,
        "six_se_within_1e_minus3": bool(np.all(6*batch["standard_error"] <= 1e-3)),
        "statistical_note": "Independent replicate standard errors; not a rigorous confidence certificate. No clipping of complex observables.",
        "integrated_channel_average_phase": 1.0,
        "baseline_phase_ratio_256_gaussian_draws": baseline_phase,
        "baseline_note": "Finite-sample raw-Gaussian diagnostic only; no sign-solution claim.",
        "log_gaussian_integral_factor": case.dimension/2*np.log(2*np.pi*case.variance),
        "log_unnormalized_auxiliary_integral": float(
            case.dimension/2*np.log(2*np.pi*case.variance) + logsumexp(authority.log_masses)),
        "samples_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("chain4_pik.py", "chain4_authority.py")},
        "elapsed_seconds": time.perf_counter()-started,
        "timing_scope": "run_pilot including gates and sample archive, excluding Python/package startup and final JSON write",
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "artifact_bytes": archive.stat().st_size,
        "limitations": [
            "Analytic Gaussian-channel control; not the nontrivial sparse-event/semigroup solver.",
            "Enumeration is explicit and hard-bounded; no Nt>=3 or larger-N target path.",
            "One/two-slice integrated channel positivity is special; no general sign-problem claim.",
        ],
    })
    _save_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--allow-nt2", action="store_true")
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--U", type=float, default=4)
    parser.add_argument("--mu", type=float, default=0)
    parser.add_argument("--kappa", type=float, default=1)
    parser.add_argument("--draws-per-channel", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args(argv)
    try:
        case = Chain4Case(beta=args.beta, U=args.U, mu=args.mu, kappa=args.kappa,
                          n_t=args.nt, allow_nt2=args.allow_nt2)
        report = run_pilot(case, args.output, draws_per_channel=args.draws_per_channel,
                           seed=args.seed)
    except (ValueError, OSError, ArithmeticError) as exc:
        parser.exit(2, f"pilot rejected: {exc}\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "validated_pilot" else 1


if __name__ == "__main__":
    raise SystemExit(main())

import numpy as np
import pytest

from taylorgauss_pik.chain4 import (
    Chain4Case,
    GaussianShift,
    channel_authority,
    dense_authority,
    determinant,
    nt4_negative_coefficient,
    physical_trace,
    validate_dense,
)


def test_one_slice_physical_trace_matches_frozen_case_a_values():
    result = physical_trace(Chain4Case())
    assert result == pytest.approx(
        {
            "log_partition": 5.063055938018466,
            "density": 1.0,
            "double_occupancy": 0.056395158136207886,
        },
        abs=1e-12,
    )


@pytest.mark.parametrize("n_t", [1, 2])
def test_explicit_channels_reconstruct_independent_physical_trace(n_t):
    case = Chain4Case(n_t=n_t, allow_nt2=n_t == 2)
    actual = channel_authority(case).thermodynamics()
    assert actual == pytest.approx(physical_trace(case), abs=1e-10)


@pytest.mark.parametrize("n_t", [1, 2])
def test_explicit_channels_reconstruct_complex_determinant(n_t):
    case = Chain4Case(n_t=n_t, allow_nt2=n_t == 2)
    authority = channel_authority(case)
    fields = np.random.default_rng(731 + n_t).normal(size=(4, case.dimension)).astype(complex)
    fields[2:] += 0.17j
    expanded = np.exp(1j * fields @ authority.modes.T) @ authority.coefficients
    scale = np.exp(-fields.imag @ authority.modes.T) @ authority.coefficients
    assert np.max(np.abs(expanded - determinant(case, fields)) / scale) < 1e-8


def test_gaussian_shift_preserves_each_original_channel_weight():
    case = Chain4Case()
    authority = channel_authority(case)
    shift = GaussianShift(authority)
    sources = np.random.default_rng(918).normal(size=(81, 3, 4)) * np.sqrt(case.variance)
    endpoints = shift.forward(sources)
    live = authority.coefficients > 0
    original = (
        np.log(authority.coefficients[live])[:, None]
        - np.square(endpoints[live]).sum(axis=-1) / (2 * case.variance)
        + 1j * np.einsum("cmd,cd->cm", endpoints[live], authority.modes[live])
    )
    expected = (
        authority.log_masses[live, None]
        - np.square(sources[live]).sum(axis=-1) / (2 * case.variance)
    )
    assert np.max(np.abs(original - expected)) < 1e-10
    assert np.max(np.abs(shift.inverse(endpoints) - sources)) == 0


def test_three_slice_dense_authority_reconstructs_physical_trace():
    case = Chain4Case(n_t=3, allow_nt34=True)
    authority = dense_authority(case)
    assert authority.coefficients.shape == (531441,)
    assert authority.statistics()["negative_channels"] == 0
    assert authority.thermodynamics() == pytest.approx(physical_trace(case), abs=1e-10)


def test_four_slice_negative_coefficient_matches_independent_closed_form():
    assert nt4_negative_coefficient(beta=1.0, kappa=1.0, mu=0.0) == pytest.approx(
        -0.00637424864053005068,
        rel=1e-13,
    )


def test_four_site_scope_rejects_unimplemented_time_slice():
    with pytest.raises(ValueError, match="Nt=1 through Nt=4"):
        Chain4Case(n_t=5, allow_nt34=True)


@pytest.mark.slow
def test_four_slice_full_dense_authority_preserves_negative_scaling_evidence():
    case = Chain4Case(n_t=4, allow_nt34=True)
    authority = dense_authority(case)
    assert authority.coefficients.shape == (43046721,)
    assert authority.statistics()["negative_channels"] == 11296
    assert validate_dense(authority)["passed"]

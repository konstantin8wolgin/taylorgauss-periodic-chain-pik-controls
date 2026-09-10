import numpy as np
import pytest

from taylorgauss_pik.chain6 import (
    CaseB,
    GaussianHistorySampler,
    analytic_controls,
    exact_preflight,
    occupation_law,
    sample_seed,
    spinful_authority,
    validate_authority,
)


@pytest.fixture(scope="module")
def authority():
    return spinful_authority()


def test_one_slice_authority_matches_frozen_finite_values(authority):
    assert authority["finite_nt"] == pytest.approx(
        {
            "log_partition": 14.024742079707293,
            "density": 1.108646416446819,
            "double_occupancy": 0.13035215212424592,
        },
        abs=1e-10,
    )


def test_hamiltonian_authority_exposes_substantial_trotter_bias(authority):
    assert authority["hamiltonian"] == pytest.approx(
        {
            "log_partition": 13.16781669816179,
            "density": 1.1415980551663614,
            "double_occupancy": 0.20104322839752392,
        },
        abs=1e-10,
    )
    differences = {
        key: authority["finite_nt"][key] - authority["hamiltonian"][key]
        for key in authority["finite_nt"]
    }
    assert differences == pytest.approx(
        {
            "log_partition": 0.8569253815455031,
            "density": -0.03295163871954232,
            "double_occupancy": -0.070691076273278,
        },
        abs=1e-10,
    )


def test_occupation_law_retains_every_positive_state():
    law = occupation_law()
    assert len(law.probabilities) == 4096
    assert len(np.unique(law.sectors)) == 49
    assert np.all(law.probabilities > 0)
    assert law.probabilities.sum() == pytest.approx(1.0, abs=1e-14)
    assert exact_preflight(law)["rare_all_double_probability"] == pytest.approx(
        8.112074017488324e-7,
        rel=1e-12,
    )


def test_lazy_law_matches_independent_spinful_authority(authority):
    law = occupation_law()
    result = validate_authority(law, authority)
    assert result["passed"]
    assert result["mean_max_error"] < 1e-11
    assert result["covariance_max_error"] < 1e-11


def test_frozen_case_rejects_unimplemented_six_site_time_slice():
    with pytest.raises(ValueError, match="frozen to N6 Nt1 case B"):
        CaseB(n_t=2)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("beta", 2.0),
        ("U", 3.0),
        ("mu", 0.0),
        ("kappa", 0.5),
        ("n_sites", 5),
        ("n_t", 2),
    ],
)
def test_public_spinful_authority_rejects_every_nonfrozen_parameter(parameter, value):
    with pytest.raises(ValueError, match="frozen to N6 Nt1 case B"):
        spinful_authority(**{parameter: value})


def test_cpu_lazy_map_preserves_weights_and_replays():
    pytest.importorskip("torch")
    sampler = GaussianHistorySampler(occupation_law(), device="cpu")
    first = sample_seed(sampler, 2500, seed=123, batch_size=1024)
    replay = sample_seed(sampler, 2500, seed=123, batch_size=1024)
    assert first["state_sha256"] == replay["state_sha256"]
    assert first["pullback_max_abs_log_residual"] < 1e-11
    assert np.array_equal(first["mean"], replay["mean"])


def test_small_direct_determinant_and_homotopy_controls_pass():
    pytest.importorskip("torch")
    result = analytic_controls(GaussianHistorySampler(occupation_law(), device="cpu"))
    assert result["passed"]
    assert result["determinant_relative_residual"] < 1e-11
    assert result["homotopy_log_residual"] < 1e-11

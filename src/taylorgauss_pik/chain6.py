"""Public frozen six-site, one-slice lazy-map control interfaces."""

from .chain6_fock import spinful_authority as _spinful_authority
from .chain6_pik import (
    CaseB,
    GaussianHistorySampler,
    OccupationLaw,
    analytic_controls,
    exact_preflight,
    occupation_law,
    sample_seed,
    validate_authority,
)
from .chain6_run import confirmation_contract, evaluate_confirmation, resource_gate


def spinful_authority(
    *, beta=1.0, U=4.0, mu=1.0, kappa=1.0, n_sites=6, n_t=1
):
    """Return both frozen case-B authorities; reject every broader scope."""
    case = CaseB(
        n_sites=n_sites,
        n_t=n_t,
        beta=beta,
        U=U,
        mu=mu,
        kappa=kappa,
    )
    return _spinful_authority(**case.__dict__)


__all__ = [
    "CaseB",
    "GaussianHistorySampler",
    "OccupationLaw",
    "analytic_controls",
    "confirmation_contract",
    "evaluate_confirmation",
    "exact_preflight",
    "occupation_law",
    "resource_gate",
    "sample_seed",
    "spinful_authority",
    "validate_authority",
]

"""Bounded four-site physical authorities; original dense-map control is Nt<=2."""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp


@dataclass(frozen=True)
class Chain4Case:
    beta: float = 1.0
    U: float = 4.0
    mu: float = 0.0
    kappa: float = 1.0
    n_sites: int = 4
    n_t: int = 1
    allow_nt2: bool = False
    allow_nt34: bool = False

    def __post_init__(self):
        if type(self.n_sites) is not int or self.n_sites != 4:
            raise ValueError("bounded authority requires exactly four sites")
        if type(self.n_t) is not int or self.n_t not in (1, 2, 3, 4):
            raise ValueError("bounded authority only supports Nt=1 through Nt=4")
        if self.n_t == 2 and self.allow_nt2 is not True:
            raise ValueError("Nt=2 requires explicit allow_nt2=True")
        if self.n_t >= 3 and self.allow_nt34 is not True:
            raise ValueError("Nt=3/4 requires explicit allow_nt34=True")
        for name, lower, upper in (
            ("beta", 0, 4), ("U", 0, 8), ("kappa", 0, 1), ("mu", -4, 4)
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < lower or value > upper or (name in ("beta", "U") and value == 0):
                raise ValueError(f"{name} outside pilot numerical envelope")
        denominators = (self.variance, self.variance**2*self.dimension, self.beta*self.U*4)
        if any(value <= 0 or not math.isfinite(value) or not math.isfinite(1/value)
               for value in denominators):
            raise ValueError("Gaussian scale/observable denominators are not representable")

    @property
    def variance(self):
        return self.U * self.beta / self.n_t

    @property
    def dimension(self):
        return 4 * self.n_t


def physical_trace(case: Chain4Case) -> dict[str, float]:
    """Independent 256-state electron trace; no Fourier or determinant inputs."""
    kinetic = np.zeros((256, 256))
    number = np.zeros(256)
    double = np.zeros(256)
    potential = np.zeros(256)
    # Interleaved spin orbitals; include the seam (3,0) exactly once.
    for state in range(256):
        occup = [(state >> orbital) & 1 for orbital in range(8)]
        number[state] = sum(occup)
        double[state] = sum(occup[2*x] * occup[2*x+1] for x in range(4)) / 4
        potential[state] = case.U / 2 * sum(
            (occup[2*x] + occup[2*x+1] - 1)**2 for x in range(4)
        ) - case.mu * number[state]
        for x in range(4):
            y = (x + 1) % 4
            for dest, src in ((x, y), (y, x)):
                for spin in range(2):
                    a, b = 2 * dest + spin, 2 * src + spin
                    if not occup[b] or occup[a]:
                        continue
                    intermediate = state ^ (1 << b)
                    sign = (-1)**(
                        (state & ((1 << b) - 1)).bit_count()
                        + (intermediate & ((1 << a) - 1)).bit_count()
                    )
                    kinetic[intermediate | (1 << a), state] -= case.kappa * sign
    dt = case.beta / case.n_t
    half = np.exp(-dt * potential / 2)
    symmetric = half[:, None] * expm(-dt * kinetic) * half[None, :]
    diagonal = np.diag(np.linalg.matrix_power(symmetric, case.n_t))
    z = diagonal.sum()
    if not np.isfinite(z) or z <= 0:
        raise ArithmeticError("invalid physical partition")
    return {
        "log_partition": float(np.log(z)),
        "density": float(diagonal @ number / (4 * z)),
        "double_occupancy": float(diagonal @ double / z),
    }


def determinant(case: Chain4Case, phi: np.ndarray) -> np.ndarray:
    """Direct holomorphic 4x4 determinant product, independent of coefficients."""
    phi = np.asarray(phi, dtype=np.complex128)
    if phi.ndim < 1 or phi.shape[-1] != case.dimension or not np.isfinite(phi).all():
        raise ValueError("fields must be finite with final dimension 4*Nt")
    shape = phi.shape[:-1]
    fields = phi.reshape(-1, case.n_t, 4)
    h = np.roll(np.eye(4), 1, axis=0) + np.roll(np.eye(4), -1, axis=0)
    values, vectors = np.linalg.eigh(case.kappa * h + case.mu * np.eye(4))
    result = np.ones(len(fields), dtype=np.complex128)
    for sign in (1, -1):
        a = (vectors * np.exp(sign * case.beta / case.n_t * values)) @ vectors.T
        p = np.broadcast_to(np.eye(4), (len(fields), 4, 4)).astype(complex).copy()
        for t in range(case.n_t):
            phases = np.exp(sign * 1j * fields[:, t])
            p = (a[None, :, :] * phases[:, None, :]) @ p
        result *= np.linalg.det(np.eye(4) + p)
    return result.reshape(shape)


@dataclass(frozen=True)
class ChannelAuthority:
    case: Chain4Case
    modes: np.ndarray
    coefficients: np.ndarray

    @property
    def log_masses(self):
        logc = np.full(len(self.coefficients), -np.inf)
        np.log(self.coefficients, out=logc, where=self.coefficients > 0)
        return logc - self.case.variance * np.square(self.modes).sum(axis=1) / 2

    @property
    def probabilities(self):
        values = self.log_masses
        return np.exp(values - logsumexp(values))

    def thermodynamics(self):
        q = self.modes.reshape(-1, self.case.n_t, 4)[:, 0].sum(axis=1)
        p = self.probabilities
        return {
            "log_partition": float(logsumexp(self.log_masses) + 4*self.case.beta*self.case.mu),
            "density": float(p @ (1 + q / 4)),
            "double_occupancy": float(p @ (self.modes == 1).mean(axis=1)),
        }


def _exterior_propagator(case: Chain4Case, sign: int):
    """All exterior powers in a bounded 16-state space, avoiding unstable minors.

    Entries equal ordered submatrix minors of exp(sign*dt*(kappa*h+mu)).
    This auxiliary occupation construction is separate from the physical
    eight-orbital Hamiltonian above.
    """
    generator = np.zeros((16, 16))
    occupations = np.array([[(s >> x) & 1 for x in range(4)] for s in range(16)])
    for state in range(16):
        generator[state, state] = case.mu * occupations[state].sum()
        for src in range(4):
            for dest in ((src + 1) % 4, (src - 1) % 4):
                if occupations[state, src] == 0 or occupations[state, dest] == 1:
                    continue
                lo, hi = sorted((src, dest))
                parity = occupations[state, lo+1:hi].sum()
                target = state ^ (1 << src) ^ (1 << dest)
                generator[target, state] = case.kappa * (-1.0)**parity
    matrix = expm(sign * case.beta / case.n_t * generator)
    return (matrix + matrix.T) / 2, occupations


def channel_authority(case: Chain4Case) -> ChannelAuthority:
    """Enumerate ONLY this bounded authority, retaining every ternary label."""
    if case.n_t > 2:
        raise ValueError("Nt=3/4 requires the bounded streaming multislice authority")
    a, occup = _exterior_propagator(case, 1)
    b, _ = _exterior_propagator(case, -1)
    modes = np.array(list(product((-1, 0, 1), repeat=case.dimension)), dtype=np.int8)
    coefficients = np.zeros(len(modes))
    powers = 3 ** np.arange(case.dimension-1, -1, -1)
    if case.n_t == 1:
        for s in range(16):
            for t in range(16):
                mode = occup[s] - occup[t]
                coefficients[(mode + 1) @ powers] += a[s, s] * b[t, t]
    else:
        histories = [(s0, s1) for s0 in range(16) for s1 in range(16)
                     if s0.bit_count() == s1.bit_count()]
        for s0, s1 in histories:
            for t0, t1 in histories:
                mode = np.concatenate((occup[s0]-occup[t0], occup[s1]-occup[t1]))
                coefficients[(mode + 1) @ powers] += a[s1, s0]**2 * b[t1, t0]**2
    if not np.isfinite(coefficients).all() or np.any(coefficients < 0):
        raise ArithmeticError("invalid nonnegative channel coefficients")
    modes.setflags(write=False)
    coefficients.setflags(write=False)
    return ChannelAuthority(case, modes, coefficients)

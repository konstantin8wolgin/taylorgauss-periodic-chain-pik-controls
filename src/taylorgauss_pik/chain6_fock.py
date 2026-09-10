"""Independent six-site spinful authorities, with no auxiliary-field inputs.

The physical Hamiltonian uses interleaved up/down orbitals and the charge
interaction U/2 sum_x (n_up+n_down-1)^2. Every (N_up,N_down) sector is retained.
The largest dense block has 400 states; no 4096-by-4096 matrix is allocated.
"""
import math
from numbers import Real

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp


def _spinful_block(states, U, mu, kappa):
    index = {state: position for position, state in enumerate(states)}
    occupation = np.array([[(state >> orbital) & 1 for orbital in range(12)]
                           for state in states])
    charge = occupation[:, 0::2] + occupation[:, 1::2] - 1
    potential = U/2*np.square(charge).sum(axis=1)-mu*occupation.sum(axis=1)
    double = (occupation[:, 0::2]*occupation[:, 1::2]).mean(axis=1)
    kinetic = np.zeros((len(states), len(states)))
    for column, state in enumerate(states):
        for site in range(6):
            neighbor = (site+1) % 6
            for destination, source in ((site, neighbor), (neighbor, site)):
                for spin in (0, 1):
                    a, b = 2*destination+spin, 2*source+spin
                    if not ((state >> b) & 1) or ((state >> a) & 1):
                        continue
                    removed = state ^ (1 << b)
                    parity = ((state & ((1 << b)-1)).bit_count()
                              + (removed & ((1 << a)-1)).bit_count())
                    target = removed | (1 << a)
                    kinetic[index[target], column] -= kappa*(-1.)**parity
    return kinetic, potential, double


def _combine(blocks):
    values = np.asarray(blocks)
    log_z = logsumexp(values[:, 0])
    weights = np.exp(values[:, 0]-log_z)
    return dict(log_partition=float(log_z), density=float(weights @ values[:, 1]),
                double_occupancy=float(weights @ values[:, 2]))


def spinful_authority(*, beta=1., U=4., mu=1., kappa=1., n_sites=6, n_t=1):
    """Return the one-slice symmetric-product trace and Hamiltonian ED trace.

    The latter quantifies discretization error; it is not the one-slice target.
    Exponentials use energy shifts and sector log-sums for stable normalization.
    Eigen residuals are floating-point checks, not interval error certificates.
    """
    if type(n_sites) is not int or n_sites != 6:
        raise ValueError('bounded spinful authority requires n_sites=6')
    if type(n_t) is not int or n_t != 1:
        raise ValueError('bounded spinful authority requires n_t=1')
    for name, value, lower, upper in (
        ('beta', beta, 0., 4.), ('U', U, 0., 8.),
        ('mu', mu, -4., 4.), ('kappa', kappa, 0., 1.),
    ):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or not lower <= value <= upper
                or (name == 'beta' and value == 0)):
            raise ValueError(f'{name} outside bounded numerical envelope')
    sectors = {(up, down): [] for up in range(7) for down in range(7)}
    for state in range(4096):
        sectors[((state & 0x555).bit_count(), (state & 0xAAA).bit_count())].append(state)
    finite, hamiltonian, insertions = [], [], []
    eigen_residual = orthogonality = 0.
    for (up, down), states in sectors.items():
        kinetic, potential, double = _spinful_block(states, U, mu, kappa)
        energies, vectors = eigh(kinetic)
        scaled_diagonal = np.square(vectors) @ np.exp(-beta*(energies-energies[0]))
        log_diagonal = -beta*(potential+energies[0])+np.log(scaled_diagonal)
        log_z = logsumexp(log_diagonal)
        diagonal_probabilities = np.exp(log_diagonal-log_z)
        finite.append((log_z, (up+down)/6, diagonal_probabilities @ double))
        # Direct diagonal physical-occupation insertions, independent of any
        # auxiliary channel or PIK sampling formula. k counts doubly occupied
        # sites and s=N_e-6 is constant inside this number sector.
        s, k = up+down-6, 6*double
        mean_k = diagonal_probabilities @ k
        insertions.append((s, s*s, mean_k, diagonal_probabilities @ np.square(k),
                           s*mean_k, float(up == 6 and down == 6)))

        physical = kinetic.copy()
        physical[np.diag_indices(len(states))] += potential
        energies, vectors = eigh(physical)
        scaled_weights = np.exp(-beta*(energies-energies[0]))
        scaled_z = scaled_weights.sum()
        probabilities = np.square(vectors) @ (scaled_weights/scaled_z)
        hamiltonian.append((-beta*energies[0]+np.log(scaled_z), (up+down)/6,
                            probabilities @ double))
        scale = max(1., np.linalg.norm(physical, ord=np.inf))
        eigen_residual = max(eigen_residual, float(np.max(np.abs(
            physical @ vectors-vectors*energies[None, :]))/scale))
        orthogonality = max(orthogonality, float(np.max(np.abs(
            vectors.T @ vectors-np.eye(len(states))))))
    log_sector_weights = np.asarray(finite)[:, 0]
    sector_probabilities = np.exp(log_sector_weights-logsumexp(log_sector_weights))
    insertion_values = sector_probabilities @ np.asarray(insertions)
    insertion_names = ('charge', 'charge_squared', 'double_count',
                       'double_count_squared', 'charge_double_count',
                       'all_double_probability')
    return dict(finite_nt=_combine(finite), hamiltonian=_combine(hamiltonian),
                finite_insertions=dict(zip(insertion_names, map(float, insertion_values))),
                metadata=dict(state_count=4096, sector_count=49, max_block_dimension=400,
                              hamiltonian_scaled_eigen_residual=eigen_residual,
                              hamiltonian_orthogonality_residual=orthogonality,
                              normalization='physical grand canonical trace',
                              finite_nt_target='symmetric one-slice Trotter trace'))

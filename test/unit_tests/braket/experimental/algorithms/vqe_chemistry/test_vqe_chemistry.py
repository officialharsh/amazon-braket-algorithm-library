# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast, local-only unit tests for the VQE chemistry algorithm.

These run on the free local statevector simulator (no AWS credentials, no billable tasks),
so they are safe for CI and for `Cells > Run All`. H2 in STO-3G is the smallest meaningful
case and converges in well under a second.
"""

import numpy as np

from braket.experimental.algorithms.vqe_chemistry import (
    build_molecular_hamiltonian,
    compute_binding_energy,
    exact_ground_state_energy,
    get_vqe_chemistry_results,
    run_vqe_chemistry,
)

H2_GEOMETRY = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]  # Bohr
H2_FCI_HA = -1.137275943839282  # known reference for STO-3G CAS(2,2)


def test_build_molecular_hamiltonian_h2():
    data = build_molecular_hamiltonian(["H", "H"], H2_GEOMETRY, 2, 2)
    assert data["n_qubits"] == 4
    # 2 singles + 1 double for H2 CAS(2,2)
    assert data["n_parameters"] == 3
    assert data["n_pauli_terms"] > 0
    assert list(data["hf_state"]) == [1, 1, 0, 0]


def test_exact_ground_state_energy_h2():
    data = build_molecular_hamiltonian(["H", "H"], H2_GEOMETRY, 2, 2)
    e_fci = exact_ground_state_energy(data["hamiltonian"], data["n_qubits"])
    assert np.isclose(e_fci, H2_FCI_HA, atol=1e-6)


def test_run_vqe_chemistry_local_reaches_chemical_accuracy():
    result = run_vqe_chemistry(
        ["H", "H"], H2_GEOMETRY, 2, 2, max_iterations=60, verbose=False
    )
    assert result.n_qubits == 4
    assert np.isclose(result.fci_energy, H2_FCI_HA, atol=1e-6)
    assert result.within_chemical_accuracy
    assert abs(result.error_mha) < 1.6


def test_get_vqe_chemistry_results_schema():
    result = run_vqe_chemistry(
        ["H", "H"], H2_GEOMETRY, 2, 2, max_iterations=40, verbose=False
    )
    summary = get_vqe_chemistry_results(result)
    for key in (
        "molecule", "device", "n_qubits", "n_pauli_terms", "n_parameters",
        "fci_energy_ha", "vqe_energy_ha", "error_mha", "within_chemical_accuracy",
    ):
        assert key in summary
    assert summary["device"] == "local"


def test_compute_binding_energy_size_consistency_h2_dimer():
    # Two well-separated H2 molecules: binding energy should be ~0 (size consistency).
    monomer = run_vqe_chemistry(
        ["H", "H"], H2_GEOMETRY, 2, 2, max_iterations=60, verbose=False
    )
    # Use the same monomer twice as the "fragments"; a true non-interacting dimer energy
    # equals 2 * monomer, so binding relative to two monomers is ~0 by construction.
    binding = compute_binding_energy(
        complex_result=_doubled(monomer), fragment_results=[monomer, monomer]
    )
    assert abs(binding["binding_energy_kcal_per_mol"]) < 1e-6


def _doubled(result):
    """Helper: a synthetic 'complex' whose energy is exactly twice the monomer (FCI and VQE)."""
    import copy

    dimer = copy.deepcopy(result)
    dimer.fci_energy = 2 * result.fci_energy
    dimer.vqe_energy = 2 * result.vqe_energy
    return dimer

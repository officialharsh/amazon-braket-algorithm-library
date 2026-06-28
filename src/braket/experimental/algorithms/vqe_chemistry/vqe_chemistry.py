# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Variational Quantum Eigensolver (VQE) for molecular ground-state and binding energies.

This module implements an end-to-end VQE workflow for quantum chemistry on Amazon Braket,
following the Algorithm Library convention of a circuit-definition function, a
``run_vqe_chemistry`` driver, and a ``get_vqe_chemistry_results`` helper.

The workflow:
  1. Build a molecular electronic Hamiltonian from atomic coordinates (PennyLane qchem,
     Jordan-Wigner mapping, optional active space).
  2. Prepare a particle-conserving AllSinglesDoubles ansatz on the Hartree-Fock reference.
  3. Optimize the parameters on the free local statevector simulator (deterministic), then
     optionally evaluate the converged energy on an Amazon Braket device (SV1 simulator or a
     QPU such as IonQ Forte) via the PennyLane-Braket plugin, which runs on the Amazon Braket
     SDK.
  4. Combine fragment energies into a binding (reaction) energy.

All quantum execution goes through the Amazon Braket SDK (directly for the device ARN, and via
the amazon-braket-pennylane-plugin for the PennyLane device). Dependencies are open source
(PennyLane, the Braket plugin, SciPy, NumPy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

HARTREE_TO_KCAL = 627.509469
CHEMICAL_ACCURACY_HA = 1.6e-3  # 1 kcal/mol in Hartree

SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"


@dataclass
class VQEChemistryResult:
    """Container for a single VQE run."""

    symbols: list
    n_qubits: int
    n_pauli_terms: int
    n_parameters: int
    fci_energy: float
    vqe_energy: float
    optimal_params: np.ndarray
    energy_trajectory: list = field(default_factory=list)
    device: str = "local"
    shots: Optional[int] = None

    @property
    def error_mha(self) -> float:
        """VQE error against the exact (FCI) reference, in milliHartree."""
        return (self.vqe_energy - self.fci_energy) * 1000.0

    @property
    def within_chemical_accuracy(self) -> bool:
        return abs(self.vqe_energy - self.fci_energy) < CHEMICAL_ACCURACY_HA


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------
def build_molecular_hamiltonian(
    symbols: list,
    geometry,
    active_electrons: int,
    active_orbitals: int,
    basis_name: str = "sto-3g",
    charge: int = 0,
    mapping: str = "jordan_wigner",
):
    """Build the electronic Hamiltonian and ansatz wiring for a molecule.

    Args:
        symbols: Atom labels, e.g. ``["O", "H", "H"]``.
        geometry: (n_atoms, 3) nuclear coordinates in atomic units (Bohr).
        active_electrons: Number of active electrons (CAS).
        active_orbitals: Number of active spatial orbitals (CAS); qubits = 2 * active_orbitals.
        basis_name: Atomic basis set (default minimal STO-3G).
        charge: Net molecular charge.
        mapping: Fermion-to-qubit mapping (default Jordan-Wigner).

    Returns:
        dict with keys: ``hamiltonian``, ``n_qubits``, ``hf_state``, ``singles``, ``doubles``,
        ``n_parameters``, ``n_pauli_terms``.
    """
    geometry = pnp.array(geometry, requires_grad=False)
    molecule = qml.qchem.Molecule(symbols, geometry, charge=charge, basis_name=basis_name)
    hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
        molecule,
        mapping=mapping,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    singles, doubles = qml.qchem.excitations(active_electrons, n_qubits)
    hf_state = qml.qchem.hf_state(active_electrons, n_qubits)
    n_parameters = len(singles) + len(doubles)
    n_pauli_terms = len(hamiltonian.terms()[0])

    return {
        "hamiltonian": hamiltonian,
        "n_qubits": n_qubits,
        "hf_state": hf_state,
        "singles": singles,
        "doubles": doubles,
        "n_parameters": n_parameters,
        "n_pauli_terms": n_pauli_terms,
    }


# ---------------------------------------------------------------------------
# Circuit definition (Algorithm Library convention: a circuit-definition function)
# ---------------------------------------------------------------------------
def vqe_ansatz(params, n_qubits, hf_state, singles, doubles) -> None:
    """Particle-conserving AllSinglesDoubles ansatz on the Hartree-Fock reference.

    This prepares |psi(params)> from the HF state using Givens-rotation single and double
    excitations. It is spin- and particle-number conserving, so it stays inside the correct
    electronic sector of Hilbert space.
    """
    qml.AllSinglesDoubles(
        params,
        wires=range(n_qubits),
        hf_state=hf_state,
        singles=singles,
        doubles=doubles,
    )


def _energy_qnode(device, hamiltonian, ham_data, diff_method):
    """Build a QNode that returns <psi(params)|H|psi(params)> on the given device."""

    @qml.qnode(device, diff_method=diff_method)
    def energy(params):
        vqe_ansatz(
            params,
            ham_data["n_qubits"],
            ham_data["hf_state"],
            ham_data["singles"],
            ham_data["doubles"],
        )
        return qml.expval(hamiltonian)

    return energy


# ---------------------------------------------------------------------------
# Exact reference
# ---------------------------------------------------------------------------
def exact_ground_state_energy(hamiltonian, n_qubits: int) -> float:
    """Exact ground-state energy in the active space (FCI), by direct diagonalization."""
    matrix = qml.matrix(hamiltonian, wire_order=range(n_qubits))
    eigvals = np.linalg.eigvalsh(matrix)
    return float(np.min(eigvals.real))


# ---------------------------------------------------------------------------
# Driver (Algorithm Library convention: run_<name>)
# ---------------------------------------------------------------------------
def run_vqe_chemistry(
    symbols: list,
    geometry,
    active_electrons: int,
    active_orbitals: int,
    basis_name: str = "sto-3g",
    charge: int = 0,
    device_arn: Optional[str] = None,
    shots: Optional[int] = None,
    max_iterations: int = 80,
    stepsize: float = 0.4,
    verbose: bool = True,
) -> VQEChemistryResult:
    """Run the full VQE workflow for one molecule and return a :class:`VQEChemistryResult`.

    The optimization is always performed on the free local statevector simulator
    (deterministic, exact gradients via adjoint). If ``device_arn`` is provided, the converged
    energy is then evaluated once on that Amazon Braket device (billable for SV1 and QPUs).

    Args:
        symbols, geometry, active_electrons, active_orbitals, basis_name, charge: passed to
            :func:`build_molecular_hamiltonian`.
        device_arn: Amazon Braket device ARN. ``None`` keeps everything on the local simulator.
            Use :data:`SV1_ARN` for the on-demand simulator, or a QPU ARN for hardware.
        shots: Shot count for device evaluation. ``None``/``0`` uses analytic mode (SV1 only;
            QPUs require a positive shot count).
        max_iterations: Local optimizer steps.
        stepsize: Gradient-descent step size for the local optimization.
        verbose: Print progress.

    Returns:
        VQEChemistryResult with the FCI reference, the VQE energy, the optimal parameters, and
        the optimization trajectory.
    """
    ham_data = build_molecular_hamiltonian(
        symbols, geometry, active_electrons, active_orbitals, basis_name, charge
    )
    hamiltonian = ham_data["hamiltonian"]
    n_qubits = ham_data["n_qubits"]

    e_fci = exact_ground_state_energy(hamiltonian, n_qubits)

    if verbose:
        print(f"Molecule              : {symbols}")
        print(f"Qubits                : {n_qubits}")
        print(f"Pauli terms           : {ham_data['n_pauli_terms']}")
        print(f"Parameters            : {ham_data['n_parameters']} "
              f"({len(ham_data['singles'])} singles, {len(ham_data['doubles'])} doubles)")
        print(f"Exact FCI energy      : {e_fci:.8f} Ha")

    # --- local optimization (free, exact) ---
    local_dev = qml.device("default.qubit", wires=n_qubits)
    local_energy = _energy_qnode(local_dev, hamiltonian, ham_data, "adjoint")

    params = pnp.zeros(ham_data["n_parameters"], requires_grad=True)
    optimizer = qml.GradientDescentOptimizer(stepsize=stepsize)
    trajectory = []
    for step in range(max_iterations):
        params, energy = optimizer.step_and_cost(local_energy, params)
        trajectory.append(float(energy))
        if verbose and step % 20 == 0:
            print(f"  step {step:3d}: E = {energy:.8f} Ha")

    vqe_energy = trajectory[-1]
    device_label = "local"

    # --- optional device evaluation (billable) ---
    if device_arn is not None:
        if verbose:
            print(f"\nEvaluating converged energy on {device_arn} (billable)...")
        braket_dev = qml.device(
            "braket.aws.qubit",
            device_arn=device_arn,
            wires=n_qubits,
            shots=shots if shots else 0,
        )
        device_energy = _energy_qnode(braket_dev, hamiltonian, ham_data, None)
        vqe_energy = float(device_energy(pnp.array(params, requires_grad=False)))
        device_label = device_arn

    result = VQEChemistryResult(
        symbols=list(symbols),
        n_qubits=n_qubits,
        n_pauli_terms=ham_data["n_pauli_terms"],
        n_parameters=ham_data["n_parameters"],
        fci_energy=e_fci,
        vqe_energy=vqe_energy,
        optimal_params=np.array(params),
        energy_trajectory=trajectory,
        device=device_label,
        shots=shots,
    )

    if verbose:
        print(f"VQE energy            : {result.vqe_energy:.8f} Ha")
        print(f"Error vs FCI          : {result.error_mha:.4f} mHa "
              f"(chemical accuracy = 1.6 mHa)")

    return result


# ---------------------------------------------------------------------------
# Results helper (Algorithm Library convention: get_<name>_results)
# ---------------------------------------------------------------------------
def get_vqe_chemistry_results(result: VQEChemistryResult) -> dict:
    """Return a plain-dict summary of a :class:`VQEChemistryResult` for display or logging."""
    return {
        "molecule": result.symbols,
        "device": result.device,
        "shots": result.shots,
        "n_qubits": result.n_qubits,
        "n_pauli_terms": result.n_pauli_terms,
        "n_parameters": result.n_parameters,
        "fci_energy_ha": result.fci_energy,
        "vqe_energy_ha": result.vqe_energy,
        "error_mha": result.error_mha,
        "within_chemical_accuracy": result.within_chemical_accuracy,
    }


# ---------------------------------------------------------------------------
# Binding / reaction energy from fragment runs
# ---------------------------------------------------------------------------
def compute_binding_energy(
    complex_result: VQEChemistryResult,
    fragment_results: list,
    use_fci: bool = False,
) -> dict:
    """Binding (reaction) energy: E(complex) - sum_i E(fragment_i).

    Args:
        complex_result: VQE result for the bound complex / product.
        fragment_results: VQE results for the isolated fragments / reactants.
        use_fci: If True, use the exact FCI energies instead of the VQE energies (useful for a
            VQE-vs-FCI consistency check).

    Returns:
        dict with the binding energy in Hartree and kcal/mol.
    """
    def pick(res: VQEChemistryResult) -> float:
        return res.fci_energy if use_fci else res.vqe_energy

    e_bind_ha = pick(complex_result) - sum(pick(f) for f in fragment_results)
    return {
        "binding_energy_ha": e_bind_ha,
        "binding_energy_kcal_per_mol": e_bind_ha * HARTREE_TO_KCAL,
        "source": "FCI" if use_fci else "VQE",
    }


# ---------------------------------------------------------------------------
# H2 hardware helpers: non-blocking submit + fetch-by-ARN
#
# A QPU task can queue for a long time. Blocking on the result inside a Jupyter
# kernel hangs (and tangles its async loop), so for hardware we decouple submission
# from retrieval: submit returns task ARNs immediately; a later call fetches the
# completed results by ARN and reduces them to an energy. These helpers use the
# Amazon Braket SDK directly (not a blocking PennyLane QNode) so we control the
# task objects. They target the minimal 4-qubit H2 ansatz used in the notebook.
# ---------------------------------------------------------------------------
def _h2_hamiltonian(bond_bohr: float = 1.4):
    symbols = ["H", "H"]
    geometry = pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_bohr]])
    mol = qml.qchem.Molecule(symbols, geometry, basis_name="sto-3g")
    H, n = qml.qchem.molecular_hamiltonian(
        mol, mapping="jordan_wigner", active_electrons=2, active_orbitals=2
    )
    return H, n


def _pauli_map(obs) -> dict:
    """Map a single-Pauli-word observable to {wire: 'X'|'Y'|'Z'}; {} for identity."""
    for pauli_word in obs.pauli_rep:
        return {int(w): p for w, p in dict(pauli_word).items()}
    return {}


def _h2_ansatz_ops(theta: float, n_qubits: int = 4):
    """Minimal H2 ansatz as PennyLane ops decomposed to {PauliX, Hadamard, RY, CNOT}."""
    ops = [qml.PauliX(0), qml.PauliX(1)]  # Hartree-Fock |1100>
    ops += qml.DoubleExcitation(theta, wires=range(n_qubits)).decomposition()
    return ops


def _braket_circuit(ansatz_ops, basis: dict):
    """Translate ansatz ops to a Braket Circuit and append measurement-basis rotations.

    basis maps a qubit to the Pauli ('X'/'Y'/'Z') it is measured in for this group;
    X -> H, Y -> S-dagger then H, Z/absent -> no rotation. The circuit is then measured
    in the computational basis (raw shots).
    """
    from braket.circuits import Circuit

    c = Circuit()
    for op in ansatz_ops:
        name = op.name
        w = [int(x) for x in op.wires]
        if name == "PauliX":
            c.x(w[0])
        elif name == "Hadamard":
            c.h(w[0])
        elif name == "RY":
            c.ry(w[0], float(op.parameters[0]))
        elif name == "CNOT":
            c.cnot(w[0], w[1])
        else:
            raise ValueError(f"unexpected gate in H2 ansatz: {name}")
    for q, pauli in basis.items():
        if pauli == "X":
            c.h(q)
        elif pauli == "Y":
            c.si(q)
            c.h(q)
    return c


def prepare_h2_hardware(theta: float, bond_bohr: float = 1.4) -> dict:
    """Build per-(QWC group) Braket circuits for the minimal H2 ansatz at angle theta.

    Returns a dict with:
      - circuits: list of Braket Circuit (one per qubit-wise-commuting group)
      - groups:   list (aligned with circuits) of [(pauli_map, coeff), ...] terms
      - identity: identity-term coefficient (added directly to the energy)
      - fci:      exact active-space energy (for comparison)
      - n_qubits, theta, bond_bohr
    The structure is deterministic in theta, so fetch can rebuild it without storing it.
    """
    H, n = _h2_hamiltonian(bond_bohr)
    coeffs, ops = H.terms()
    e_fci = exact_ground_state_energy(H, n)
    ansatz_ops = _h2_ansatz_ops(theta, n)

    identity, obs, obs_coeffs = 0.0, [], []
    for c, o in zip(coeffs, ops):
        pmap = _pauli_map(o)
        if not pmap:
            identity += float(c)
        else:
            obs.append(o)
            obs_coeffs.append(float(c))

    grouped_ops, grouped_coeffs = qml.pauli.group_observables(
        obs, obs_coeffs, grouping_type="qwc", method="rlf"
    )

    circuits, groups = [], []
    for g_ops, g_coeffs in zip(grouped_ops, grouped_coeffs):
        basis, terms = {}, []
        for o, c in zip(g_ops, g_coeffs):
            pmap = _pauli_map(o)
            basis.update(pmap)
            terms.append((pmap, float(c)))
        circuits.append(_braket_circuit(ansatz_ops, basis))
        groups.append(terms)

    return {
        "circuits": circuits,
        "groups": groups,
        "identity": identity,
        "fci": e_fci,
        "n_qubits": n,
        "theta": float(theta),
        "bond_bohr": bond_bohr,
    }


def energy_from_measurements(prep: dict, measurements_per_group: list) -> float:
    """Reduce raw measurement arrays (one (shots, n_qubits) array per group) to energy.

    For a Pauli word P with support S, <P> = mean over shots of prod_{q in S} (-1)^bit[q].
    """
    energy = prep["identity"]
    for terms, meas in zip(prep["groups"], measurements_per_group):
        meas = np.asarray(meas)
        for pmap, coeff in terms:
            qubits = sorted(pmap.keys())
            signs = np.prod(1 - 2 * meas[:, qubits], axis=1)  # (-1)^bit
            energy += coeff * float(np.mean(signs))
    return energy

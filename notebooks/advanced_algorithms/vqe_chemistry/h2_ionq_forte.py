"""
H2 (4-qubit) VQE on Amazon Braket -- first real-hardware run on IonQ Forte.

This is the smallest meaningful chemistry VQE: H2 in STO-3G is 2 electrons in
2 spatial orbitals -> 4 spin-orbitals -> 4 qubits under Jordan-Wigner. The ground
state is captured by a SINGLE DoubleExcitation gate (1 parameter), which is the
shallowest chemically-correct ansatz and therefore the most NISQ-hardware-friendly.

Workflow (cost-aware, two phases):
  PHASE 1 (free, always runs): build the Hamiltonian, compute the exact FCI
          reference by diagonalization, and optimize the VQE parameter locally on
          the analytic statevector simulator. This is deterministic and costs nothing.
  PHASE 2 (billable, only with --submit): evaluate the optimized energy on the real
          IonQ Forte QPU via Amazon Braket, using finite shots. Optionally refine
          with a few COBYLA steps directly on hardware (--hw-optimize).

Nothing is sent to hardware unless you pass --submit. Without it the script does a
full dry run and prints exactly what it WOULD submit.

Run (dry run / local only):
    python h2_ionq_forte.py
Run (REAL IonQ Forte, billable):
    python h2_ionq_forte.py --submit --shots 2000
"""

import argparse
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

# IonQ Forte on Amazon Braket (verify with: aws braket search-devices --region us-east-1).
# As of the last device check, Forte-1 was OFFLINE and Forte Enterprise 1 was ONLINE,
# so the default points at the online device. Override with --device-arn if status changes.
IONQ_FORTE1_ARN = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"            # OFFLINE (last check)
IONQ_FORTE_ENTERPRISE_ARN = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1"  # ONLINE
IONQ_FORTE_ARN = IONQ_FORTE_ENTERPRISE_ARN
SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
HARTREE2KCAL = 627.509


def build_h2(bond_bohr=1.4):
    """H2 at the given bond length (Bohr). Returns (Hamiltonian, n_qubits, hf_state)."""
    symbols = ["H", "H"]
    geometry = pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_bohr]])
    mol = qml.qchem.Molecule(symbols, geometry, basis_name="sto-3g")
    H, qubits = qml.qchem.molecular_hamiltonian(
        mol, mapping="jordan_wigner", active_electrons=2, active_orbitals=2)
    hf = qml.qchem.hf_state(2, 4)  # [1, 1, 0, 0]
    return H, qubits, hf


def ansatz(theta, wires, hf):
    """Minimal H2 ansatz: HF reference + one DoubleExcitation (1 parameter)."""
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(theta, wires=wires)


def fci_energy(H, qubits):
    """Exact ground-state energy in this basis (matrix diagonalization)."""
    return float(np.min(np.linalg.eigvalsh(qml.matrix(H, wire_order=range(qubits)))))


def optimize_local(H, qubits, hf):
    """Deterministic, free local optimization (analytic statevector). Returns theta, energy."""
    dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def energy(theta):
        ansatz(theta, range(qubits), hf)
        return qml.expval(H)

    res = minimize(lambda x: float(energy(x[0])), x0=[0.0], method="COBYLA",
                   options={"maxiter": 200, "rhobeg": 0.3})
    return float(res.x[0]), float(res.fun)


def make_ionq_qnode(H, qubits, hf, device_arn, shots):
    """Build a QNode bound to a real QPU (finite shots, no analytic gradient)."""
    dev = qml.device("braket.aws.qubit", device_arn=device_arn, wires=qubits, shots=shots)

    @qml.qnode(dev, diff_method=None)  # gradient-free; hardware cannot do adjoint
    def energy(theta):
        ansatz(theta, range(qubits), hf)
        return qml.expval(H)

    return energy


def run_on_ionq(H, qubits, hf, theta_opt, device_arn, shots, hw_optimize):
    """PHASE 2: evaluate (and optionally refine) the energy on the real QPU."""
    energy = make_ionq_qnode(H, qubits, hf, device_arn, shots)

    if hw_optimize:
        # Gradient-free refinement directly on hardware (COBYLA). Each step submits tasks.
        print("  [ionq] refining with COBYLA on hardware (this submits many tasks)...")
        res = minimize(lambda x: float(energy(x[0])), x0=[theta_opt], method="COBYLA",
                       options={"maxiter": 20, "rhobeg": 0.1})
        return float(res.x[0]), float(res.fun)
    else:
        # Cheapest meaningful hardware result: one energy evaluation at the optimal angle.
        print("  [ionq] evaluating energy once at the locally-optimized angle...")
        return float(theta_opt), float(energy(theta_opt))


def main():
    p = argparse.ArgumentParser(description="H2 VQE on IonQ Forte via Amazon Braket")
    p.add_argument("--submit", action="store_true",
                   help="actually submit to the real QPU (BILLABLE). Omit for a dry run.")
    p.add_argument("--shots", type=int, default=2000, help="shots per circuit on hardware")
    p.add_argument("--device-arn", default=IONQ_FORTE_ARN, help="QPU device ARN")
    p.add_argument("--bond", type=float, default=1.4, help="H-H bond length in Bohr")
    p.add_argument("--hw-optimize", action="store_true",
                   help="run COBYLA refinement on hardware (many more tasks)")
    args = p.parse_args()

    print("=" * 70)
    print("H2 VQE  (4 qubits, STO-3G)  ->  IonQ Forte via Amazon Braket")
    print("=" * 70)

    # ---- PHASE 1: free, local, deterministic ----
    H, qubits, hf = build_h2(args.bond)
    n_terms = len(H.terms()[0])
    e_fci = fci_energy(H, qubits)
    theta_opt, e_local = optimize_local(H, qubits, hf)

    print(f"Qubits                : {qubits}")
    print(f"Pauli terms           : {n_terms}")
    print(f"HF reference state    : {[int(x) for x in hf]}")
    print(f"Exact FCI energy      : {e_fci:.8f} Ha")
    print(f"Local VQE energy      : {e_local:.8f} Ha")
    print(f"Local error vs FCI    : {(e_local - e_fci) * 1000:.4f} mHa  (chem acc = 1.6 mHa)")
    print(f"Optimized angle theta : {theta_opt:.6f} rad")

    if not args.submit:
        print("\n--- DRY RUN (no hardware tasks submitted) ---")
        print(f"Would submit to       : {args.device_arn}")
        print(f"Shots per circuit     : {args.shots}")
        print(f"Mode                  : {'COBYLA refine on hardware' if args.hw_optimize else 'single energy evaluation'}")
        print("Re-run with --submit to execute on the real QPU (billable).")
        return

    # ---- PHASE 2: real QPU (billable) ----
    print(f"\n--- SUBMITTING TO REAL QPU: {args.device_arn} (BILLABLE) ---")
    theta_hw, e_hw = run_on_ionq(H, qubits, hf, theta_opt, args.device_arn,
                                 args.shots, args.hw_optimize)
    print(f"IonQ Forte VQE energy : {e_hw:.8f} Ha  (theta = {theta_hw:.6f})")
    print(f"Hardware error vs FCI : {(e_hw - e_fci) * 1000:.4f} mHa")
    print("(Expect a positive error from device noise; layer error mitigation to reduce it.)")

    with open("h2_ionq_result.txt", "w") as f:
        f.write(f"device,{args.device_arn}\n")
        f.write(f"shots,{args.shots}\n")
        f.write(f"E_FCI_Ha,{e_fci:.8f}\n")
        f.write(f"E_local_Ha,{e_local:.8f}\n")
        f.write(f"E_ionq_Ha,{e_hw:.8f}\n")
        f.write(f"error_vs_FCI_mHa,{(e_hw - e_fci) * 1000:.4f}\n")
    print("Wrote h2_ionq_result.txt")


if __name__ == "__main__":
    main()

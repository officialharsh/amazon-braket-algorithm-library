"""
H2 (4-qubit) VQE on IonQ Forte WITH error mitigation -- the follow-up to
h2_ionq_forte.py, which gave a raw (unmitigated) energy ~50 mHa above FCI.

Mitigation used: IonQ's native debiasing + sharpening (braket.error_mitigation.Debias).
It runs each circuit across multiple symmetric qubit configurations and aggregates,
suppressing coherent/systematic device errors. It is enabled with a single device
parameter and improves every measurement group (unlike post-selection, which only
cleanly applies to the diagonal terms).

Same two-phase, cost-aware structure as h2_ionq_forte.py:
  PHASE 1 (free): exact FCI + local VQE angle (deterministic, no cost).
  PHASE 2 (billable, only with --submit): evaluate the energy on IonQ Forte Enterprise
          WITH debiasing, then compare to FCI and to the unmitigated result.

Run (dry run / local only):
    python h2_ionq_mitigated.py
Run (REAL IonQ Forte, billable, mitigated):
    python h2_ionq_mitigated.py --submit --shots 2500
"""

import argparse
import pennylane as qml
from braket.error_mitigation import Debias

# Reuse the validated building blocks from the first script (importing does NOT run its main()).
from h2_ionq_forte import (
    build_h2, ansatz, fci_energy, optimize_local, IONQ_FORTE_ENTERPRISE_ARN,
)

UNMITIGATED_HW = -1.08728253  # the raw IonQ result from h2_ionq_forte.py, verified from S3


def make_mitigated_qnode(H, qubits, hf, device_arn, shots):
    """QNode on a real QPU with IonQ debiasing + sharpening enabled."""
    dev = qml.device(
        "braket.aws.qubit", device_arn=device_arn, wires=qubits, shots=shots,
        device_parameters={"errorMitigation": Debias()},  # IonQ debiasing + sharpening
    )

    @qml.qnode(dev, diff_method=None)
    def energy(theta):
        ansatz(theta, range(qubits), hf)
        return qml.expval(H)

    return energy


def main():
    p = argparse.ArgumentParser(description="H2 VQE on IonQ Forte WITH error mitigation")
    p.add_argument("--submit", action="store_true",
                   help="actually submit to the real QPU (BILLABLE). Omit for a dry run.")
    p.add_argument("--shots", type=int, default=2500,
                   help="shots per circuit (IonQ debiasing prefers a multiple of its variants)")
    p.add_argument("--device-arn", default=IONQ_FORTE_ENTERPRISE_ARN, help="QPU device ARN")
    p.add_argument("--bond", type=float, default=1.4, help="H-H bond length in Bohr")
    args = p.parse_args()

    print("=" * 70)
    print("H2 VQE  (4 qubits)  ->  IonQ Forte + DEBIASING (error mitigation)")
    print("=" * 70)

    # ---- PHASE 1: free, local ----
    H, qubits, hf = build_h2(args.bond)
    e_fci = fci_energy(H, qubits)
    theta_opt, e_local = optimize_local(H, qubits, hf)
    print(f"Qubits / terms        : {qubits} / {len(H.terms()[0])}")
    print(f"Exact FCI energy      : {e_fci:.8f} Ha")
    print(f"Local VQE energy      : {e_local:.8f} Ha  ({(e_local - e_fci)*1000:.4f} mHa)")
    print(f"Optimized angle theta : {theta_opt:.6f} rad")
    print(f"Unmitigated hardware  : {UNMITIGATED_HW:.8f} Ha  ({(UNMITIGATED_HW - e_fci)*1000:.2f} mHa above FCI)")

    if not args.submit:
        print("\n--- DRY RUN (no hardware tasks submitted) ---")
        print(f"Would submit to       : {args.device_arn}")
        print(f"Shots per circuit     : {args.shots}")
        print(f"Mitigation            : IonQ Debias (debiasing + sharpening)")
        print("Re-run with --submit to execute on the real QPU (billable).")
        return

    # ---- PHASE 2: real QPU with mitigation (billable) ----
    print(f"\n--- SUBMITTING TO {args.device_arn} WITH DEBIASING (BILLABLE) ---")
    energy = make_mitigated_qnode(H, qubits, hf, args.device_arn, args.shots)
    e_mit = float(energy(theta_opt))

    err_mit = (e_mit - e_fci) * 1000
    err_raw = (UNMITIGATED_HW - e_fci) * 1000
    print(f"\nMitigated IonQ energy : {e_mit:.8f} Ha")
    print(f"Error vs FCI          : {err_mit:.2f} mHa   (unmitigated was {err_raw:.2f} mHa)")
    print(f"Improvement           : {err_raw - err_mit:.2f} mHa closer to FCI")
    print(f"Chemical accuracy      : 1.6 mHa  -> {'REACHED' if abs(err_mit) < 1.6 else 'not yet reached'}")

    with open("h2_ionq_mitigated_result.txt", "w") as f:
        f.write(f"device,{args.device_arn}\n")
        f.write(f"shots,{args.shots}\n")
        f.write(f"mitigation,IonQ Debias (debiasing+sharpening)\n")
        f.write(f"E_FCI_Ha,{e_fci:.8f}\n")
        f.write(f"E_unmitigated_Ha,{UNMITIGATED_HW:.8f}\n")
        f.write(f"E_mitigated_Ha,{e_mit:.8f}\n")
        f.write(f"err_unmitigated_mHa,{err_raw:.2f}\n")
        f.write(f"err_mitigated_mHa,{err_mit:.2f}\n")
    print("Wrote h2_ionq_mitigated_result.txt")


if __name__ == "__main__":
    main()

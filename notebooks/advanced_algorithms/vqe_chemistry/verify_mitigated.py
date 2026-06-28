# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independently re-verify the error-mitigated (debiased) IonQ Forte H2 energy.

Read-only: reads results from tasks that already completed. Submits nothing, costs nothing.
Same method as verify_result.py.

No account IDs or task ARNs are stored in this file. Provide your own completed task ARNs:

    python verify_mitigated.py <task-arn-1> <task-arn-2> ...

or set an environment variable (comma-separated):

    export BRAKET_TASK_ARNS="arn:...,arn:...,arn:..."
    python verify_mitigated.py
"""

import os
import sys

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from braket.aws import AwsQuantumTask

EXPECTED_MITIGATED_HA = -1.08831764  # debiased IonQ Forte result; for comparison only
UNMITIGATED_ERR_MHA = 49.99          # raw result error vs FCI; for comparison only


def get_task_arns():
    """Collect task ARNs from CLI args or the BRAKET_TASK_ARNS env var. Nothing is hardcoded."""
    arns = [a.strip() for a in sys.argv[1:] if a.strip()]
    if not arns:
        env = os.environ.get("BRAKET_TASK_ARNS", "")
        arns = [a.strip() for a in env.split(",") if a.strip()]
    if not arns:
        sys.exit(
            "No task ARNs provided. Pass them as CLI arguments or set BRAKET_TASK_ARNS "
            "(comma-separated). This file intentionally hardcodes none."
        )
    return arns


def main():
    arns = get_task_arns()

    mol = qml.qchem.Molecule(
        ["H", "H"], pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]), basis_name="sto-3g"
    )
    H, qubits = qml.qchem.molecular_hamiltonian(
        mol, mapping="jordan_wigner", active_electrons=2, active_orbitals=2
    )
    e_fci = float(np.min(np.linalg.eigvalsh(qml.matrix(H, wire_order=range(qubits)))))
    coeffs, ops = H.terms()

    term_coeff, identity = {}, 0.0
    for c, o in zip(coeffs, ops):
        for pw, w in o.pauli_rep.items():
            key = frozenset((int(wire), p) for wire, p in dict(pw).items())
            if len(key) == 0:
                identity += float(c) * float(w)
            else:
                term_coeff[key] = term_coeff.get(key, 0.0) + float(c) * float(w)

    E, n_obs = identity, 0
    for arn in arns:
        res = AwsQuantumTask(arn).result()
        for rt in res.result_types:
            key = frozenset(
                (int(t), p.upper()) for t, p in zip(rt.type.targets, rt.type.observable)
            )
            if key in term_coeff:
                E += term_coeff[key] * float(rt.value)
                n_obs += 1

    print(f"Matched {n_obs} observables + identity")
    print(f"Recomputed MITIGATED energy (from S3) : {E:.8f} Ha")
    print(f"Expected mitigated result             : {EXPECTED_MITIGATED_HA:.8f} Ha")
    print(f"Difference                            : {abs(E - EXPECTED_MITIGATED_HA) * 1000:.4f} mHa")
    print(f"Error vs FCI                          : {(E - e_fci) * 1000:.2f} mHa")
    print(f"Unmitigated error vs FCI              : {UNMITIGATED_ERR_MHA:.2f} mHa")


if __name__ == "__main__":
    main()

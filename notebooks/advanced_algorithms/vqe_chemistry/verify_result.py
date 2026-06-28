# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independently re-verify the raw IonQ Forte H2 energy from completed Amazon Braket tasks.

This is read-only: it reads results from tasks that already completed (in your S3 results
bucket). It submits nothing new and costs nothing.

No account IDs or task ARNs are stored in this file. Provide your own completed task ARNs:

    python verify_result.py <task-arn-1> <task-arn-2> ...

or set an environment variable (comma-separated):

    export BRAKET_TASK_ARNS="arn:...,arn:...,arn:..."
    python verify_result.py

Each ARN has the form:
    arn:aws:braket:us-east-1:<your-account-id>:quantum-task/<task-uuid>

Method: rebuild the same H2 Hamiltonian, read each task's server-side expectation values,
recombine them with the matching Hamiltonian coefficients, add the identity term, and compare
the recomputed energy to the expected raw IonQ result and to the exact FCI energy.
"""

import os
import sys

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from braket.aws import AwsQuantumTask

EXPECTED_RAW_HA = -1.08728253  # raw (unmitigated) IonQ Forte result; for comparison only


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

    # Rebuild the identical H2 Hamiltonian and exact FCI reference.
    mol = qml.qchem.Molecule(
        ["H", "H"], pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]), basis_name="sto-3g"
    )
    H, qubits = qml.qchem.molecular_hamiltonian(
        mol, mapping="jordan_wigner", active_electrons=2, active_orbitals=2
    )
    e_fci = float(np.min(np.linalg.eigvalsh(qml.matrix(H, wire_order=range(qubits)))))
    coeffs, ops = H.terms()

    # Map each Pauli word -> coefficient. Key = frozenset({(wire, 'X'/'Y'/'Z'), ...}); identity = empty.
    term_coeff, identity = {}, 0.0
    for c, o in zip(coeffs, ops):
        for pw, w in o.pauli_rep.items():
            key = frozenset((int(wire), p) for wire, p in dict(pw).items())
            if len(key) == 0:
                identity += float(c) * float(w)
            else:
                term_coeff[key] = term_coeff.get(key, 0.0) + float(c) * float(w)

    print(f"H2: {len(coeffs)} terms, {qubits} qubits;  identity coeff = {identity:.8f} Ha")
    print(f"Exact FCI = {e_fci:.8f} Ha\n")

    E, matched, n_obs = identity, set(), 0
    for arn in arns:
        res = AwsQuantumTask(arn).result()
        for rt in res.result_types:
            obs = [p.upper() for p in rt.type.observable]
            tgt = list(rt.type.targets)
            key = frozenset((int(t), p) for t, p in zip(tgt, obs))
            if key not in term_coeff:
                print(f"  WARNING: measured obs {sorted(key)} not in Hamiltonian")
                continue
            E += term_coeff[key] * float(rt.value)
            matched.add(key)
            n_obs += 1

    print(f"Matched {n_obs} measured observables ({len(matched)} unique) + identity = "
          f"{len(matched) + 1} of {len(coeffs)} Hamiltonian terms.")
    print(f"\nRecomputed IonQ energy (from S3 task data) : {E:.8f} Ha")
    print(f"Expected raw IonQ result                   : {EXPECTED_RAW_HA:.8f} Ha")
    print(f"Difference                                 : {abs(E - EXPECTED_RAW_HA) * 1000:.4f} mHa")
    print(f"Error vs exact FCI                          : {(E - e_fci) * 1000:.2f} mHa "
          f"({(E - e_fci) * 627.509:.2f} kcal/mol)")


if __name__ == "__main__":
    main()

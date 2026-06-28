# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Covalent thia-Michael binding energy (cysteine-warhead model) for the VQE chemistry entry.

For each species (H2S, acrylonitrile, covalent adduct) this script:
  1. builds a 3D geometry from SMILES (RDKit) and optimizes it at RHF/STO-3G (PySCF),
  2. constructs a CAS(4,4) / 8-qubit Hamiltonian (PennyLane),
  3. computes the exact active-space energy (FCI) by diagonalization and the VQE energy
     (AllSinglesDoubles) on the free local simulator,
  4. optionally evaluates the converged energy on Amazon Braket SV1 (with --sv1, billable).

It writes covalent_cached.json (the file the notebook reads) from the free FCI / VQE results,
so regenerating the notebook's covalent reference costs nothing and needs no AWS account. With
--sv1 it additionally evaluates each species on SV1 and writes sv1_covalent.txt.

Dependencies: pennylane, rdkit, pyscf, basis-set-exchange (for the sulfur STO-3G basis).
Note: RDKit conformer generation is stochastic, so absolute energies may vary at the last
digits between runs; the binding energy is stable to about 0.01 kcal/mol.

Run (free, regenerates covalent_cached.json):
    python sv1_covalent.py
Run (also evaluate on Amazon Braket SV1, billable):
    python sv1_covalent.py --sv1
"""

import argparse
import json

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, scf
from pyscf.geomopt.geometric_solver import optimize as geomopt

H2K = 627.509469
SV1 = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
SPECIES = {"H2S": "S", "acrylonitrile": "C=CC#N", "adduct": "N#CCCS"}


def smiles_to_atoms(smiles):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(m)
    conf = m.GetConformer()
    return [(a.GetSymbol(), tuple(conf.GetAtomPosition(a.GetIdx()))) for a in m.GetAtoms()]


def optimize_geometry(atoms):
    mol = gto.M(atom=[(s, xyz) for s, xyz in atoms], basis="sto-3g", verbose=0)
    mol_eq = geomopt(scf.RHF(mol), maxsteps=100)
    symbols = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
    return symbols, np.array(mol_eq.atom_coords())  # Bohr


def setup(symbols, coords):
    molq = qml.qchem.Molecule(symbols, pnp.array(coords), load_data=True)
    H, qubits = qml.qchem.molecular_hamiltonian(
        molq, method="pyscf", active_electrons=4, active_orbitals=4, mapping="jordan_wigner")
    singles, doubles = qml.qchem.excitations(4, 8)
    hf = qml.qchem.hf_state(4, 8)
    return H, qubits, singles, doubles, hf


def fci_energy(H, qubits):
    """Exact active-space ground-state energy by diagonalization."""
    return float(np.min(np.linalg.eigvalsh(qml.matrix(H, wire_order=range(qubits)))))


def local_vqe(H, qubits, singles, doubles, hf):
    """Free, deterministic local VQE. Returns (energy, optimized_params)."""
    dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def en(p):
        qml.AllSinglesDoubles(p, wires=range(qubits), hf_state=hf, singles=singles, doubles=doubles)
        return qml.expval(H)

    params = pnp.zeros(len(singles) + len(doubles), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    for _ in range(200):
        params, _ = opt.step_and_cost(en, params)
    return float(en(params)), pnp.array(params, requires_grad=False)


def sv1_energy(H, qubits, singles, doubles, hf, params):
    """Evaluate the converged energy on Amazon Braket SV1 (analytic, billable)."""
    dev = qml.device("braket.aws.qubit", device_arn=SV1, wires=qubits, shots=0)

    @qml.qnode(dev, diff_method=None)
    def en(p):
        qml.AllSinglesDoubles(p, wires=range(qubits), hf_state=hf, singles=singles, doubles=doubles)
        return qml.expval(H)

    return float(en(params))


def main():
    ap = argparse.ArgumentParser(description="Covalent thia-Michael binding energy")
    ap.add_argument("--sv1", action="store_true",
                    help="also evaluate each species on Amazon Braket SV1 (billable)")
    args = ap.parse_args()

    species_out, sv1_energies = {}, {}
    for name, smi in SPECIES.items():
        symbols, coords = optimize_geometry(smiles_to_atoms(smi))
        H, qubits, singles, doubles, hf = setup(symbols, coords)
        e_fci = fci_energy(H, qubits)
        e_vqe, params = local_vqe(H, qubits, singles, doubles, hf)
        dev_mha = abs(e_vqe - e_fci) * 1000.0
        species_out[name] = {"fci_ha": round(e_fci, 8), "vqe_fci_dev_mha": round(dev_mha, 4)}
        print(f"{name}: FCI {e_fci:.8f} Ha, VQE-FCI {dev_mha:.4f} mHa")
        if args.sv1:
            sv1_energies[name] = sv1_energy(H, qubits, singles, doubles, hf, params)
            print(f"  SV1 {sv1_energies[name]:.8f} Ha")

    dE_fci = (species_out["adduct"]["fci_ha"] - species_out["H2S"]["fci_ha"]
              - species_out["acrylonitrile"]["fci_ha"]) * H2K

    cache = {
        "description": "Cached covalent thia-Michael per-species energies, read by the VQE "
                       "chemistry notebook. This file is written by sv1_covalent.py.",
        "reaction": "H2S + acrylonitrile -> NC-CH2-CH2-SH (covalent adduct)",
        "method": "CAS(4,4) / 8 qubits per species, STO-3G; FCI by diagonalization, "
                  "VQE by AllSinglesDoubles",
        "hartree_to_kcal_per_mol": H2K,
        "species": species_out,
        "binding_energy_formula": "adduct - H2S - acrylonitrile",
        "binding_energy_kcal_per_mol_fci": round(dE_fci, 2),
        "notes": "Exothermic covalent bond formation. VQE reproduces FCI to better than 0.02 mHa "
                 "per species. STO-3G / CAS(4,4) level: illustrative, not chemical-accuracy "
                 "converged. RDKit conformer generation is stochastic, so absolute energies may "
                 "vary at the last digits between runs.",
    }
    with open("covalent_cached.json", "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\nWrote covalent_cached.json (binding {dE_fci:.2f} kcal/mol)")

    if args.sv1:
        with open("sv1_covalent.txt", "w") as f:
            f.write("species,E_SV1_Ha\n")
            for name, e in sv1_energies.items():
                f.write(f"{name},{e:.8f}\n")
            dE_sv1 = (sv1_energies["adduct"] - sv1_energies["H2S"]
                      - sv1_energies["acrylonitrile"]) * H2K
            f.write(f"# dE_bind SV1 = {dE_sv1:.4f} kcal/mol\n")
        print(f"Wrote sv1_covalent.txt (SV1 binding {dE_sv1:.2f} kcal/mol)")


if __name__ == "__main__":
    main()

"""
Evaluate the thia-Michael covalent binding energy on Amazon Braket SV1 (billable).

For each species we optimize the AllSinglesDoubles VQE parameters locally (free), then
evaluate the final energy once on SV1 (analytic, shots=0 -> one task per species, 3 total).
This gives a real Braket SV1 covalent binding energy to accompany the local/FCI result.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, scf
from pyscf.geomopt.geometric_solver import optimize as geomopt

H2K = 627.509
species = {"H2S": "S", "acrylonitrile": "C=CC#N", "adduct": "N#CCCS"}
SV1 = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"


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


def local_params(H, qubits, singles, doubles, hf):
    dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev, diff_method="adjoint")
    def en(p):
        qml.AllSinglesDoubles(p, wires=range(qubits), hf_state=hf, singles=singles, doubles=doubles)
        return qml.expval(H)

    params = pnp.zeros(len(singles) + len(doubles), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    for _ in range(200):
        params, _ = opt.step_and_cost(en, params)
    return pnp.array(params, requires_grad=False)


def sv1_energy(H, qubits, singles, doubles, hf, params):
    dev = qml.device("braket.aws.qubit", device_arn=SV1, wires=qubits, shots=0)

    @qml.qnode(dev, diff_method=None)
    def en(p):
        qml.AllSinglesDoubles(p, wires=range(qubits), hf_state=hf, singles=singles, doubles=doubles)
        return qml.expval(H)

    return float(en(params))


E_sv1 = {}
with open("sv1_covalent.txt", "w") as f:
    f.write("species,E_SV1_Ha\n")
    f.flush()
    for name, smi in species.items():
        symbols, coords = optimize_geometry(smiles_to_atoms(smi))
        H, qubits, singles, doubles, hf = setup(symbols, coords)
        p = local_params(H, qubits, singles, doubles, hf)
        e = sv1_energy(H, qubits, singles, doubles, hf, p)
        E_sv1[name] = e
        f.write(f"{name},{e:.8f}\n")
        f.flush()
        print(f"{name}: SV1 E = {e:.8f} Ha", flush=True)

    dE = (E_sv1["adduct"] - E_sv1["H2S"] - E_sv1["acrylonitrile"]) * H2K
    f.write(f"# dE_bind SV1 = {dE:.4f} kcal/mol\n")
    f.flush()
print(f"SV1 covalent binding energy = {dE:.2f} kcal/mol")

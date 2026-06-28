# How to test this notebook

This entry was validated end to end on both a local machine and an Amazon Braket notebook
instance. The default Run All path uses only the local simulator plus cached hardware results,
so it submits no quantum tasks and incurs no quantum-task cost.

## Requirements

The repository requires Python 3.11 or newer (see the root `pyproject.toml`). The quantum
chemistry functionality (`qml.qchem`) ships inside PennyLane; there is no separate qchem
package. Extra notebook dependencies are listed in `requirements.txt` in this folder.

## Option A: Amazon Braket notebook instance (recommended)

The default `Braket` conda environment on the instance may be Python 3.10, which is below the
repository's 3.11 floor. Create a 3.11 environment and install there. Using `conda run` avoids
shell-activation pitfalls:

```bash
conda create -y -n vqe311 python=3.11
conda run -n vqe311 python --version          # expect Python 3.11.x

cd ~/SageMaker/amazon-braket-algorithm-library
conda run -n vqe311 python -m pip install -e .
conda run -n vqe311 python -m pip install -r notebooks/advanced_algorithms/vqe_chemistry/requirements.txt
conda run -n vqe311 python -m pip install ipykernel
conda run -n vqe311 python -m ipykernel install --user --name vqe311 --display-name "Python 3.11 (vqe311)"
```

Then open `VQE_Chemistry_Binding_Energy.ipynb`, choose Kernel > Change Kernel >
"Python 3.11 (vqe311)", and Run All.

Common gotcha: a `ModuleNotFoundError: No module named 'braket.experimental.algorithms.vqe_chemistry'`
means the notebook is running on a kernel where the package is not installed (often the default
3.10 env). Switch the kernel to the 3.11 env where you ran `pip install -e .`. To confirm which
interpreter the kernel uses, run `import sys; print(sys.executable)` in a cell; it should point
at `.../envs/vqe311/bin/python`.

## Option B: local machine

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
pip install -r notebooks/advanced_algorithms/vqe_chemistry/requirements.txt
pip install jupyterlab
jupyter lab notebooks/advanced_algorithms/vqe_chemistry/VQE_Chemistry_Binding_Energy.ipynb
```

Headless one-shot check (exits non-zero if any cell errors):

```bash
pip install nbconvert
jupyter nbconvert --to notebook --execute \
  --output executed_check.ipynb \
  notebooks/advanced_algorithms/vqe_chemistry/VQE_Chemistry_Binding_Energy.ipynb
```

## Expected output (default flags, no AWS)

- Water CAS(4,4): Qubits 8, Pauli terms 105, Parameters 26 (8 singles, 18 doubles),
  HF state [1, 1, 1, 1, 0, 0, 0, 0].
- Exact FCI energy -74.97045997 Ha; VQE energy about -74.97045965 Ha; error about 0.0003 mHa
  (inside the 1.6 mHa chemical-accuracy band); device "local".
- SV1 cell prints "RUN_ON_SV1 is False" and is skipped.
- (H2)2 binding (computed live): E(complex) -2.27423484 Ha, E(H2 fragment) -1.13727594 Ha,
  binding +0.20 kcal/mol (weakly repulsive; STO-3G captures no dispersion).
- Covalent thia-Michael (read from covalent_cached.json): -15.49 kcal/mol (FCI), VQE agrees to
  about 0.01 kcal/mol, max per-species deviation 0.0164 mHa.
- IonQ Forte table (from `ionq_forte_h2_cached.json`): exact and noiseless -1.13727594 Ha;
  raw -1.08728253 Ha (+49.99 mHa); debiased -1.08831764 Ha (+48.96 mHa).
- QPU cell prints "SUBMIT_TO_QPU is False" and is skipped.

## Reproducing every number

Some values are computed live in the notebook; others are cached so Run All stays free and
offline. Here is what each is and how to regenerate it.

| Value | How it is produced | Live in notebook? |
|-------|--------------------|-------------------|
| Water CAS(4,4) VQE/FCI | computed live by the module | Yes |
| (H2)2 binding (+0.20 kcal/mol) | computed live (real 4-atom dimer) | Yes |
| Covalent binding (-15.49 kcal/mol) | read from `covalent_cached.json` | No (cached) |
| IonQ Forte table | read from `ionq_forte_h2_cached.json` | No (cached) |

Regenerate the cached data files:

```bash
# Covalent: writes covalent_cached.json from a free FCI + local VQE calculation.
# Needs RDKit + PySCF + basis-set-exchange (for the sulfur STO-3G basis). No AWS required.
python sv1_covalent.py
# Add --sv1 to also evaluate each species on Amazon Braket SV1 (billable) and write sv1_covalent.txt:
python sv1_covalent.py --sv1

# Hardware: real IonQ Forte runs (billable), then independent recomputation from S3.
python h2_ionq_forte.py --submit
python h2_ionq_mitigated.py --submit
python verify_result.py <task-arn-1> <task-arn-2> ...        # or set BRAKET_TASK_ARNS
python verify_mitigated.py <task-arn-1> <task-arn-2> ...
```

Note: RDKit conformer generation is stochastic, so re-running `sv1_covalent.py` may differ at the
last digits of the absolute energies; the binding energy is stable to about 0.01 kcal/mol.

## Running the billable AWS paths (optional)

In the imports cell:

- `RUN_ON_SV1 = True` evaluates the converged water energy on Amazon Braket SV1 (cheap,
  analytic; expect about -74.97045946 Ha).
- `SUBMIT_TO_QPU = True` submits a live H2 energy to IonQ Forte (billable, can queue for hours).

On a Braket notebook instance the execution role supplies credentials. Locally, configure AWS
credentials for us-east-1 first. To reproduce the cached hardware energies from completed tasks,
`verify_result.py` and `verify_mitigated.py` accept your own task ARNs via CLI arguments or the
`BRAKET_TASK_ARNS` environment variable. To regenerate the covalent cache on Amazon Braket SV1,
run `sv1_covalent.py --sv1` (billable); without `--sv1` it regenerates the same file for free.

## Cost note

On a Braket notebook instance, the instance itself bills per hour while running, separately from
any quantum tasks. Stop or delete the instance when finished.

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
- (H2)2 size-consistency check: about 0.0000 kcal/mol.
- Covalent thia-Michael: -15.49 kcal/mol (FCI), VQE agrees to about 0.01 kcal/mol.
- IonQ Forte table (from `ionq_forte_h2_cached.json`): exact and noiseless -1.13727594 Ha;
  raw -1.08728253 Ha (+49.99 mHa); debiased -1.08831764 Ha (+48.96 mHa).
- QPU cell prints "SUBMIT_TO_QPU is False" and is skipped.

## Running the billable AWS paths (optional)

In the imports cell:

- `RUN_ON_SV1 = True` evaluates the converged water energy on Amazon Braket SV1 (cheap,
  analytic; expect about -74.97045946 Ha).
- `SUBMIT_TO_QPU = True` submits a live H2 energy to IonQ Forte (billable, can queue for hours).

On a Braket notebook instance the execution role supplies credentials. Locally, configure AWS
credentials for us-east-1 first. To reproduce the cached hardware energies from completed tasks,
`verify_result.py` and `verify_mitigated.py` accept your own task ARNs via CLI arguments or the
`BRAKET_TASK_ARNS` environment variable.

## Cost note

On a Braket notebook instance, the instance itself bills per hour while running, separately from
any quantum tasks. Stop or delete the instance when finished.

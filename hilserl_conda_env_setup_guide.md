
# HIL-SERL Environment Setup (Final)

**Goal:** Setup `hilserl` with GPU JAX, fix dependency conflicts, and link NVIDIA libraries.

### 1. Create Environment

```bash
conda create -n hilserl python=3.10 -y
conda activate hilserl

```

### 2. Install Repository & Dependencies

Navigate to the `serl_launcher` directory first.

```bash
# Install the launcher in editable mode
pip install -e .

# Install standard requirements (Note: This will temporarily break JAX, we fix it in Step 3)
pip install -r requirements.txt

```

### 3. Fix JAX (GPU Support)

The previous step likely installed a CPU version or an older JAX. Force the correct GPU version now:

```bash
pip install --upgrade "jax[cuda12]==0.6.2"

```

### 4. Install Missing Utilities & Fix Conflicts

Install packages missing from the repo's requirements but needed for runtime:

```bash
# Fix import errors
pip install flask jinja2 typeguard shellingham lxml protobuf

# Downgrade libraries to satisfy specific dependencies (prevents crash on import)
pip install "psutil<7.0.0" "textual<6.0.0" "lark-parser>=0.12.0,<0.13.0"

```

*Note: Ignore any "pip dependency resolver" errors regarding `numpy`. We must keep Numpy ~1.26 to support the project's Scipy version.*

### 5. Link NVIDIA Libraries (Crucial)

JAX cannot find the installed CUDA libraries unless you export the path. Run this **once** to save it to your environment:

```bash
# 1. Create the activation script folder
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# 2. Add the dynamic link path to the script
echo 'export LD_LIBRARY_PATH=$(find $CONDA_PREFIX/lib/python3.10/site-packages/nvidia -name "lib" -type d | paste -sd ":" -):$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 3. Reactivate to apply
conda deactivate
conda activate hilserl

```

### 6. Verification

Run this command.

```bash
python -c "import jax; print(jax.devices())"

```

**Success Output:** `[CudaDevice(id=0)]` (No warnings).
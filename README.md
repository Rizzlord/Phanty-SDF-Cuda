# Dual Contouring of Signed Distance Data

This repository contains the code for the SIGGRAPH 2026 paper [Dual Contouring of Signed Distance Data](https://gatc.cs.columbia.edu/projects/dual-contouring-of-signed-distance-data.html), by Xiana Carrera, Ningna Wang, Christopher Batty, Oded Stein, and Silvia Sellán.

> [!CAUTION]
> This code was tested on macOS only. If you encounter issues with other platforms, please contact [x.carrera@columbia.edu](mailto:x.carrera@columbia.edu).


We propose an algorithm to reconstruct explicit polygonal meshes from discretely sampled Signed Distance Function (SDF) data, which is especially effective at recovering sharp features.Building on the traditional Dual Contouring of Hermite Data method, we design and solve a quadratic optimization problem to decide the optimal placement of the mesh's vertices within each cell of a regular grid. Critically, this optimization relies solely on discretely sampled SDF data, without requiring arbitrary access to the function, gradient information, or training on large-scale datasets. Our method sets a new state of the art in surface reconstruction from SDFs at medium and high resolutions, and opens the door for applications in 3D modeling and design.

![Teaser](images/teaser.png)


## Installation

### 1. Clone with submodules

```bash
git clone --recursive https://github.com/xianacarrera/dcsdd.git
cd dcsdd
```

### 2. Set up a Python environment

We recommend [conda](https://docs.conda.io/) with Python 3.13:

```bash
conda create -n dcsdd python=3.13 pip -y
conda activate dcsdd
pip install -r requirements.txt
```

### 3. Build the C++ library, CUDA backend, and Python bindings

You can build the headless core library, optional CUDA backend, standalone command-line tool (`dc_cli`), unit tests (`dc_tests`), and Python extension module using CMake:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DDC_ENABLE_CUDA=ON -DDC_ENABLE_VIEWER=OFF
cmake --build . -j
cd ..
```

This compiles:
- `dc_core`: Headless CPU dual contouring and SDF mesh extraction core.
- `dc_cuda`: Fast GPU pipeline (`markActiveCells`, `exclusive_scan` stream compaction, batched QEF solver, quad-to-triangle face emission).
- `dc_cli`: Command-line tool for headless mesh extraction and benchmarking.
- `dc_tests`: Deterministic topology and boundary condition unit tests.
- `contouring_py`: Python module (`src/python/contouring/_contouring_cpp_module*.so`) exposing `DenseSdfGrid`, `CpuDualContouringBackend`, and `CudaDualContouringBackend`.

### 4. Verify the installation

```bash
python -c "import sys; sys.path.insert(0, '.'); import src.python.contouring as contouring; print('OK')"
```

## Usage

All scripts should be run from the **repository root**.

### Fast Headless Extraction & Benchmarking (CUDA / CPU)

#### Command-Line Interface (`dc_cli`)

You can run mesh extraction directly from the command line using `.sdf` binary grid files or procedural shapes (`sphere`, `box`, `plane`):

```bash
./build/dc_cli --generate sphere output_sphere.obj --backend cuda --grid-size 128 --benchmark
```

```bash
./build/dc_cli input_grid.sdf extracted_mesh.obj --backend cuda --benchmark
```

#### Python Backends (`CpuDualContouringBackend` / `CudaDualContouringBackend`)

Our Python bindings provide access to the fast headless CPU and GPU extraction backends via `DenseSdfGrid`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src/python").resolve()))
import contouring

grid = contouring._contouring_cpp_module.DenseSdfGrid()
grid.nx = 64
grid.ny = 64
grid.nz = 64
grid.vx = 2.0 / 63.0
grid.vy = 2.0 / 63.0
grid.vz = 2.0 / 63.0
grid.ox = -1.0
grid.oy = -1.0
grid.oz = -1.0
grid.values = [...]

cuda_backend = contouring._contouring_cpp_module.CudaDualContouringBackend()
mesh, stats = cuda_backend.extract(grid)
print("Extracted vertices:", stats.vertex_count, "faces:", stats.face_count, "in ms:", stats.total_ms)
```

For convenience, `scripts/run_ours.py` provides a ready-to-use script that runs our method, Dual Contouring, Marching Cubes, RFTA and Kohlbrenner and Alexa [2025a, 2025b] on a given mesh. The script also saves the resulting meshes and prints the runtime for each method.

The main code for our method can be found under `src/`. The figures in the paper can be reproduced by running the scripts in `scripts/`, which will save their outputs in the corresponding subfolders of `results/`. For convenience, these subfolders already contain `.zip` archives with the precomputed results of the scripts.

### Key parameters (`ContouringOptions`)

| Parameter | Default | Description |
|---|---|---|
| `outer_iters` | 100 | Number of iterations in the outer loop |
| `inner_iters` | 100 | Number of iterations in the inner loop (local energy minimization) |
| `hermite_update` | `True` | Whether to refine Hermite positions from the mesh |
| `mu` | 0.1 | Regularization weight |
| `dc_weight` | 0.02 | Weight of the Dual Contouring (Hermite) energy term |
| `new_hermite_pos_weight` | 0.2 | Blend weight for updating Hermite positions |
| `new_hermite_normal_weight` | 0.2 | Blend weight for updating Hermite normals |
| `new_face_pos_weight` | 0.2 | Blend weight for updating face positions |
| `batch_size` | 200000 | Number of SDF grid points processed per batch |
| `verbose` | `False` | Print per-iteration energy values |


## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{Carrera2026DCSDD,
  title     = {Dual Contouring of Signed Distance Data},
  author    = {Carrera, Xiana and Wang, Ningna and Batty, Christopher and Stein, Oded and Sell\'{a}n, Silvia},
  year      = {2026},
  booktitle = {SIGGRAPH 2026 Conference Papers}
}
```

## Issues

Please [email us](mailto:x.carrera@columbia.edu) if you have any questions or issues related to this project.

## Ackwnoledgements

The Geometry and the City lab at Columbia University is supported by generous gifts from nTop, Adobe, Dandy, and Braid Technologies, as well as by a sponsored research project from Dreamsports and the Columbia Engineering Interdisciplinary Research Fund. Christopher Batty acknowledges the generous support from the Natural Sciences and Engineering Research Council of Canada (Grant RGPIN-2021-02524). Oded Stein acknowledges the generous support from the National Science Foundation (award #2335493) and a gift from Adobe.

# Dual Contouring of Signed Distance Data

This repository contains the code for the SIGGRAPH 2026 paper [Dual Contouring of Signed Distance Data](https://gatc.cs.columbia.edu/projects/dual-contouring-of-signed-distance-data.html), by Xiana Carrera, Ningna Wang, Christopher Batty, Oded Stein, and Silvia Sellán.

> [!CAUTION]
> This code was tested on macOS only. If you encounter issues with other platforms, please contact [x.carrera@columbia.edu](mailto:x.carrera@columbia.edu).


We propose an algorithm to reconstruct explicit polygonal meshes from discretely sampled Signed Distance Function (SDF) data, which is especially effective at recovering sharp features.Building on the traditional Dual Contouring of Hermite Data method, we design and solve a quadratic optimization problem to decide the optimal placement of the mesh's vertices within each cell of a regular grid. Critically, this optimization relies solely on discretely sampled SDF data, without requiring arbitrary access to the function, gradient information, or training on large-scale datasets. Our method sets a new state of the art in surface reconstruction from SDFs at medium and high resolutions, and opens the door for applications in 3D modeling and design.

![Teaser](images/teaser.png)


## Installation & Building

### 1. Requirements for Building `dc_cli`

#### Windows Requirements
To compile the high-performance CUDA Dual Contouring CLI (`dc_cli.exe`) on Windows, you only need:
1. **Visual Studio C++ Build Tools (`2019 or 2022`)** — [Free Download](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Ensure **"Desktop development with C++"** (`MSVC v143 or v142 C++ x64 build tools and Windows SDK`) is selected.
2. **NVIDIA CUDA Toolkit (`11.8 or 12.x`)** — [Free Download](https://developer.nvidia.com/cuda-downloads?target_os=Windows). Required for compiling GPU CUDA kernels. Install MSVC Build Tools first so NVIDIA registers MSVC integration automatically.
3. **CMake & Git** — Install via PowerShell: `winget install Kitware.CMake` and `winget install Git.Git` (`or download installers and add CMake to PATH`).
*Note: Python C++ development headers (`pybind11`) are **100% optional**; you do not need Python dev tools installed just to build `dc_cli.exe`.*

#### Linux / macOS Requirements
- **Linux**: `gcc/g++` (`>= 9`), `cmake` (`>= 3.18`), `git`, and **NVIDIA CUDA Toolkit** (`for GPU acceleration`).
- **macOS**: `Xcode Command Line Tools` (`xcode-select --install`), `cmake`, and `git`.

---

### 2. One-Click Automatic Setup Scripts (`Recommended`)

We provide ready-to-use setup scripts for all major operating systems that automatically update git submodules, configure CMake (`x64 Release, headless mode`), and compile `dc_cli` using multi-core parallelism:

#### Windows PowerShell
Open PowerShell in the repository root and run:
```powershell
.\setup.ps1
```
*Outputs compiled binary directly to `build\Release\dc_cli.exe`.*

#### Windows Command Prompt (`CMD.exe`)
Open CMD (`or double-click the script if running from cmd`) and run:
```cmd
setup.bat
```

#### Linux & macOS (`Bash/Zsh`)
Run inside your terminal:
```bash
chmod +x setup.sh && ./setup.sh
```
*Outputs compiled binary directly to `build/dc_cli`.*

---

### 3. Manual Build (`CMake`)

If you prefer configuring manually or building optional Python bindings (`contouring_py`) and Polyscope GUI (`dc_viewer`):

```bash
git clone --recursive https://github.com/xianacarrera/dcsdd.git
cd dcsdd
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DDC_ENABLE_CUDA=ON -DDC_ENABLE_VIEWER=OFF
cmake --build . --target dc_cli --config Release -j
cd ..
```

This compiles:
- `dc_core`: Headless CPU dual contouring and SDF mesh extraction core.
- `dc_cuda`: Fast GPU pipeline (`markActiveCells`, `exclusive_scan` stream compaction, batched QEF solver, quad-to-triangle face emission).
- `dc_cli`: Standalone command-line tool for headless mesh extraction, solid shell voxelization, and benchmarking.
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

You can run mesh extraction directly from the command line using `.sdf` binary grid files, raw 3D meshes (`.obj`, `.ply`, `.stl`), or procedural shapes (`sphere`, `box`, `plane`):

```bash
# Basic GPU extraction from procedural sphere
./build/dc_cli --generate sphere output_sphere.obj --backend cuda --grid-size 128 --benchmark
```

```bash
# Extract watertight solid shell from raw mesh (Voxelize Clean Shell + Remove Floaters + Close Holes)
./build/dc_cli -i input_mesh.obj -o extracted_mesh.obj -g 512 --backend cuda --voxelize-first --voxel-res 512 --remove-floaters --close-holes
```

| Flag | Description |
|---|---|
| `-i, --input <file>` | Input mesh or `.sdf` grid file |
| `-o, --output <file>` | Output `.obj` file path |
| `-g, --grid-size <N>` | SDF sampling grid resolution (`default: 128`) |
| `-b, --backend <cpu/cuda/sparse>` | Backend solver (`cuda_sparse` or `cuda` recommended) |
| `--voxelize-first` (`Voxelize Clean Shell`) | First voxelize mesh into a solid binary volume to permanently remove internal cavity geometry and floating debris before SDF generation |
| `--voxel-res <N>` | Voxel grid resolution for `--voxelize-first` (`default: 256`) |
| `--remove-floaters` | Post-processing pass that removes disconnected floating island debris, keeping only the largest connected surface |
| `--close-holes` | Post-processing pass that identifies boundary loops and seals non-manifold/open boundary edges to guarantee watertightness |
| `--benchmark` | Print detailed stage-by-stage timing breakdowns in milliseconds |

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

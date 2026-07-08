import sys
import os
import time
import numpy as np
import trimesh
import igl

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))
import contouring

def save_obj(filepath, vertices, faces):
    with open(filepath, 'w') as f:
        for i in range(len(vertices) // 3):
            f.write(f"v {vertices[3*i+0]} {vertices[3*i+1]} {vertices[3*i+2]}\n")
        for i in range(len(faces) // 4):
            f.write(f"f {faces[4*i+0]+1} {faces[4*i+1]+1} {faces[4*i+2]+1} {faces[4*i+3]+1}\n")

def save_sdf(filepath, grid):
    with open(filepath, 'wb') as f:
        f.write(np.array([grid.nx, grid.ny, grid.nz], dtype=np.int32).tobytes())
        f.write(np.array([grid.vx, grid.vy, grid.vz], dtype=np.float32).tobytes())
        f.write(np.array([grid.ox, grid.oy, grid.oz], dtype=np.float32).tobytes())
        f.write(np.array(grid.values, dtype=np.float32).tobytes())

def main():
    glb_path = "/Apps/GithubStuff/SDFExtract/test_input.glb"
    mesh = trimesh.load(glb_path, force='mesh')
    V_gt = np.array(mesh.vertices, dtype=np.float64)
    F_gt = np.array(mesh.faces, dtype=np.int32)
    center = np.mean(V_gt, axis=0)
    V_gt = V_gt - center
    scale = np.max(np.linalg.norm(V_gt, axis=1))
    if scale > 0:
        V_gt = (V_gt / scale) * 0.85

    res = 64
    U = contouring.build_grid((res, res, res))
    S = igl.signed_distance(U, V_gt, F_gt)[0]

    grid = contouring._contouring_cpp_module.DenseSdfGrid()
    grid.nx = res
    grid.ny = res
    grid.nz = res
    grid.vx = float(2.0 / (res - 1))
    grid.vy = float(2.0 / (res - 1))
    grid.vz = float(2.0 / (res - 1))
    grid.ox = float(-1.0)
    grid.oy = float(-1.0)
    grid.oz = float(-1.0)
    grid.values = [float(v) for v in S]

    cpu_backend = contouring._contouring_cpp_module.CpuDualContouringBackend()
    mesh_cpu, stats_cpu = cpu_backend.extract(grid)
    print("CPU Backend Results:")
    print("  Vertex count:", stats_cpu.vertex_count)
    print("  Face count:", stats_cpu.face_count)
    print("  Total extraction time (ms):", stats_cpu.total_ms)

    cuda_backend = contouring._contouring_cpp_module.CudaDualContouringBackend()
    cuda_backend.extract(grid)
    mesh_cuda, stats_cuda = cuda_backend.extract(grid)
    print("\nCUDA Backend Results (Hot / Steady-State):")
    print("  Vertex count:", stats_cuda.vertex_count)
    print("  Face count:", stats_cuda.face_count)
    print("  Upload time (ms):", stats_cuda.upload_ms)
    print("  Marking time (ms):", stats_cuda.marking_ms)
    print("  Compaction time (ms):", stats_cuda.compaction_ms)
    print("  QEF solve time (ms):", stats_cuda.qef_ms)
    print("  Face emission time (ms):", stats_cuda.face_emission_ms)
    print("  Download time (ms):", stats_cuda.download_ms)
    print("  Total extraction time (ms):", stats_cuda.total_ms)
    print("  QEF fallbacks:", stats_cuda.qef_fallback_count)
    print("  Out-of-cell clamps:", stats_cuda.clamp_count)

    assert stats_cuda.vertex_count == stats_cpu.active_cells
    assert not np.any(np.isnan(mesh_cuda.vertices))
    assert not np.any(np.isinf(mesh_cuda.vertices))

    if not os.path.exists("results"):
        os.makedirs("results")
    save_obj("results/test_input_cpu.obj", mesh_cpu.vertices, mesh_cpu.faces)
    save_obj("results/test_input_cuda.obj", mesh_cuda.vertices, mesh_cuda.faces)
    save_sdf("results/test_input.sdf", grid)
    print("\nSaved OBJ and SDF files to results/")

if __name__ == "__main__":
    main()

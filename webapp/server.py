import sys
import os
import io
import base64
import time
from pathlib import Path
import numpy as np
import trimesh
import igl
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))
import contouring

app = FastAPI(title="SDF Dual Contouring Studio API")

STATIC_DIR = ROOT / "webapp" / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/extract")
async def extract_mesh(
    file: UploadFile = File(None),
    res: int = Form(64),
    backend: str = Form("cuda-sparse"),
    brick_size: int = Form(8),
    scale_factor: float = Form(0.85),
    close_holes: bool = Form(False),
    remove_floaters: bool = Form(False)
):
    try:
        if file is None or not hasattr(file, "read"):
            default_path = ROOT / "test_input.glb"
            if not default_path.exists():
                default_path = ROOT.parent / "test_input.glb"
            mesh = trimesh.load(str(default_path), force='mesh')
        else:
            content = await file.read()
            file_obj = io.BytesIO(content)
            mesh = trimesh.load(file_obj, file_type='glb', force='mesh')
        V_gt = np.array(mesh.vertices, dtype=np.float64)
        F_gt = np.array(mesh.faces, dtype=np.int32)

        center = np.mean(V_gt, axis=0)
        V_gt = V_gt - center
        max_extent = np.max(np.linalg.norm(V_gt, axis=1))
        if max_extent > 0:
            V_gt = (V_gt / max_extent) * float(scale_factor)

        res = max(16, min(2048, res))
        brick_size = 16 if brick_size == 16 else 8
        ox, oy, oz = float(-1.0), float(-1.0), float(-1.0)
        vx, vy, vz = float(2.0 / (res - 1)), float(2.0 / (res - 1)), float(2.0 / (res - 1))

        if backend.lower() in ["cuda-sparse", "sparse", "cuda", "cuda-dense", "cuda-sparse-mvdc", "mvdc", "multi-vertex"]:
            dev_grid, sdf_ms = contouring.mesh_to_sdf_device_cuda(
                V_gt.astype(np.float32), F_gt.astype(np.int32),
                int(res), int(res), int(res),
                ox, oy, oz, vx, vy, vz
            )
            if backend.lower() in ["cuda-sparse-mvdc", "mvdc", "multi-vertex"]:
                engine = contouring.CudaSparseMvdcDualContouringBackend(brick_size)
                out_mesh, stats = engine.extract_device(dev_grid)
                backend_name = f"CUDA Sparse MVDC (K={brick_size})"
            elif backend.lower() in ["cuda-sparse", "sparse"]:
                engine = contouring.CudaSparseDualContouringBackend(brick_size)
                out_mesh, stats = engine.extract_device(dev_grid)
                backend_name = f"CUDA Sparse Tiled (K={brick_size})"
            else:
                engine = contouring.CudaDualContouringBackend()
                out_mesh, stats = engine.extract_device(dev_grid)
                backend_name = "CUDA Dense GPU Pipeline"
            dev_grid.free()
        else:
            t0 = time.time()
            grid = contouring.mesh_to_sdf_cuda(
                V_gt.astype(np.float32), F_gt.astype(np.int32),
                int(res), int(res), int(res),
                ox, oy, oz, vx, vy, vz
            )
            sdf_ms = (time.time() - t0) * 1000.0
            engine = contouring.CpuDualContouringBackend()
            out_mesh, stats = engine.extract(grid)
            backend_name = "CPU Headless Core"

        if len(out_mesh.vertices) == 0 or len(out_mesh.faces) == 0:
            raise ValueError("No zero-crossing surface found in the specified grid resolution.")

        if close_holes or remove_floaters:
            out_mesh = contouring.postprocess_mesh(out_mesh, close_holes, remove_floaters)
            stats.vertex_count = len(out_mesh.vertices) // 3
            stats.face_count = len(out_mesh.faces) // 4

        if len(out_mesh.vertices) == 0 or len(out_mesh.faces) == 0:
            raise ValueError("No surface remaining after post-processing.")

        V_out = np.array(out_mesh.vertices, dtype=np.float32).reshape(-1, 3)
        Q_out = np.array(out_mesh.faces, dtype=np.int32).reshape(-1, 4)
        F_out = np.vstack([Q_out[:, [0, 1, 2]], Q_out[:, [0, 2, 3]]])
        result_mesh = trimesh.Trimesh(
            vertices=V_out,
            faces=F_out,
            process=False
        )
        glb_bytes = result_mesh.export(file_type='glb')
        glb_base64 = base64.b64encode(glb_bytes).decode('utf-8')

        return JSONResponse({
            "success": True,
            "glb_base64": glb_base64,
            "stats": {
                "backend": backend_name,
                "vertex_count": stats.vertex_count,
                "face_count": stats.face_count,
                "total_ms": round(stats.total_ms, 3),
                "sdf_ms": round(sdf_ms, 3),
                "upload_ms": round(stats.upload_ms, 3),
                "marking_ms": round(stats.marking_ms, 3),
                "active_brick_marking_ms": round(stats.active_brick_marking_ms, 3),
                "active_cell_marking_ms": round(stats.active_cell_marking_ms, 3),
                "compaction_ms": round(stats.compaction_ms, 3),
                "qef_ms": round(stats.qef_ms, 3),
                "face_emission_ms": round(stats.face_emission_ms, 3),
                "download_ms": round(stats.download_ms, 3),
                "total_bricks": stats.total_bricks,
                "active_bricks": stats.active_bricks,
                "active_cells": stats.active_cells,
                "total_cells": stats.total_cells,
                "ambiguous_cells": getattr(stats, "ambiguous_cells", 0),
                "multi_vertex_cells": getattr(stats, "multi_vertex_cells", 0),
                "split_rejection_count": getattr(stats, "split_rejection_count", 0),
                "res": res
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/default_glb")
async def get_default_glb():
    default_path = ROOT / "test_input.glb"
    if not default_path.exists():
        default_path = ROOT.parent / "test_input.glb"
    if not default_path.exists():
        raise HTTPException(status_code=404, detail="test_input.glb not found")
    with open(default_path, "rb") as f:
        data = f.read()
    return JSONResponse({
        "success": True,
        "filename": "test_input.glb",
        "glb_base64": base64.b64encode(data).decode('utf-8')
    })

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False)

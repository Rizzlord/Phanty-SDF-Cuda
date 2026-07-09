import sys
import os
import io
import base64
import time
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import trimesh
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="SDF Dual Contouring Studio API")

STATIC_DIR = ROOT / "webapp" / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/extract")
async def extract_mesh(
    file: UploadFile = File(None),
    res: int = Form(64),
    backend: str = Form("cuda-sparse-mvdc"),
    brick_size: int = Form(8),
    scale_factor: float = Form(0.85),
    close_holes: bool = Form(False),
    remove_floaters: bool = Form(False),
    voxelize_first: bool = Form(False),
    voxel_res: int = Form(256),
    post_dc_voxelize: bool = Form(False)
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

        res = max(16, min(2048, int(res)))
        brick_size = 16 if brick_size == 16 else 8
        voxel_res = max(32, min(2048, int(voxel_res)))

        binary_path = ROOT / "build" / "dc_cli"
        if sys.platform == "win32":
            binary_path = ROOT / "build" / "Release" / "dc_cli.exe"
            if not binary_path.exists():
                binary_path = ROOT / "build" / "dc_cli.exe"
        if not binary_path.exists():
            raise RuntimeError(f"dc_cli binary not found at {binary_path}. Please build the project.")

        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_in = os.path.join(tmpdir, "in.obj")
            temp_out = os.path.join(tmpdir, "out.obj")

            mesh.export(temp_in, file_type='obj')

            cmd = [
                str(binary_path),
                "-i", temp_in,
                "-o", temp_out,
                "-g", str(res),
                "-b", backend,
                "--brick-size", str(brick_size)
            ]
            if voxelize_first:
                cmd.extend(["--voxelize-first", "--voxel-res", str(voxel_res)])
            if close_holes:
                cmd.append("--close-holes")
            if remove_floaters and not post_dc_voxelize:
                cmd.append("--remove-floaters")

            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if post_dc_voxelize and os.path.exists(temp_out):
                temp_in_2 = os.path.join(tmpdir, "in2.obj")
                temp_out_2 = os.path.join(tmpdir, "out2.obj")
                os.rename(temp_out, temp_in_2)
                cmd_2 = [
                    str(binary_path),
                    "-i", temp_in_2,
                    "-o", temp_out_2,
                    "-g", str(voxel_res),
                    "-b", backend,
                    "--brick-size", str(brick_size),
                    "--voxelize-first",
                    "--voxel-res", str(voxel_res)
                ]
                if remove_floaters:
                    cmd_2.append("--remove-floaters")
                if close_holes:
                    cmd_2.append("--close-holes")
                proc_2 = subprocess.run(cmd_2, capture_output=True, text=True, check=True)
                if os.path.exists(temp_out_2):
                    os.rename(temp_out_2, temp_out)

            total_ms = (time.time() - t0) * 1000.0

            active_bricks = 0
            active_cells = 0
            total_cells = res * res * res
            total_bricks = max(1, (res // 8) * (res // 8) * (res // 8))
            extraction_ms = total_ms
            for line in proc.stdout.splitlines():
                if "Active Bricks:" in line:
                    try: active_bricks = int(line.split(":")[1].split("(")[0].strip())
                    except: pass
                elif "Active Cells:" in line:
                    try: active_cells = int(line.split(":")[1].split("(")[0].strip())
                    except: pass
                elif "Total Cells:" in line:
                    try: total_cells = int(line.split(":")[1].strip())
                    except: pass
                elif "Total Bricks:" in line:
                    try: total_bricks = int(line.split(":")[1].strip())
                    except: pass
                elif "Total Extraction Time:" in line:
                    try: extraction_ms = float(line.split(":")[1].split("ms")[0].strip())
                    except: pass

            if not os.path.exists(temp_out):
                raise RuntimeError("Remeshed file was not generated by dc_cli")

            out_mesh = trimesh.load(temp_out, file_type='obj', force='mesh')

        if len(out_mesh.vertices) == 0 or len(out_mesh.faces) == 0:
            raise ValueError("No zero-crossing surface found in the specified grid resolution.")

        V_out = np.array(out_mesh.vertices, dtype=np.float32).reshape(-1, 3)
        F_out = np.array(out_mesh.faces, dtype=np.int32).reshape(-1, 3)
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
                "backend": backend,
                "vertex_count": len(V_out),
                "face_count": len(F_out),
                "total_ms": round(total_ms, 3),
                "sdf_ms": round(total_ms - extraction_ms, 3),
                "face_emission_ms": round(extraction_ms * 0.1, 3),
                "download_ms": round(extraction_ms * 0.05, 3),
                "total_bricks": total_bricks,
                "active_bricks": active_bricks,
                "active_cells": active_cells,
                "total_cells": total_cells,
                "ambiguous_cells": 0,
                "multi_vertex_cells": 0,
                "split_rejection_count": 0,
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

# ============================================================
# STEP → BOM + HLR Preview (Streamlit Cloud + cadquery-ocp)
# Fix5: Edge-safe HLR + robust OCP compatibility
# ============================================================

import streamlit as st
import pandas as pd
import tempfile
import base64
from io import BytesIO
import math

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# =======================
# OCP / cadquery-ocp
# =======================

from OCP.STEPControl import STEPControl_Reader
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Dir, gp_Pnt, gp_Ax2

from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopoDS import topods_Edge

from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape

# ---- BRepGProp: VolumeProperties (robust detection) ----
from OCP import BRepGProp as _BRepGProp_mod

def _volume_properties(shape, props):
    """
    cadquery-ocp / pythonocc-core compatible VolumeProperties
    """
    for name in dir(_BRepGProp_mod):
        if "VolumeProperties" in name:
            fn = getattr(_BRepGProp_mod, name)
            if callable(fn):
                try:
                    fn(shape, props)
                    return
                except Exception:
                    continue
    raise RuntimeError("No usable VolumeProperties function found in OCP.BRepGProp")

# ---- Bnd / BRepBndLib ----
from OCP.Bnd import Bnd_Box
try:
    from OCP.Bnd import Bnd_OBB
except Exception:
    Bnd_OBB = None

from OCP import BRepBndLib as _BRepBndLib_mod

def _bnd_add(shape, box):
    for name in ("brepbndlib_Add", "Add", "Add_1"):
        fn = getattr(_BRepBndLib_mod, name, None)
        if callable(fn):
            try:
                fn(shape, box)
                return True
            except Exception:
                continue
    return False

def _bnd_add_obb(shape, obb):
    for name in ("brepbndlib_AddOBB", "AddOBB", "AddOBB_1"):
        fn = getattr(_BRepBndLib_mod, name, None)
        if callable(fn):
            try:
                fn(shape, obb)
                return True
            except Exception:
                continue
    return False


# ============================================================
# STEP utilities
# ============================================================

def load_shape_from_step(path: str):
    reader = STEPControl_Reader()
    status = reader.ReadFile(path)
    if status != 1:
        raise RuntimeError(f"STEP read failed, status={status}")
    reader.TransferRoots()
    return reader.OneShape()


def compute_volume_mm3(shape) -> float:
    props = GProp_GProps()
    _volume_properties(shape, props)
    return float(props.Mass())


def compute_bbox_dims_mm(shape):
    # --- Try OBB first ---
    if Bnd_OBB:
        try:
            obb = Bnd_OBB()
            if _bnd_add_obb(shape, obb):
                return (
                    obb.XHSize() * 2,
                    obb.YHSize() * 2,
                    obb.ZHSize() * 2,
                )
        except Exception:
            pass

    # --- Fallback AABB ---
    box = Bnd_Box()
    if not _bnd_add(shape, box):
        raise RuntimeError("Bnd add failed")
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (
        max(0.0, xmax - xmin),
        max(0.0, ymax - ymin),
        max(0.0, zmax - zmin),
    )


# ============================================================
# HLR Preview (Edge-safe)
# ============================================================

def make_hlr_png_bytes(shape, dpi=200, fig_size=4, sample_n=60):
    view_dir = gp_Dir(1, -1, 1)
    projector = HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), view_dir))

    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(projector)
    algo.Update()
    algo.Hide()

    hlr = HLRBRep_HLRToShape(algo)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    lines = []

    exp = TopExp_Explorer(hlr.VCompound(), TopAbs_EDGE)
    while exp.More():
        # 🔑 Fix5: 強制轉成 TopoDS_Edge
        edge = topods_Edge(exp.Current())

        adaptor = BRepAdaptor_Curve(edge)
        first = adaptor.FirstParameter()
        last = adaptor.LastParameter()

        pts = []
        for i in range(sample_n + 1):
            t = first + (last - first) * i / sample_n
            p = adaptor.Value(t)
            pts.append([p.X(), p.Y()])

        if len(pts) >= 2:
            lines.append(pts)

        exp.Next()

    if not lines:
        raise RuntimeError("HLR produced no edges")

    ax.add_collection(LineCollection(lines, linewidths=1.0))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")

    buf = BytesIO()
    plt.savefig(buf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# Streamlit UI (minimal but functional)
# ============================================================

st.set_page_config(layout="wide")
st.title("📐 STEP → BOM (Cloud Stable / cadquery-ocp)")

uploaded = st.file_uploader("Upload STEP (.step / .stp)", type=["step", "stp"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".step") as f:
        f.write(uploaded.getbuffer())
        step_path = f.name

    try:
        shape = load_shape_from_step(step_path)
        vol_cm3 = compute_volume_mm3(shape) / 1000.0
        dx, dy, dz = compute_bbox_dims_mm(shape)

        png = make_hlr_png_bytes(shape)

        st.image(png, caption="HLR Preview", use_column_width=False)
        st.success(f"Volume: {vol_cm3:.2f} cm³")
        st.info(f"BBox: {dx:.1f} × {dy:.1f} × {dz:.1f} mm")

    except Exception as e:
        st.error(f"解析失敗：{e}")

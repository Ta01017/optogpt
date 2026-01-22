#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, time, hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import numpy as np
import tidy3d as td
from tidy3d import web

# ---------------------------
# 0) utils
# ---------------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def wl_nm_to_freqs(wl_nm: np.ndarray) -> np.ndarray:
    return td.C_0 / (wl_nm * 1e-9)

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

# ---------------------------
# 1) Meta-GPT style METASTRINGS
#    <STACK>|<LATTICE>|<GEOM>
# ---------------------------
_STACK_RE = re.compile(r"^([A-Za-z0-9]+)\.(Inf|[0-9]+nm)$")
_LAT_RE   = re.compile(r"^([0-9]+)x([0-9]+)\.P=([0-9]+)nm$")
_CELL_RE  = re.compile(r"^(?:\(\)|([A-Z])\.([0-9]+)\.([0-9]+))$")

SUPPORTED_SHAPES = ["C", "S", "R"]  # 先最小集合，后续可加 X/L/H 等

METALS = ["Al", "Au", "Ag", "Cr", "Cu", "Ti"]
DIELS  = ["AlN", "Al2O3", "TiO2", "Si", "SiN", "SiO2"]

METAL_THK_NM = list(range(20, 75, 5))      # 20..70 step 5
DIEL_THK_NM  = [40, 90, 140, 190, 240]     # coarse grid
PERIOD_NM    = [200, 250, 300, 350, 400, 450, 500, 600]

def make_metastring(stack: List[Tuple[str, str]], nr: int, nc: int, period_nm: int, rows: List[List[str]]) -> str:
    seg_stack = "/".join([f"{m}.{t}" for m, t in stack])
    seg_lat = f"{nr}x{nc}.P={period_nm}nm"
    seg_geom = ";".join(["_".join(r) for r in rows])
    return f"{seg_stack}|{seg_lat}|{seg_geom}"

def parse_metastring(ms: str) -> Dict[str, Any]:
    segs = ms.split("|")
    if len(segs) != 3:
        raise ValueError("metastring must have 3 segments split by '|'")
    seg_stack, seg_lat, seg_geom = segs

    layers = []
    for item in seg_stack.split("/"):
        m = _STACK_RE.match(item.strip())
        if not m:
            raise ValueError(f"bad stack token: {item}")
        layers.append((m.group(1), m.group(2)))  # (mat, thk)

    m = _LAT_RE.match(seg_lat.strip())
    if not m:
        raise ValueError(f"bad lattice segment: {seg_lat}")
    nr, nc, period_nm = int(m.group(1)), int(m.group(2)), int(m.group(3))

    row_strs = [r.strip() for r in seg_geom.split(";") if r.strip()]
    if len(row_strs) != nr:
        raise ValueError(f"geom rows != nr (got {len(row_strs)}, expected {nr})")

    cells = []
    for r in row_strs:
        cols = [c.strip() for c in r.split("_")]
        if len(cols) != nc:
            raise ValueError(f"geom cols != nc (got {len(cols)}, expected {nc})")
        prow = []
        for c in cols:
            mc = _CELL_RE.match(c)
            if not mc:
                raise ValueError(f"bad cell token: {c}")
            if c == "()":
                prow.append({"type": "empty"})
            else:
                shape = mc.group(1)
                p1 = int(mc.group(2)) / 100.0
                p2 = int(mc.group(3)) / 100.0
                prow.append({"type": shape, "p1": p1, "p2": p2})
        cells.append(prow)

    return {"layers": layers, "nr": nr, "nc": nc, "period_nm": period_nm, "cells": cells}

# ---------------------------
# 2) Random generator (MDM + pattern lattice)
# ---------------------------
@dataclass
class GenCfg:
    nr_min: int = 1
    nr_max: int = 4
    nc_min: int = 1
    nc_max: int = 4
    p_empty: float = 0.15
    pmin: int = 10
    pmax: int = 90

def sample_mdm_stack(rng: np.random.Generator) -> List[Tuple[str, str]]:
    m1 = str(rng.choice(METALS))
    diel = str(rng.choice(DIELS))
    m2 = str(rng.choice(METALS))
    diel_thk = int(rng.choice(DIEL_THK_NM))
    m_thk = int(rng.choice(METAL_THK_NM))
    return [(m1, "Inf"), (diel, f"{diel_thk}nm"), (m2, f"{m_thk}nm")]

def sample_cell_token(rng: np.random.Generator, cfg: GenCfg) -> str:
    if rng.random() < cfg.p_empty:
        return "()"
    shape = str(rng.choice(SUPPORTED_SHAPES))
    p1 = int(rng.integers(cfg.pmin, cfg.pmax + 1))
    p2 = int(rng.integers(cfg.pmin, cfg.pmax + 1))
    return f"{shape}.{p1}.{p2}"

def sample_metastring(rng: np.random.Generator, cfg: GenCfg) -> str:
    nr = int(rng.integers(cfg.nr_min, cfg.nr_max + 1))
    nc = int(rng.integers(cfg.nc_min, cfg.nc_max + 1))
    period_nm = int(rng.choice(PERIOD_NM))
    stack = sample_mdm_stack(rng)
    rows = [[sample_cell_token(rng, cfg) for _ in range(nc)] for __ in range(nr)]
    return make_metastring(stack, nr, nc, period_nm, rows)

# ---------------------------
# 3) Materials (placeholder; replace with dispersive models later)
# ---------------------------
def medium_from_name(name: str) -> td.Medium:
    # Quick bootstrap:
    # - metals: lossy via conductivity
    # - dielectrics: constant n
    if name in METALS:
        return td.Medium(permittivity=1.0, conductivity=5e6)
    n_map = {
        "SiO2": 1.45, "SiN": 2.0, "Si": 3.48, "TiO2": 2.4,
        "Al2O3": 1.75, "AlN": 2.1
    }
    n = float(n_map.get(name, 1.6))
    return td.Medium(permittivity=n**2)

# ---------------------------
# 4) Build Tidy3D Simulation (unit-cell periodic; z PML)
# ---------------------------
@dataclass
class SimCfg:
    wl0_nm: float = 400.0
    wl1_nm: float = 2600.0
    n_wl: int = 121

    air_above_um: float = 1.0
    air_below_um: float = 0.2
    substrate_thk_um: float = 1.5  # for "Inf" layer inside box

    pattern_h_um: float = 0.07     # 70 nm
    min_steps_per_wvl: int = 18
    run_time: float = 2e-12

def cell_center(i: int, n: int, pitch_um: float) -> float:
    L = n * pitch_um
    return -0.5 * L + (i + 0.5) * pitch_um

def build_structures(parsed: Dict[str, Any], simcfg: SimCfg, with_pattern: bool) -> Tuple[List[td.Structure], Tuple[float, float, float]]:
    nr, nc = parsed["nr"], parsed["nc"]
    pitch_um = parsed["period_nm"] * 1e-3
    Lx, Ly = nc * pitch_um, nr * pitch_um

    # Stack thicknesses
    layer_thk = []
    for mat, thk in parsed["layers"]:
        if thk == "Inf":
            layer_thk.append(simcfg.substrate_thk_um)
        else:
            layer_thk.append(float(thk.replace("nm", "")) * 1e-3)

    stack_total = float(sum(layer_thk))
    z_size = simcfg.air_above_um + simcfg.pattern_h_um + stack_total + simcfg.air_below_um

    # We set z=0 at center of sim; build downward from top-air bottom plane.
    z_top = 0.5 * z_size
    z_cursor = z_top - simcfg.air_above_um - simcfg.pattern_h_um  # start at top of stack (just under pattern layer)

    structures: List[td.Structure] = []

    # Add stack blocks (infinite in x/y)
    for (mat, thk), thk_um in zip(parsed["layers"], layer_thk):
        zc = z_cursor - 0.5 * thk_um
        structures.append(
            td.Structure(
                geometry=td.Box(center=(0, 0, zc), size=(td.inf, td.inf, thk_um)),
                medium=medium_from_name(mat),
            )
        )
        z_cursor -= thk_um

    if with_pattern:
        # Pattern medium: use top metal (last layer in stack string, typical MDM top)
        top_metal = parsed["layers"][-1][0]
        pat_med = medium_from_name(top_metal)

        # Pattern sits in the "pattern layer slab" just above stack
        z_pat_center = z_top - simcfg.air_above_um - 0.5 * simcfg.pattern_h_um

        for ri in range(nr):
            for ci in range(nc):
                c = parsed["cells"][ri][ci]
                if c["type"] == "empty":
                    continue
                cx = cell_center(ci, nc, pitch_um)
                cy = cell_center(ri, nr, pitch_um)

                # convert relative params to absolute sizes within cell
                sx = max(0.05, min(0.95, float(c["p1"]))) * pitch_um
                sy = max(0.05, min(0.95, float(c["p2"]))) * pitch_um

                if c["type"] == "C":
                    r = 0.5 * min(sx, sy)
                    geom = td.Cylinder(center=(cx, cy, z_pat_center), radius=r, length=simcfg.pattern_h_um, axis=2)
                elif c["type"] in ("S", "R"):
                    geom = td.Box(center=(cx, cy, z_pat_center), size=(sx, sy, simcfg.pattern_h_um))
                else:
                    continue

                structures.append(td.Structure(geometry=geom, medium=pat_med))

    return structures, (Lx, Ly, z_size)

def build_sim(parsed: Dict[str, Any], simcfg: SimCfg, with_pattern: bool) -> td.Simulation:
    wl_nm = np.linspace(simcfg.wl0_nm, simcfg.wl1_nm, simcfg.n_wl)
    freqs = wl_nm_to_freqs(wl_nm)

    structures, size = build_structures(parsed, simcfg, with_pattern=with_pattern)
    z_size = size[2]

    # PlaneWave (direction '+'/'-'), GaussianPulse source_time  —— doc API :contentReference[oaicite:4]{index=4}
    src = td.PlaneWave(
        center=(0, 0, 0.5 * z_size - 0.2),
        size=(td.inf, td.inf, 0),
        direction="-",
        pol_angle=0.0,
        source_time=td.GaussianPulse(freq0=float(np.mean(freqs)), fwidth=float(freqs.max() - freqs.min())),
    )

    # FluxMonitor + data access by name -> .flux  —— doc API :contentReference[oaicite:5]{index=5}
    refl = td.FluxMonitor(center=(0, 0, 0.5 * z_size - 0.35), size=(td.inf, td.inf, 0), freqs=freqs, name="refl")
    tran = td.FluxMonitor(center=(0, 0, -0.5 * z_size + 0.35), size=(td.inf, td.inf, 0), freqs=freqs, name="tran")

    # BoundarySpec explicit periodic x/y, pml z —— doc style example :contentReference[oaicite:6]{index=6}
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.periodic(),
        y=td.Boundary.periodic(),
        z=td.Boundary.pml(),
    )

    sim = td.Simulation(
        size=size,
        medium=td.Medium(permittivity=1.0),
        structures=structures,
        sources=[src],
        monitors=[refl, tran],
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=simcfg.min_steps_per_wvl),
        boundary_spec=boundary_spec,
        run_time=simcfg.run_time,
    )
    return sim

# ---------------------------
# 5) Run two sims (ref + struct) and compute R/T/A
# ---------------------------
def run_one(ms: str, simcfg: SimCfg, folder_name: str, out_dir: str) -> Dict[str, Any]:
    parsed = parse_metastring(ms)
    sim_ref = build_sim(parsed, simcfg, with_pattern=False)
    sim_st  = build_sim(parsed, simcfg, with_pattern=True)

    key = sha1(ms)[:10]
    ref_path = os.path.join(out_dir, f"ref_{key}.hdf5")
    st_path  = os.path.join(out_dir, f"st_{key}.hdf5")

    # web.run returns SimulationData —— doc API :contentReference[oaicite:7]{index=7}
    data_ref = web.run(sim_ref, task_name=f"ref_{key}", folder_name=folder_name, path=ref_path)
    data_st  = web.run(sim_st,  task_name=f"st_{key}",  folder_name=folder_name, path=st_path)

    wl_nm = np.linspace(simcfg.wl0_nm, simcfg.wl1_nm, simcfg.n_wl)

    refl_s = np.array(data_st["refl"].flux, dtype=np.float64)
    tran_s = np.array(data_st["tran"].flux, dtype=np.float64)
    tran_r = np.array(data_ref["tran"].flux, dtype=np.float64)

    inc = np.maximum(np.abs(tran_r), 1e-12)
    R = np.abs(refl_s) / inc
    T = np.abs(tran_s) / inc
    A = 1.0 - R - T

    # validity (simple)
    energy_err = float(np.max(np.abs((R + T + A) - 1.0)))
    valid = bool(np.all(np.isfinite(R)) and np.all(np.isfinite(T)) and np.all(np.isfinite(A)) and (energy_err < 0.07))

    return {
        "metastring": ms,
        "parsed": parsed,
        "response": {
            "wl_nm": wl_nm.tolist(),
            "R": clip01(R).tolist(),
            "T": clip01(T).tolist(),
            "A": clip01(A).tolist(),
            "energy_err_max": energy_err,
        },
        "valid": valid,
        "invalid_reason": "ok" if valid else f"energy_or_nan (energy_err_max={energy_err:.3f})",
        "sim_meta": asdict(simcfg),
    }

# ---------------------------
# 6) Dataset loop + cache
# ---------------------------
@dataclass
class DatasetCfg:
    out_dir: str = "./tidy3d_metagpt_mdm_dataset"
    folder_name: str = "mdm_dataset"
    n_samples: int = 50
    seed: int = 42
    cache: bool = True
    log_every: int = 1

def main():
    dcfg = DatasetCfg()
    gcfg = GenCfg()
    scfg = SimCfg(n_wl=121)  # 免费额度先粗采样

    ensure_dir(dcfg.out_dir)
    cache_dir = os.path.join(dcfg.out_dir, "cache")
    ensure_dir(cache_dir)
    out_jsonl = os.path.join(dcfg.out_dir, "dataset.jsonl")

    rng = np.random.default_rng(dcfg.seed)
    n_valid = 0

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(dcfg.n_samples):
            ms = sample_metastring(rng, gcfg)
            key = sha1(ms)
            cache_path = os.path.join(cache_dir, f"{key}.json")

            if dcfg.cache and os.path.exists(cache_path):
                sample = json.load(open(cache_path, "r", encoding="utf-8"))
            else:
                t0 = time.time()
                try:
                    sample = run_one(ms, scfg, dcfg.folder_name, dcfg.out_dir)
                except Exception as e:
                    sample = {
                        "metastring": ms,
                        "valid": False,
                        "invalid_reason": f"exception:{type(e).__name__}:{str(e)[:200]}",
                    }
                sample["id"] = i
                sample["hash"] = key
                sample["dt_sec"] = round(time.time() - t0, 2)

                if dcfg.cache:
                    json.dump(sample, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if sample.get("valid", False):
                n_valid += 1

            if (i + 1) % dcfg.log_every == 0:
                print(f"[{i+1}/{dcfg.n_samples}] valid={n_valid} last={sample.get('invalid_reason')} dt={sample.get('dt_sec','-')}s")

    summary = {"n_samples": dcfg.n_samples, "valid": n_valid, "valid_rate": n_valid / max(1, dcfg.n_samples)}
    json.dump(summary, open(os.path.join(dcfg.out_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("DONE:", summary)

if __name__ == "__main__":
    main()

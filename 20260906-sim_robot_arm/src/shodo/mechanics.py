"""Independent native-rod fixtures and quantitative constitutive checks."""

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from shodo.artifacts import provenance
from shodo.cable import add_cables
from shodo.config import free_hair_bundle


def fixture(config, timestep=0.0001):
    root = ET.Element("mujoco", model="clamped brush fixture")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(
        root, "option", timestep=str(timestep), integrator="implicitfast", gravity="0 0 0"
    )
    world = ET.SubElement(root, "worldbody")
    tool = ET.SubElement(world, "body", name="brush", pos="0 0 .16", quat="0 1 0 0")
    ET.SubElement(tool, "site", name="tip", pos="0 0 .16")
    add_cables(root, tool, config)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, mujoco.MjData(model)


def cantilever(
    *, bundles=7, segments=6, timestep=0.0001, total_force=0.0005, duration=3.0, recovery=0.0
):
    config = free_hair_bundle(bundles=bundles, segments=segments)
    model, data = fixture(config, timestep)
    mujoco.mj_forward(model, data)
    sites = [model.site(f"bristle{i}_S_last").id for i in range(bundles)]
    bodies = [model.site_bodyid[site] for site in sites]
    initial = data.site_xpos[sites].copy()
    force = np.array([total_force / bundles, 0.0, 0.0])
    zero = np.zeros(3)
    history = []
    start = time.perf_counter()
    sample_every = max(1, round(0.005 / timestep))
    loaded_displacement = None
    for i in range(round((duration + recovery) / timestep)):
        data.qfrc_applied[:] = 0
        if i < round(duration / timestep):
            for site, body in zip(sites, bodies, strict=True):
                mujoco.mj_applyFT(
                    model, data, force, zero, data.site_xpos[site], body, data.qfrc_applied
                )
        elif loaded_displacement is None:
            mujoco.mj_forward(model, data)
            loaded_displacement = float((data.site_xpos[sites, 0] - initial[:, 0]).mean())
        mujoco.mj_step(model, data)
        if i % sample_every == 0:
            history.append([data.time, float((data.site_xpos[sites, 0] - initial[:, 0]).mean())])
    mujoco.mj_forward(model, data)
    displacement = float((data.site_xpos[sites, 0] - initial[:, 0]).mean())
    if loaded_displacement is None:
        loaded_displacement = displacement
    rigidity = bundles * config.young_modulus * np.pi * config.rod_radius**4 / 4
    mass = float(model.body_mass.sum())
    expected_mass = 1000 * 1300 * np.pi * (75e-6) ** 2 * 0.03
    # The first finite segment is clamped; bounds bracket its effective free length.
    lower = total_force * (0.03 * (segments - 1) / segments) ** 3 / (3 * rigidity)
    upper = total_force * 0.03**3 / (3 * rigidity)
    # Small-angle discrete beam: each joint contributes F * lever² * segment_length / EI.
    # This accounts for the fixed first segment; slight bundle splay is not in this estimate.
    discrete = (
        total_force * (0.03 / segments) ** 3 * sum(j * j for j in range(1, segments)) / rigidity
    )
    report = {
        "bundles": bundles,
        "segments": segments,
        "timestep": timestep,
        "load_n": total_force,
        "displacement_m": loaded_displacement,
        "duration_s": duration,
        "recovery_s": recovery,
        "recovered_displacement_m": displacement if recovery else None,
        "discrete_beam_m": discrete,
        "discrete_relative_error": abs(loaded_displacement / discrete - 1),
        "beam_lower_m": lower,
        "beam_upper_m": upper,
        "mass_kg": mass,
        "expected_mass_kg": expected_mass,
        "rigidity_nm2": rigidity,
        "warnings": data.warning.number.tolist(),
        "seconds": time.perf_counter() - start,
        "config": asdict(config),
    }
    return report, np.asarray(history)


def audit(output="runs/v2/mechanics"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results, curves = [], {}
    for segments in (3, 6, 12, 24):
        for dt in (0.0001, 0.00005):
            report, curve = cantilever(segments=segments, timestep=dt)
            results.append(report)
            curves[f"segments{segments}-dt{dt}"] = curve
            print(json.dumps({k: v for k, v in report.items() if k != "config"}), flush=True)
    for bundles in (7, 19, 37):
        report, curve = cantilever(bundles=bundles, timestep=0.00005)
        results.append(report)
        curves[f"bundles{bundles}"] = curve
        print(json.dumps({k: v for k, v in report.items() if k != "config"}), flush=True)
    for load in (0.00025, 0.001):
        report, curve = cantilever(total_force=load, recovery=2.0)
        results.append(report)
        curves[f"load{load}"] = curve
        print(json.dumps({k: v for k, v in report.items() if k != "config"}), flush=True)
    checks = {
        "no_solver_warnings": all(not any(r["warnings"]) for r in results),
        "mass_preserved": all(
            abs(r["mass_kg"] / r["expected_mass_kg"] - 1) < 1e-6 for r in results
        ),
        "discrete_beam_agreement_1pct": all(r["discrete_relative_error"] < 0.01 for r in results),
        "elastic_recovery_1um": all(
            abs(r["recovered_displacement_m"]) < 1e-6 for r in results if r["recovery_s"]
        ),
    }
    (output / "cantilever.json").write_text(
        json.dumps(
            {
                "results": results,
                "checks": checks,
                "passed": all(checks.values()),
                "provenance": provenance(),
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(output / "cantilever.npz", **curves)
    return results


if __name__ == "__main__":
    audit()

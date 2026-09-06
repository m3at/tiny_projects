"""Paired timings of equivalent IK prerequisites, not whole-controller speedups."""

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from shodo.artifacts import provenance
from shodo.config import SimConfig, load_config
from shodo.robot import Panda


def main():
    results = {}
    for name, config in (
        ("reduced", SimConfig()),
        ("native", load_config(Path("experiments/cable.toml"))),
    ):
        robot = Panda(config.timestep, config.brush)
        robot.reset(np.array([0.5, 0, 0.025]), np.zeros(3))
        data = mujoco.MjData(robot.model)
        data.qpos[:] = robot.data.qpos
        jac = np.zeros((6, robot.model.nv))

        def pipeline(full, robot=robot, data=data, jac=jac):
            if full:
                mujoco.mj_forward(robot.model, data)
            else:
                mujoco.mj_kinematics(robot.model, data)
                mujoco.mj_comPos(robot.model, data)
            mujoco.mj_jacSite(robot.model, data, jac[:3], jac[3:], robot.site)

        pipeline(True)
        expected = jac.copy()
        position = data.site_xpos[robot.site].copy()
        pipeline(False)
        np.testing.assert_array_equal(jac, expected)
        np.testing.assert_array_equal(data.site_xpos[robot.site], position)
        timings = {"full_forward": [], "kinematics_com": []}
        rng = np.random.default_rng(7)
        for _ in range(7):
            for full in rng.permutation([True, False]):
                start = time.perf_counter()
                for _ in range(1000):
                    pipeline(full)
                timings["full_forward" if full else "kinematics_com"].append(
                    (time.perf_counter() - start) * 1000
                )
        medians = {k: float(np.median(v)) for k, v in timings.items()}
        results[name] = {
            "trials_microseconds_per_call": timings,
            "median_microseconds_per_call": medians,
            "pipeline_speedup": medians["full_forward"] / medians["kinematics_com"],
            "dofs": robot.model.nv,
        }
    report = {
        "scope": "Warm fixed air pose; prerequisite stages plus site Jacobian only; host/load dependent",
        "provenance": provenance(),
        "results": results,
    }
    output = Path("runs/v2/ik-benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

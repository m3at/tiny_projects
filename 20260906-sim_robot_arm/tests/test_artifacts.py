import json

import numpy as np
from PIL import Image

from shodo.artifacts import save_rollout
from shodo.env import HISTORY_COLUMNS


def test_decimal_experiment_names_do_not_overwrite_each_other(tmp_path):
    frame = Image.new("RGB", (8, 8), "white")
    history = np.zeros((1, len(HISTORY_COLUMNS)))
    for name in ("dt-0.0001-04e00", "dt-0.0002-06c38"):
        save_rollout(tmp_path / name, ({"case": name}, history, frame, [frame]))
        for suffix in (".png", ".npz", ".gif", "-scene.png", ".json"):
            assert (tmp_path / f"{name}{suffix}").is_file()
        assert json.loads((tmp_path / f"{name}.json").read_text())["metrics"]["case"] == name
        with np.load(tmp_path / f"{name}.npz") as archive:
            np.testing.assert_array_equal(archive["history"], history)
            assert archive["columns"].tolist() == list(HISTORY_COLUMNS)
    assert len(list(tmp_path.iterdir())) == 10

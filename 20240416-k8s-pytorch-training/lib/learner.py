import logging

# Suppress tensorflow useless verbosity
# Why is lightning even shipping tf in the first place?
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from listwise_utils.data_utils import upload_to_gcp  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402

logger = logging.getLogger("base." + __file__)


class CheckpointCopyGCP(ModelCheckpoint):
    transform_uploaded: bool = False

    def on_validation_end(self, *args, **kwargs):
        super().on_validation_end(*args, **kwargs)

        logger.debug("Uploading training artefacts at epoch end")

        to_upload = []

        # TODO: copy top-k?
        for f in Path("./checkpoints").glob("*/last.ckpt"):
            to_upload.append((f, "checkpoints"))

        for f in Path("./training_logs").rglob("events*"):
            if f.is_file():
                to_upload.append((f, "lightning_logs"))

        for f in Path("./training_logs").rglob("metrics.csv"):
            if f.is_file():
                to_upload.append((f, "lightning_logs"))

        if not self.transform_uploaded:
            _path_transform = Path.cwd() / "power_transform.onnx"
            to_upload.append((_path_transform, None))
            self.transform_uploaded = True

        upload_to_gcp(to_upload, gcp_dir="onnx_models/v0")

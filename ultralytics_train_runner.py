from __future__ import annotations

import multiprocessing
import sys


def run() -> None:
    """Run the normal Ultralytics CLI with its unrelated YOLO26 AMP probe disabled."""
    import torch
    import ultralytics.engine.trainer as trainer_module
    from ultralytics.cfg import entrypoint

    amp_requested = any(arg.lower() == "amp=true" for arg in sys.argv[1:])
    if torch.cuda.is_available() and amp_requested:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"MYAUTOTRAIN_AMP=enabled ({gpu_name})", flush=True)

        def use_amp_for_current_model(_model) -> bool:
            return True

        trainer_module.check_amp = use_amp_for_current_model
    elif torch.cuda.is_available():
        print("MYAUTOTRAIN_AMP=disabled for stable rectangular training", flush=True)

    from ultralytics.models.yolo.detect.train import DetectionTrainer

    original_preprocess_batch = DetectionTrainer.preprocess_batch

    def report_first_batch_shape(self, batch):
        batch = original_preprocess_batch(self, batch)
        if not getattr(self, "_myautotrain_shape_reported", False):
            tensor = batch.get("img")
            if tensor is not None and tensor.ndim == 4:
                print(
                    f"ACTUAL_TRAIN_TENSOR={tensor.shape[3]}x{tensor.shape[2]} "
                    f"BATCH={tensor.shape[0]}",
                    flush=True,
                )
            self._myautotrain_shape_reported = True
        return batch

    DetectionTrainer.preprocess_batch = report_first_batch_shape

    entrypoint()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run()

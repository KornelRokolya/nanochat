"""Small experiment logging helpers for Nanochat research runs."""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

import torch


class TeeStream:
    def __init__(self, original, file_handle):
        self.original = original
        self.file_handle = file_handle

    def write(self, data):
        self.original.write(data)
        self.file_handle.write(data)
        self.file_handle.flush()
        return len(data)

    def flush(self):
        self.original.flush()
        self.file_handle.flush()

    def isatty(self):
        return getattr(self.original, "isatty", lambda: False)()

    def fileno(self):
        return self.original.fileno()


class ExperimentLogger:
    """Owns config, text log, metrics CSV, event CSV, and raw parameter paths."""

    METRIC_FIELDS = [
        "step", "num_iterations", "pct_done", "training_loss", "smooth_training_loss",
        "lrm", "total_batch_size", "device_batch_size", "grad_accum_steps", "dt_ms",
        "tok_per_sec", "bf16_mfu", "epoch", "pq_idx", "rg_idx", "training_time_s",
        "process_wall_time_s", "training_tokens_so_far", "total_training_flops",
        "model_n_layer", "model_n_embd", "model_n_head", "model_num_params",
    ]
    EVENT_FIELDS = [
        "training_time_s", "process_wall_time_s", "step", "event", "details_json"
    ]

    def __init__(self, experiment_dir: str, master_process: bool, append: bool = False):
        self.experiment_dir = experiment_dir
        self.master_process = master_process
        self._log_handle = None
        self._metrics_handle = None
        self._events_handle = None
        self.metrics_writer = None
        self.events_writer = None
        self._old_stdout = None
        self._old_stderr = None
        if not master_process:
            return

        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(self.raw_root, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        mode = "a" if append else "w"
        self._log_handle = open(os.path.join(experiment_dir, "log.txt"), mode, encoding="utf-8", buffering=1)
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeStream(sys.stdout, self._log_handle)
        sys.stderr = TeeStream(sys.stderr, self._log_handle)

        metrics_path = os.path.join(experiment_dir, "metrics.csv")
        events_path = os.path.join(experiment_dir, "events.csv")
        metrics_exists = append and os.path.exists(metrics_path)
        events_exists = append and os.path.exists(events_path)
        self._metrics_handle = open(metrics_path, "a" if metrics_exists else "w", newline="", encoding="utf-8", buffering=1)
        self._events_handle = open(events_path, "a" if events_exists else "w", newline="", encoding="utf-8", buffering=1)
        self.metrics_writer = csv.DictWriter(self._metrics_handle, fieldnames=self.METRIC_FIELDS, delimiter=";")
        self.events_writer = csv.DictWriter(self._events_handle, fieldnames=self.EVENT_FIELDS, delimiter=";")
        if not metrics_exists:
            self.metrics_writer.writeheader()
        if not events_exists:
            self.events_writer.writeheader()

    @property
    def raw_root(self):
        return os.path.join(self.experiment_dir, "raw_params")

    @property
    def checkpoint_dir(self):
        return os.path.join(self.experiment_dir, "checkpoints")

    def save_config(self, config: dict[str, Any]):
        if not self.master_process:
            return
        with open(os.path.join(self.experiment_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

    def metric(self, row: dict[str, Any]):
        if self.master_process:
            self.metrics_writer.writerow(row)

    def event(self, training_time_s: float, process_wall_time_s: float, step: int, event: str, details: dict[str, Any]):
        if self.master_process:
            self.events_writer.writerow({
                "training_time_s": f"{training_time_s:.9f}",
                "process_wall_time_s": f"{process_wall_time_s:.9f}",
                "step": step,
                "event": event,
                "details_json": json.dumps(details, separators=(",", ":"), default=str),
            })

    def close(self):
        if not self.master_process:
            return
        if self._old_stdout is not None:
            sys.stdout, sys.stderr = self._old_stdout, self._old_stderr
        for handle in (self._metrics_handle, self._events_handle, self._log_handle):
            if handle is not None:
                handle.flush()
                handle.close()


def _safe_param_filename(name: str) -> str:
    return name.replace(".", "_").replace("/", "_").replace("\\", "_")


@torch.no_grad()
def save_model_as_raw_params(model, output_directory: str, enabled: bool = True) -> None:
    """Save every learned parameter as little-endian float32 plus a manifest."""
    if not enabled:
        return
    import numpy as np

    os.makedirs(output_directory, exist_ok=True)
    manifest_path = os.path.join(output_directory, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["parameter_name", "dtype", "shape", "raw_file"])
        for name, parameter in model.named_parameters():
            tensor = parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
            shape = tuple(tensor.shape)
            file_shape = shape if len(shape) != 1 else (shape[0], 1)
            filename = f"{_safe_param_filename(name)}_f32_{'x'.join(map(str, file_shape))}.raw"
            tensor.numpy().astype("<f4", copy=False).tofile(os.path.join(output_directory, filename))
            writer.writerow([name, "float32_le", "x".join(map(str, shape)), filename])

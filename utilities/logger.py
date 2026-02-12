import os
import csv
from typing import Any, Dict, Optional, Union, List, OrderedDict
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from datetime import datetime
from argparse import Namespace
from typing import NamedTuple
import logging


class LogKey(NamedTuple):
    step: int
    epoch: int
    training: int
    batch: int


class LocalLogger(Logger):
    def __init__(
            self,
            save_dir: str = "logs",
            name: str = "default",
            version: Optional[str] = None,
            rank: int = 0,
    ):
        super().__init__()
        self.rank = rank
        self.name_ = name
        self.version_ = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dirs = save_dir
        self._log_dir = os.path.join(self.save_dirs, self.name_, str(self.version_))
        os.makedirs(self._log_dir, exist_ok=True)

        self.buffer: Dict[LogKey, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return self.name_

    @property
    def version(self) -> str:
        return self.version_

    @property
    def log_dir(self) -> str:
        return self._log_dir

    @rank_zero_experiment
    def experiment(self):
        return self

    def log_hyperparams(self, params: Union[dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        # print(f"[Dummy] Logging metrics {metrics} @ Step {step}")
        pass

    def log_real(
            self,
            metrics: Dict[str, float],
            step: int,
            epoch: int,
            batch: Optional[int] = None,
            training: Optional[bool] = True,
            prefix: Optional[str] = None,
    ) -> None:
        # print(f"[Real] Logging metrics {metrics} @ Step {step}, Epoch {epoch}, Prefix: {prefix}")

        key = LogKey(step=step, epoch=epoch, batch=batch or -1, training=int(training))
        if key not in self.buffer:
            self.buffer[key] = {
                "step": step,
                "epoch": epoch,
                "batch": batch if batch is not None else -1,
                "training": int(training),
            }

        # Update with metrics
        for k, v in metrics.items():
            keyname = f"{prefix}/{k}" if prefix else k
            self.buffer[key][keyname] = float(v)

    def flush_metrics(self, stage: str) -> None:

        if not self.buffer:
            return

        # Sort steps to preserve order
        keys = sorted(self.buffer.keys(), key=lambda k: (k.step, k.epoch, k.training, k.batch))
        records = [self.buffer[key] for key in keys]

        # Dynamically infer fieldnames from all merged records
        fieldnames = sorted(set().union(*(record.keys() for record in records)))

        os.makedirs(os.path.join(self._log_dir, stage), exist_ok=True)
        log_file_path = os.path.join(self._log_dir, stage, f"metrics_rank{self.rank}.csv")
        write_header = not os.path.exists(log_file_path)
        with open(log_file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(records)

        logging.info(f"Flushed {len(records)} records to {log_file_path} for stage '{stage}'")

        self.buffer.clear()

    def finalize(self, status: str) -> None:
        pass


def setup_logging(log_level=logging.INFO, rank: int = 0, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"rank_{rank}.log")

    # Remove previous handlers if any
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            # logging.StreamHandler() if rank == 0 else logging.NullHandler(),  # Only rank 0 logs to stdout
        ]
    )

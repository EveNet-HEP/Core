import os
import torch
import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter
from typing import Any
from pathlib import Path


class PredWriter(BasePredictionWriter):
    def __init__(self, output_dir: Path, filename: str = "predictions.pt", write_interval="epoch"):
        super().__init__(write_interval=write_interval)
        self.output_dir = output_dir
        self.filename = filename
        os.makedirs(self.output_dir, exist_ok=True)

    def write_on_epoch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            predictions: Any,
            batch_indices: list[list[list[int]]],
    ) -> None:
        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, predictions)
            torch.distributed.barrier()
        else:
            gathered = [predictions]

        if not trainer.is_global_zero:
            return

        # Flatten predictions
        all_predicts = sum(gathered, [])

        save_path = os.path.join(self.output_dir, self.filename)

        torch.save(all_predicts, save_path)
        print(f"--> Saved {len(all_predicts)} predictions to {save_path}")

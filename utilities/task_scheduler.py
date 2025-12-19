import numpy as np


class ProgressiveTaskScheduler:
    def __init__(self, config, total_epochs, steps_per_epoch, model_parts=None):
        self.total_epochs = total_epochs
        self.model_parts = model_parts or {}
        self.stages = []
        epoch_start = 0

        print("--> Progressive Training:")

        for i, stage_cfg in enumerate(config["stages"]):
            epoch_length = int(round(stage_cfg["epoch_ratio"] * total_epochs))
            epoch_end = epoch_start + epoch_length

            transition_ratio = stage_cfg.get("transition_ratio", 0.0)
            transition_epochs = epoch_length * transition_ratio
            transition_start_epoch = epoch_start
            transition_end_epoch = epoch_start + transition_epochs

            # transition_start = int(transition_start_epoch * steps_per_epoch)
            # transition_end = int(transition_end_epoch * steps_per_epoch)

            self.stages.append({
                "name": stage_cfg.get("name", f"stage_{len(self.stages)}"),
                "epoch_start": epoch_start,
                "epoch_end": epoch_end,
                # "transition_start": transition_start,
                # "transition_end": transition_end,
                "transition_start_epoch": transition_start_epoch,
                "transition_end_epoch": transition_end_epoch,
                "loss_weights": stage_cfg.get("loss_weights", {}),
                "train_parameters": stage_cfg.get("train_parameters", {}),
                "freeze": stage_cfg.get("freeze", []),
                "unfreeze": stage_cfg.get("unfreeze", []),
            })

            print(f"{self.stages[-1]['name']}: {epoch_length:.0f} epochs, total {epoch_end:.0f} epochs")
            print(f"  --> Transition (epochs): {transition_start_epoch:.1f} -> {transition_end_epoch:.1f}")
            # print(f"  --> Transition (steps):  {transition_start} -> {transition_end}")

            for component in stage_cfg.get("freeze", []) + stage_cfg.get("unfreeze", []):
                if self.model_parts.get(component, None):
                    print(f"  --> Model Part: {component} ✅")
                else:
                    print(f"  --> Model Part: {component} ❌")

            epoch_start = epoch_end

        self.scheduler = None

    def get_current_stage(self, epoch):
        for stage in self.stages:
            if stage["epoch_start"] <= epoch < stage["epoch_end"]:
                return stage
        return self.stages[-1]

    # def get_transition_factor(self, step, stage):
    #     if step < stage["transition_start"]:
    #         return 0.0
    #     t = (step - stage["transition_start"]) / max(stage["transition_end"] - stage["transition_start"], 1)
    #     return 0.5 * (1 - np.cos(np.pi * np.clip(t, 0, 1)))
    #
    # def get_current_parameters(self, epoch, step):
    #     stage = self.get_current_stage(epoch)
    #     t = self.get_transition_factor(step, stage)
    #
    #     weights = {}
    #     train_parameters = {}
    #     for task, (start, end) in stage["loss_weights"].items():
    #         weights[task] = (1 - t) * start + t * end
    #     for task, (start, end) in stage["train_parameters"].items():
    #         train_parameters[task] = (1 - t) * start + t * end
    #     return {
    #         'loss_weights': weights,
    #         'train_parameters': train_parameters,
    #     }

    def _epoch_progress(self, epoch: int, batch_idx: int, batches_per_epoch: int) -> float:
        """Fractional epoch index, robust to varying batches_per_epoch."""
        den = max(int(batches_per_epoch), 1)
        frac = np.clip(batch_idx / den, 0.0, 1.0)
        return float(epoch) + float(frac)

    def _transition_factor_from_progress(self, epoch_progress: float, stage: dict) -> float:
        """
        Smooth cosine ramp within [transition_start_epoch, transition_end_epoch].
        Before the window -> 0. After the window -> 1.
        Zero-length window -> step function at the start boundary.
        """
        ts = stage["transition_start_epoch"]
        te = stage["transition_end_epoch"]
        if te <= ts:  # no ramp, instantaneous switch at ts
            return 0.0 if epoch_progress < ts else 1.0

        if epoch_progress <= ts:
            return 0.0
        if epoch_progress >= te:
            return 1.0

        t = (epoch_progress - ts) / (te - ts)
        # print(f"Transition factor: {t:.3f} (epoch_progress={epoch_progress:.3f}, ts={ts:.3f}, te={te:.3f})")
        t = np.clip(t, 0.0, 1.0)
        # Smooth cosine transition
        # print(f"Transition factor (cosine): {0.5 * (1.0 - np.cos(np.pi * t)):.3f}")
        return 0.5 * (1.0 - np.cos(np.pi * t))  # smooth

    def get_current_parameters(self, epoch: int, batch_idx: int, batches_per_epoch: int):
        """
        Call this inside training with the *current* batches_per_epoch (which may change across runs).
        """
        stage = self.get_current_stage(epoch)
        ep = self._epoch_progress(epoch, batch_idx, batches_per_epoch)
        t = self._transition_factor_from_progress(ep, stage)

        weights = {}
        train_parameters = {}
        for task, (start, end) in stage["loss_weights"].items():
            weights[task] = (1 - t) * start + t * end
        for task, (start, end) in stage["train_parameters"].items():
            train_parameters[task] = (1 - t) * start + t * end

        # print(f"Epoch {epoch}, Batch {batch_idx}/{batches_per_epoch} -> Stage: {stage['name']}, ep: {ep:.3f}")

        return {
            "loss_weights": weights,
            "train_parameters": train_parameters
        }

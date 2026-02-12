from typing import Callable

import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix, roc_curve, auc


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch.nn.functional as F

from evenet.utilities.debug_tool import time_decorator, debug_nonfinite_batch
import logging

logger = logging.getLogger(__name__)

class ClassificationMetrics:
    def __init__(self, num_classes, device, normalize=False, num_bins=100):
        self.train_matrix = None
        self.train_hist_store = None
        self.device = device
        self.num_classes = num_classes
        self.normalize = normalize
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.valid = 0
        self.total = 0
        self.l = logging.getLogger("ClassificationMetrics")

        # for logits histogram
        self.bins = np.linspace(0, 1, num_bins + 1)
        self.hist_store = np.zeros((self.num_classes, self.num_classes, num_bins), dtype=np.int64)

    def update(self, y_true: torch.Tensor, y_pred_raw: torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred_raw.argmax(dim=-1).detach().cpu().numpy()
        logits = y_pred_raw.detach().cpu()

        # Filter out ignored targets like -1 (often used for masking)
        valid = y_true >= 0
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        self.valid += valid.sum()
        self.total += len(valid)

        if len(y_true) == 0:
            return  # Skip empty updates safely

        present_labels = np.unique(np.concatenate([y_true, y_pred]))
        cm_partial = confusion_matrix(y_true, y_pred, labels=present_labels)

        for i, true_label in enumerate(present_labels):
            for j, pred_label in enumerate(present_labels):
                if true_label < self.num_classes and pred_label < self.num_classes:
                    self.matrix[true_label, pred_label] += cm_partial[i, j]

        # For Logits histogram
        probs = F.softmax(logits, dim=1).numpy()
        for true_cls in range(self.num_classes):
            mask = (y_true == true_cls)
            if not np.any(mask):
                continue
            probs_true = probs[mask]  # (N_true, num_classes)

            for pred_cls in range(self.num_classes):
                scores = probs_true[:, pred_cls]
                hist, _ = np.histogram(scores, bins=self.bins)
                self.hist_store[true_cls, pred_cls] += hist

    def reset(self, cm: bool = True, logits: bool = True):
        self.valid = 0
        self.total = 0
        if cm:
            self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        if logits:
            self.hist_store = np.zeros((self.num_classes, self.num_classes, self.bins.size - 1), dtype=np.int64)

    def reduce_across_gpus(self):
        """All-reduce across DDP workers"""
        if torch.distributed.is_initialized():
            tensor = torch.tensor(self.matrix, dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            self.matrix = tensor.cpu().numpy()

            valid_tensor = torch.tensor([self.valid], dtype=torch.long, device=self.device)
            total_tensor = torch.tensor([self.total], dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(valid_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
            self.valid = valid_tensor.item()
            self.total = total_tensor.item()

            hist_store = torch.tensor(self.hist_store, dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(hist_store, op=torch.distributed.ReduceOp.SUM)
            self.hist_store = hist_store.cpu().numpy()

    def compute(self, matrix=None):
        """Return normalized or raw matrix"""
        cm = matrix.astype(np.float64)
        if self.normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm / row_sums)
        return cm

    def assign_train_result(self, train_hist_store=None, train_matrix=None):
        self.train_hist_store = train_hist_store
        self.train_matrix = train_matrix

    def plot_cm(self, class_names, normalize=True):
        # --- Teal-Navy gradient colormap ---
        gradient_colors = ('#f0f9fa', "#4ca1af")
        cmap = mcolors.LinearSegmentedColormap.from_list("teal_navy", gradient_colors)

        # --- Text colors for contrast ---
        text_colors = {
            "train_light": "#1E6B74",
            "train_dark": "#70E1E1",
            "valid_light": "#832424",
            "valid_dark": "#FFB4A2"
        }

        cm_valid = self.compute(self.matrix) if normalize else self.matrix

        # Optional: Compute train confusion matrix
        cm_train = None
        if self.train_matrix is not None:
            cm_train = self.compute(self.train_matrix) if normalize else self.train_matrix

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_valid, interpolation="nearest", cmap=cmap)
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(self.num_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names or tick_marks, rotation=45, ha="right")
        ax.set_yticklabels(class_names or tick_marks)

        fmt = ".2f" if normalize else "d"

        for i in range(self.num_classes):
            for j in range(self.num_classes):

                cell_val = cm_valid[i, j]
                bg_val = cell_val / cm_valid.max()  # normalized background for contrast logic

                # Choose adaptive colors
                train_color = text_colors["train_dark"] if bg_val > 0.5 else text_colors["train_light"]
                valid_color = text_colors["valid_dark"] if bg_val > 0.5 else text_colors["valid_light"]

                y_offset = 0.15 if cm_train is not None else 0.0

                if cm_train is not None:
                    ax.text(j, i - y_offset, format(cm_train[i, j], fmt),
                            ha="center", va="center", color=train_color, fontsize=11)
                    ax.text(j, i + y_offset, format(cm_valid[i, j], fmt),
                            ha="center", va="center", color=valid_color, fontsize=11)
                else:
                    ax.text(j, i, format(cm_valid[i, j], fmt),
                            ha="center", va="center", color=valid_color, fontsize=11)

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix (Train in Red, Valid in Black)")
        fig.tight_layout()
        return fig

    def plot_logits(self, class_names):
        results = {}
        auc_scores = {}
        auc_scores_valid = {}
        roc_curves = {}

        # Use training store if provided, else default to self.hist_store
        train_hist_store = self.train_hist_store if self.train_hist_store is not None else self.hist_store

        # Custom color palette (max 10 classes)
        colors = [
            "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD",
            "#CD5582", "#E1BE6A", "#E1BE6A", "#E89A7A", "#EC6B2D"
        ]

        bin_centers = 0.5 * (self.bins[1:] + self.bins[:-1])
        bin_widths = np.diff(self.bins)

        for true_cls in range(self.num_classes):
            fig = plt.figure(figsize=(10, 8))

            for cls in range(self.num_classes):
                # Plot training histogram (bars)
                train_counts = train_hist_store[true_cls, cls]
                if np.sum(train_counts) > 0:
                    train_density = train_counts / (np.sum(train_counts) * bin_widths)
                    color = colors[cls % len(colors)]
                    label = f"{class_names[cls]} (True, Train)" if cls == true_cls else f"{class_names[cls]} (Train)"

                    if cls == true_cls:
                        plt.bar(bin_centers, train_density, width=bin_widths, color=color, alpha=0.85, label=None,
                                edgecolor='black')
                    else:
                        plt.bar(bin_centers, train_density, width=bin_widths, color=color, alpha=0.7,
                                label=None, edgecolor=color, fill=False)

                # Plot validation histogram (lines with markers)
                val_counts = self.hist_store[true_cls, cls]
                if np.sum(val_counts) > 0:
                    val_density = val_counts / (np.sum(val_counts) * bin_widths)
                    color = colors[cls % len(colors)]
                    label = f"{class_names[cls]} (True)" if cls == true_cls else f"{class_names[cls]}"
                    plt.plot(
                        bin_centers, val_density,
                        color=color,
                        label=label,
                        linestyle='-' if cls == true_cls else '--',
                        marker='o' if cls == true_cls else 'x',
                        linewidth=3 if cls == true_cls else 2,
                        markersize=6 if cls == true_cls else 4,
                    )

            title = f"True Class {class_names[true_cls] if class_names else true_cls}"
            plt.title(f"{title}: Softmax Score Distribution (Train vs Val)")
            plt.xlabel("Softmax Score")
            plt.ylabel("Density")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()

            results[true_cls] = fig

            plt.close(fig)

        # === 2. ROC Plot ===
        for target_cls in range(self.num_classes):
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            y_true_list = np.array([])
            y_score_list = np.array([])
            y_true_list_valid = np.array([])
            y_score_list_valid = np.array([])
            weights = np.array([])
            weights_valid = np.array([])
            bin_centers = 0.5 * (self.bins[1:] + self.bins[:-1])

            for true_cls in range(self.num_classes):
                counts = train_hist_store[true_cls, target_cls]
                counts_valid = self.hist_store[true_cls, target_cls]
                if true_cls == target_cls:
                    y_true = np.ones_like(counts)
                    y_true_valid = np.ones_like(counts_valid)
                else:
                    y_true = np.zeros_like(counts)
                    y_true_valid = np.zeros_like(counts_valid)
                y_true_list = np.concatenate([y_true_list, y_true])
                y_score_list = np.concatenate([y_score_list, bin_centers])
                weights = np.concatenate([weights, counts])
                y_true_list_valid = np.concatenate([y_true_list_valid, y_true_valid])
                y_score_list_valid = np.concatenate([y_score_list_valid, bin_centers])
                weights_valid = np.concatenate([weights_valid, counts_valid])

            if weights.sum() == 0 or weights_valid.sum() == 0:
                self.l.warning(f"Warning: No data for target class {target_cls}. Skipping ROC plot.")
                continue
            fpr, tpr, _ = roc_curve(y_true_list, y_score_list, sample_weight=weights)
            roc_auc = auc(fpr, tpr)
            auc_scores[target_cls] = roc_auc

            # Plot ROC curve
            fpr_valid, tpr_valid, _ = roc_curve(y_true_list_valid, y_score_list_valid, sample_weight=weights_valid)
            auc_valid = auc(fpr_valid, tpr_valid)
            auc_scores_valid[target_cls] = auc_valid

            plt.plot(fpr, tpr, color=colors[target_cls], lw=2, label=f"AUC = {roc_auc:.3f}")
            plt.plot(fpr_valid, tpr_valid, '--', color=colors[target_cls], lw=2, label=f"AUC[valid] = {auc_valid:.3f}")
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.grid(True)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve: {class_names[target_cls]}")
            plt.legend(loc="lower right")
            plt.tight_layout()

            roc_curves[target_cls] = fig_roc
            plt.close(fig_roc)

        return results, roc_curves, auc_scores, auc_scores_valid


@time_decorator(name="[Classification] shared_step")
def shared_step(
        target_classification: torch.Tensor,
        cls_output: torch.Tensor,
        cls_loss_fn: Callable,
        class_weight: torch.Tensor,
        loss_dict: dict,
        loss_scale: float,
        metrics: ClassificationMetrics,
        device: torch.device,
        update_metric: bool = True,
        event_weight: torch.Tensor = None,
        loss_name: str = "classification"
):
    debug_nonfinite_batch(
        {
            "logits": cls_output,
            "labels": target_classification,
            "event_weight": event_weight,
        },
        batch_dim=0, name=loss_name, logger=logger
    )

    cls_loss = cls_loss_fn(
        cls_output,
        target_classification,
        class_weight=class_weight.to(device=device),
        event_weight=event_weight,
    )

    loss = cls_loss * loss_scale
    loss_dict[loss_name] = cls_loss

    if update_metric:
        metrics.update(
            y_true=target_classification,
            y_pred_raw=cls_output
        )

    return loss


@time_decorator(name="[Classification] shared_epoch_end")
def shared_epoch_end(
        global_rank,
        metrics_valid: ClassificationMetrics,
        metrics_train: ClassificationMetrics,
        num_classes: list[str],
        logger,
        module=None,
        prefix: str = "",
):
    metrics_valid.reduce_across_gpus()
    if metrics_train:
        metrics_train.reduce_across_gpus()

    if global_rank == 0:
        metrics_valid.assign_train_result(
            train_hist_store=metrics_train.hist_store if metrics_train else None,
            train_matrix=metrics_train.matrix if metrics_train else None,
        )

        fig_cm = metrics_valid.plot_cm(class_names=num_classes)
        logger.log({
            # "classification/CM": wandb.Image(fig_cm)
            f"{prefix}classification/CM": fig_cm
        })
        plt.close(fig_cm)

        fig_logits, fig_rocs, aucs, aucs_valid = metrics_valid.plot_logits(class_names=num_classes)
        for i, class_name in enumerate(num_classes):
            logger.log({
                f"{prefix}classification/logits_{class_name}": wandb.Image(fig_logits[i])
                # f"classification/logits_{class_name}": fig_logits[i]
            })
            plt.close(fig_logits[i])

            if i in fig_rocs:
                logger.log({
                    f"{prefix}classification/roc_{class_name}": wandb.Image(fig_rocs[i])
                })
                plt.close(fig_rocs[i])

                logger.log({
                    f"{prefix}classification/train_auc_{class_name}": aucs[i]
                })

                logger.log({
                    f"{prefix}classification/valid_auc_{class_name}": aucs_valid[i]
                })

            # if module is not None:
            #     module.log(f"{prefix}classification/train_auc_{class_name}", aucs[i], prog_bar=True, sync_dist=False)
            #     module.log(f"{prefix}classification/valid_auc_{class_name}", aucs_valid[i], prog_bar=True, sync_dist=False)

    metrics_valid.reset(cm=True, logits=True)
    if metrics_train:
        metrics_train.reset(cm=True, logits=True)

from typing import Callable

import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch.nn.functional as F
from typing import Optional

from evenet.utilities.debug_tool import time_decorator, debug_nonfinite_batch
from evenet.network.loss.segmentation import hungarian_matching
from sklearn.metrics import confusion_matrix, roc_curve, auc
import logging


logger = logging.getLogger(__name__)

class SegmentationMetrics:
    def __init__(
        self,
        device,
        hist_xmin: float = 0,
        hist_xmax: float = 1.0,
        num_bins: int = 100,
        mask_threshold: float = 0.5,
        process: int = "",
        clusters_label = None,
        num_queries: int = 5,
        processes_labels = None
    ):
        self.device = device
        self.hist_xmin = hist_xmin
        self.hist_xmax = hist_xmax
        self.num_bins = num_bins
        self.mask_threshold = mask_threshold
        self.process = process
        self.clusters_label = clusters_label
        self.num_classes = len(clusters_label)
        self.num_queries = num_queries
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.matrix_number = np.zeros((self.num_queries + 1, self.num_queries + 1), dtype=np.int64)  # +1 for null class
        self.train_matrix = None
        self.train_matrix_number = None
        self.labels = np.arange(self.num_classes)
        self.labels_num = np.arange(self.num_queries + 1)  # +1 for null class
        self.labels_num_str = [str(i) for i in self.labels_num]

        # For Histogram
        self.bins = np.linspace(self.hist_xmin, self.hist_xmax, self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])

        self.score_distribution = {
            cluster_name: {
                "true-cls-true-mask": np.zeros(self.num_bins),
                "false-cls-true-mask": np.zeros(self.num_bins),
                "true-cls-false-mask": np.zeros(self.num_bins),
                "false-cls-false-mask": np.zeros(self.num_bins),
            }
            for cluster_name in clusters_label.keys()
        }

        self.cluster_matching = {
            cluster_name: {
                "true-cluster": 0,
                "pred-cls-correct-cluster": 0,
                "pred-cls-wrong-cluster": 0,
                "true-cluster-entries": 0,
                "pred-cls-correct-cluster-entries": 0,
                "pred-cls-wrong-cluster-entries": 0,
                "pred-cluster-sub-num": 0
            }
            for cluster_name in clusters_label.keys()
        }

        self.processes_labels = processes_labels if processes_labels is not None else []
        self.process_matching = {
            process_label: { num_q:
                {
                    "total-event": 0,
                    "total-cluster": 0,
                    "matched-event": 0,
                    "matched-cluster": 0,
                }
                for num_q in range(self.num_queries)
            }
            for process_label in self.processes_labels
        }

        self.train_score_distribution = None

    def update(
            self,
            y_true_mask: torch.Tensor,
            y_true_cls: torch.Tensor,
            y_pred_mask: torch.Tensor,
            y_pred_cls: torch.Tensor,
            process_cls: torch.Tensor,
    ):
        # y_pred_mask = (y_pred_mask.sigmoid() > self.mask_threshold).float() # (B, N, P)
        # max_idx = y_pred_mask.argmax(dim=1, keepdim=True)  # shape: (B, 1, P)
        # # Create a mask with 1 at max index, 0 elsewhere
        # mask = torch.zeros_like(y_pred_mask)
        # mask.scatter_(1, max_idx, 1.0)
        # y_pred_mask = y_pred_mask * mask  # Apply the mask to y_pred_mask

        pred_indices, tgt_indices = hungarian_matching(
            predict_cls = y_pred_cls,
            predict_mask = y_pred_mask,
            target_cls = y_true_cls,
            target_mask = y_true_mask.float(),
            include_cls_cost = True, # Use best combination
        )

        B, N_match = pred_indices.shape
        batch_idx = torch.arange(B, device=pred_indices.device).unsqueeze(-1)  # (B, 1)

        # print("y_pred_mask", y_pred_mask[0, 1])

        y_true_cls = y_true_cls[batch_idx, tgt_indices]
        y_true_mask = y_true_mask[batch_idx, tgt_indices]

        y_pred_cls = y_pred_cls[batch_idx, pred_indices]
        y_pred_mask = y_pred_mask[batch_idx, pred_indices]

        y_pred_mask = y_pred_mask.sigmoid()
        scores, labels = F.softmax(y_pred_cls, dim=-1).max(-1)

        # empty_mask = ~((y_pred_mask > self.mask_threshold).long().sum(dim=-1) == 0)  # shape: (B, N)
        # labels = labels * empty_mask.long()  # Set labels to 0 for empty masks
        keep = labels > 0  # Keep non-null class


        cur_masks = y_pred_mask * keep.unsqueeze(-1).float()  # Zero out masks for null class predictions
        cur_prob_masks = cur_masks * scores.unsqueeze(-1).float()  # Weight masks by their confidence scores

        if cur_masks.sum() > 0:
            # Apply class zero mask and uniqueness to the predicted mask
            max_idx = cur_prob_masks.argmax(dim=1, keepdim=True)  # shape: (B, 1, P)
            mask = torch.zeros_like(cur_masks)
            mask.scatter_(1, max_idx, 1.0)
            y_pred_mask = y_pred_mask * mask  # Apply the mask to y_pred_mask
        else:
            y_pred_mask = cur_masks


        y_true_cls_np = y_true_cls.argmax(dim=-1).flatten().cpu().numpy()
        y_pred_cls_np = labels.flatten().cpu().numpy()
        # Cluster confusion matrix
        cm_partial = confusion_matrix(y_true_cls_np, y_pred_cls_np, labels=self.labels)
        for i, true_label in enumerate(self.labels):
            for j, pred_label in enumerate(self.labels):
                if true_label < self.num_classes and pred_label < self.num_classes:
                    self.matrix[true_label, pred_label] += cm_partial[i, j]


        # Number confusion matrix
        y_true_num_np = (y_true_cls.argmax(dim=-1) > 0).sum(-1).flatten().cpu().numpy()
        y_pred_num_np = (labels > 0).sum(-1).flatten().cpu().numpy()
        cm_number = confusion_matrix(y_true_num_np, y_pred_num_np, labels=self.labels_num)
        for i, true_label in enumerate(self.labels_num):
            for j, pred_label in enumerate(self.labels_num):
                if true_label < self.num_queries + 1 and pred_label < self.num_queries + 1:
                    self.matrix_number[true_label, pred_label] += cm_number[i, j]

        # Mask distribution
        # Setup
        cls_true_filter = (y_true_cls.argmax(dim=-1) == labels)  # (B, N)
        inside_filter = (y_true_mask.float() > 0)  # (B, N, P)
        outside_filter = ~inside_filter  # (B, N, P)

        correct_masking = ((y_pred_mask > self.mask_threshold).long()) == (y_true_mask.long()) # (B, N, P)
        predict_sub_num = (y_pred_mask > self.mask_threshold).float().sum(-1) # (B, N, P)
        num_q_tensor = y_true_mask.any(dim=-1).float().sum(dim=-1)  # (B, N) → (B)
        correct_masking_full_cluster = correct_masking.all(dim=-1) # (B, N)
        correct_masking_full_cluster = correct_masking_full_cluster.float() *  y_true_mask.any(dim=-1).float() # (B, N) → (B, N)
        correct_masking_full_cluster_num = correct_masking_full_cluster.float().sum(dim=-1)  # (B)
        correct_masking_full_event = (correct_masking_full_cluster_num.long() == num_q_tensor.long()).float()

        # Loop over processes
        for i, process_label in enumerate(self.processes_labels):
            process_filter = (process_cls == i) # (B)
            # Update process matching
            for num_q in range(self.num_queries):
                num_q_filter = (num_q_tensor == num_q) # (B)
                self.process_matching[process_label][num_q]["total-event"] += (process_filter.float() * num_q_filter.float()).sum().item()
                self.process_matching[process_label][num_q]["total-cluster"] += (num_q_tensor * process_filter.float() * num_q_filter.float()).sum().item()
                self.process_matching[process_label][num_q]["matched-event"] += (process_filter.float() * num_q_filter.float() * correct_masking_full_event.float()).sum().item()
                self.process_matching[process_label][num_q]["matched-cluster"] += (process_filter.float() * num_q_filter.float() * correct_masking_full_cluster_num.float()).sum().item()

        # Loop over clusters
        for i, cluster_name in enumerate(self.clusters_label.keys()):
            cluster_filter = (y_true_cls.argmax(dim=-1) == i)  # (B, N)


            # calculate metrics:
            self.cluster_matching[cluster_name]["true-cluster"] += (cluster_filter.float()).sum().item()
            self.cluster_matching[cluster_name]["true-cluster-entries"] += (inside_filter.float() * cluster_filter.unsqueeze(-1).float()).sum().item()
            self.cluster_matching[cluster_name]["pred-cls-correct-cluster-entries"] += (correct_masking.float() * inside_filter.float() * cluster_filter.unsqueeze(-1).float() * cls_true_filter.unsqueeze(-1).float()).sum().item()
            self.cluster_matching[cluster_name]["pred-cls-wrong-cluster-entries"] += (correct_masking.float() * inside_filter.float() * cluster_filter.unsqueeze(-1).float() * (~cls_true_filter).unsqueeze(-1).float()).sum().item()
            self.cluster_matching[cluster_name]["pred-cls-correct-cluster"] += (correct_masking_full_cluster.float() * cluster_filter.float() * cls_true_filter.float()).sum().item()
            self.cluster_matching[cluster_name]["pred-cls-wrong-cluster"] += (correct_masking_full_cluster.float() * cluster_filter.float() * (~cls_true_filter).float()).sum().item()
            self.cluster_matching[cluster_name]["pred-cluster-sub-num"] += (predict_sub_num * cluster_filter.float()).sum().item()

            # Expand cluster and classification filters to (B, N, P) to match mask shape
            cluster_filter_3d = cluster_filter.unsqueeze(-1).expand_as(y_pred_mask)
            cls_true_filter_3d = cls_true_filter.unsqueeze(-1).expand_as(y_pred_mask)

            # Final filters for each category (B, N, P)
            true_cls_true_mask = cls_true_filter_3d & cluster_filter_3d & inside_filter
            false_cls_true_mask = (~cls_true_filter_3d) & cluster_filter_3d & inside_filter
            true_cls_false_mask = cls_true_filter_3d & cluster_filter_3d & outside_filter
            false_cls_false_mask = (~cls_true_filter_3d) & cluster_filter_3d & outside_filter

            for label, mask in zip(
                    ["true-cls-true-mask", "false-cls-true-mask", "true-cls-false-mask", "false-cls-false-mask"],
                    [true_cls_true_mask, false_cls_true_mask, true_cls_false_mask, false_cls_false_mask]
            ):
                # Extract values from y_pred_mask where the filter is True
                values = y_pred_mask.flatten()[mask.flatten()]  # 1D tensor of selected predicted mask values
                values = values[values > 1e-2]  # Apply threshold, 0 means no prediction
                if values.numel() == 0:
                    continue
                # Histogram using numpy
                hist, _ = np.histogram(values.detach().flatten().cpu().numpy(), bins=self.bins)

                self.score_distribution[cluster_name][label] += hist


    def reset(self, cm: bool = True):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.matrix_number = np.zeros((self.num_queries + 1, self.num_queries + 1), dtype=np.int64)  # +1 for null class
        for cluster_name in self.score_distribution.keys():
            for label in self.score_distribution[cluster_name].keys():
                self.score_distribution[cluster_name][label] = np.zeros(self.num_bins)

        for cluster_name in self.cluster_matching.keys():
            for key in self.cluster_matching[cluster_name].keys():
                self.cluster_matching[cluster_name][key] = 0

        for process_label in self.process_matching.keys():
            for num_q in self.process_matching[process_label].keys():
                self.process_matching[process_label][num_q]["total-event"] = 0
                self.process_matching[process_label][num_q]["total-cluster"] = 0
                self.process_matching[process_label][num_q]["matched-event"] = 0
                self.process_matching[process_label][num_q]["matched-cluster"] = 0


    def reduce_across_gpus(self):
        """All-reduce across DDP workers"""
        if torch.distributed.is_initialized():
            tensor = torch.tensor(self.matrix, dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            self.matrix = tensor.cpu().numpy()

            tensor_number = torch.tensor(self.matrix_number, dtype=torch.long, device=self.device)
            torch.distributed.all_reduce(tensor_number, op=torch.distributed.ReduceOp.SUM)
            self.matrix_number = tensor_number.cpu().numpy()

            # Reduce score distribution
            for cluster_name in self.score_distribution.keys():
                for label in self.score_distribution[cluster_name].keys():
                    hist_tensor = torch.tensor(
                        self.score_distribution[cluster_name][label],
                        dtype=torch.long,
                        device=self.device
                    )
                    torch.distributed.all_reduce(hist_tensor, op=torch.distributed.ReduceOp.SUM)
                    self.score_distribution[cluster_name][label] = hist_tensor.cpu().numpy()

            for cluster_name in self.cluster_matching.keys():
                for key in self.cluster_matching[cluster_name].keys():
                    tensor = torch.tensor(
                        self.cluster_matching[cluster_name][key],
                        dtype=torch.long,
                        device=self.device
                    )
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    self.cluster_matching[cluster_name][key] = tensor.item()

            for process_label in self.process_matching.keys():
                for num_q in self.process_matching[process_label].keys():
                    for key in self.process_matching[process_label][num_q].keys():
                        tensor = torch.tensor(
                            self.process_matching[process_label][num_q][key],
                            dtype=torch.long,
                            device=self.device
                        )
                        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                        self.process_matching[process_label][num_q][key] = tensor.item()

    def assign_train_result(self, train_matrix=None, train_matrix_number=None, score_distribution=None):
        self.train_matrix = train_matrix
        self.train_matrix_number = train_matrix_number
        self.train_score_distribution = score_distribution


    def compute(self, matrix=None, normalize=False):
        """Return normalized or raw matrix"""
        cm = matrix.astype(np.float64)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm / (row_sums + 1e-8))
        return cm


    def plot_cm_func(self, matrix, label, train_matrix=None, normalize=True):
        # --- Teal-Navy gradient colormap ---
        class_names = label
        gradient_colors = ('#f0f9fa', "#4ca1af")
        cmap = mcolors.LinearSegmentedColormap.from_list("teal_navy", gradient_colors)

        # --- Text colors for contrast ---
        text_colors = {
            "train_light": "#1E6B74",
            "train_dark": "#70E1E1",
            "valid_light": "#832424",
            "valid_dark": "#FFB4A2"
        }

        cm_valid = self.compute(matrix, normalize=normalize) if normalize else matrix

        # Optional: Compute train confusion matrix
        cm_train = None
        if train_matrix is not None:
            cm_train = self.compute(train_matrix, normalize=normalize) if normalize else train_matrix

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_valid, interpolation="nearest", cmap=cmap)
        plt.colorbar(im, ax=ax)

        num_classes = len(class_names)
        tick_marks = np.arange(num_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names or tick_marks, rotation=45, ha="right")
        ax.set_yticklabels(class_names or tick_marks)

        fmt = ".2f" if normalize else "d"

        for i in range(num_classes):
            for j in range(num_classes):

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
        ax.set_title("Confusion Matrix (Train in Black, Valid in Red)")
        fig.tight_layout()
        return fig

    def plot_cm(self, normalize=True):

        fig1 = self.plot_cm_func(
            matrix=self.matrix,
            label=self.clusters_label.keys(),
            train_matrix=self.train_matrix,
            normalize=normalize
        )
        fig2 = self.plot_cm_func(
            matrix=self.matrix_number,
            label=self.labels_num_str,
            train_matrix=self.train_matrix_number,
            normalize=normalize
        )

        return fig1, fig2

    def plot_score_distributions(self, cluster_name, logy=False):
        fig, ax = plt.subplots(figsize=(10, 8))
        metrics = dict()

        # Total points inside mask (used for ratio)
        total_true = (
            np.sum(self.score_distribution[cluster_name]["true-cls-true-mask"])
            + np.sum(self.score_distribution[cluster_name]["false-cls-true-mask"])
        ) + 1e-8  # avoid divide-by-zero

        # print("total_true", total_true)

        # Custom color palette (max 10 classes)
        colors = [
            "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD",
            "#CD5582", "#E1BE6A", "#E1BE6A", "#E89A7A", "#EC6B2D"
        ]

        bin_centers = 0.5 * (self.bins[1:] + self.bins[:-1])
        bin_widths = np.diff(self.bins)

        for idx, label in enumerate(["true-cls-true-mask", "false-cls-true-mask", "true-cls-false-mask", "false-cls-false-mask"]):
            hist = self.score_distribution[cluster_name][label]
            # Plot training histogram (bars)
            ratio = np.sum(hist) / total_true
            if np.sum(hist) > 0:
                density = hist / (np.sum(hist) * bin_widths)
            else:
                density = hist
            color = colors[idx % len(colors)]
            label_ratio = f"{label} (ratio={ratio:.2f})"
            ax.bar(bin_centers, density, width=bin_widths, color=color, alpha=0.85, label=label_ratio, edgecolor = 'black')
            # metrics[label] = ratio

            hist_train = self.train_score_distribution[cluster_name][label] if self.train_score_distribution is not None else None

            if hist_train is not None:
                train_density = hist_train / (np.sum(hist_train) * bin_widths) if np.sum(hist_train) > 0 else hist_train
                color = colors[idx % len(colors)]
                label_train = f"{label} (Train)"
                ax.plot(
                    bin_centers, train_density,
                    color=color,
                    label=label_train,
                    linestyle='-',
                    marker='o',
                    linewidth=3,
                    markersize=6,
                )

        ax.set_xlabel("Predicted Mask Score")
        ax.set_ylabel("Density")
        ax.set_title(f"Score Distribution for Cluster: {cluster_name}")
        ax.legend()
        if logy:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()

        return fig, metrics


    def compute_metrics(self):
        metrics = dict()
        for cluster_name in self.cluster_matching.keys():
            metrics[cluster_name] = dict()
            metrics[cluster_name]["cluster-purity/true-class"] = self.cluster_matching[cluster_name]["pred-cls-correct-cluster"] / self.cluster_matching[cluster_name]["true-cluster"] if self.cluster_matching[cluster_name]["true-cluster"] > 0 else 0
            metrics[cluster_name]["cluster-purity/false-class"] = self.cluster_matching[cluster_name]["pred-cls-wrong-cluster"] / self.cluster_matching[cluster_name]["true-cluster"] if self.cluster_matching[cluster_name]["true-cluster"] > 0 else 0
            metrics[cluster_name]["cluster-purity/all"] = metrics[cluster_name]["cluster-purity/true-class"] + metrics[cluster_name]["cluster-purity/false-class"]
            metrics[cluster_name]["cluster-entries-purity/true-class"] = self.cluster_matching[cluster_name]["pred-cls-correct-cluster-entries"] / self.cluster_matching[cluster_name]["true-cluster-entries"] if self.cluster_matching[cluster_name]["true-cluster-entries"] > 0 else 0
            metrics[cluster_name]["cluster-entries-purity/false-class"] = self.cluster_matching[cluster_name]["pred-cls-wrong-cluster-entries"] / self.cluster_matching[cluster_name]["true-cluster-entries"] if self.cluster_matching[cluster_name]["true-cluster-entries"] > 0 else 0
            metrics[cluster_name]["cluster-entries-purity/all"] = metrics[cluster_name]["cluster-entries-purity/true-class"] + metrics[cluster_name]["cluster-entries-purity/false-class"]
            metrics[cluster_name]["cluster-num"] =  self.cluster_matching[cluster_name]["pred-cluster-sub-num"] / self.cluster_matching[cluster_name]["true-cluster"] if self.cluster_matching[cluster_name]["true-cluster"] > 0 else 0

        metrics_process = dict()
        for process_label in self.process_matching.keys():
            metrics_process[process_label] = dict()
            total_event = 0
            total_matched_event = 0
            total_cluster = 0
            total_matched_cluster = 0
            for num_q in self.process_matching[process_label].keys():
                if num_q > 0:
                    total_event += self.process_matching[process_label][num_q]["total-event"]
                    total_matched_event += self.process_matching[process_label][num_q]["matched-event"]
                    total_cluster += self.process_matching[process_label][num_q]["total-cluster"]
                    total_matched_cluster += self.process_matching[process_label][num_q]["matched-cluster"]
                metrics_process[process_label][num_q] = {
                    "event-purity": self.process_matching[process_label][num_q]["matched-event"] / self.process_matching[process_label][num_q]["total-event"] if self.process_matching[process_label][num_q]["total-event"] > 0 else 0,
                    "cluster-purity": self.process_matching[process_label][num_q]["matched-cluster"] / self.process_matching[process_label][num_q]["total-cluster"] if self.process_matching[process_label][num_q]["total-cluster"] > 0 else 0,
                }
            metrics_process[process_label]["*"] = {
                    "event-purity": total_matched_event / total_event if total_event > 0 else 0,
                    "cluster-purity": total_matched_cluster / total_cluster if total_cluster > 0 else 0,
            }

        return metrics, metrics_process



@time_decorator(name="[Segmentation] shared_step")
def shared_step(
        target_classification: torch.Tensor,
        target_mask: torch.Tensor,
        predict_classification: torch.Tensor,
        predict_mask: torch.Tensor,
        point_cloud_mask: torch.Tensor,
        seg_loss_fn: Callable,
        class_label: torch.Tensor,
        class_weight: torch.Tensor,
        metrics: SegmentationMetrics,
        loss_dict: dict,
        mask_loss_scale: float,
        dice_loss_scale: float,
        cls_loss_scale: float,
        event_weight: torch.Tensor = None,
        loss_name: str = "segmentation",
        update_metrics: bool = True,
        aux_outputs: Optional[dict] = None
):


    # Null class don't need to be predicted, so we mask it out
    # predict_class_label = predict_classification.argmax(dim=-1)
    # class_zero_mask = (predict_class_label == 0)  # (B, N)
    # mask_expanded = class_zero_mask.unsqueeze(-1).expand(-1, -1, predict_mask.shape[-1]) # (B, N, P)
    # predict_mask = predict_mask.masked_fill(mask_expanded, -99999)  # Apply class zero mask to the predicted mask

    # print("class_zero_mask", mask_expanded.shape, "predict_class_label", predict_mask.shape)

    # Make non-detectable points to be null class as well.

    debug_nonfinite_batch(
        {
            "target_classification": target_classification,
            "target_mask": target_mask,
            "predict_classification": predict_classification,
            "predict_mask": predict_mask,
            "point_cloud_mask": point_cloud_mask,
            "class_label": class_label,
            "event_weight": event_weight
        },
        batch_dim=0, name=loss_name, logger=logger
    )

    mask_loss, dice_loss, cls_loss, mask_loss_aux, dice_loss_aux, cls_loss_aux = seg_loss_fn(
        predict_cls = predict_classification,
        predict_mask = predict_mask,
        target_cls = target_classification,
        target_mask = target_mask,
        class_weight = class_weight,
        point_cloud_mask = point_cloud_mask,
        cls_loss_weight = cls_loss_scale,
        dice_loss_weight = dice_loss_scale,
        mask_loss_weight = mask_loss_scale,
        event_weight=event_weight,
        aux_outputs = aux_outputs
    )

    # print("predict_mask", predict_mask, "predict_cls", predict_classification)

    if update_metrics:
        metrics.update(
            y_true_mask=target_mask,
            y_true_cls=target_classification,
            y_pred_mask=predict_mask,
            y_pred_cls=predict_classification,
            process_cls=class_label
        )

    loss_dict[f"{loss_name}-cls"] = cls_loss
    loss_dict[f"{loss_name}-mask"] = mask_loss
    loss_dict[f"{loss_name}-dice"] = dice_loss
    loss_dict[f"{loss_name}-cls-aux"] = cls_loss_aux
    loss_dict[f"{loss_name}-mask-aux"] = mask_loss_aux
    loss_dict[f"{loss_name}-dice-aux"] = dice_loss_aux

    loss = cls_loss * cls_loss_scale + mask_loss * mask_loss_scale + dice_loss * dice_loss_scale
    loss += cls_loss_aux * cls_loss_scale + mask_loss_aux * mask_loss_scale + dice_loss_aux * dice_loss_scale

    return loss


@time_decorator(name="[Segmentation] shared_epoch_end")
def shared_epoch_end(
    global_rank,
    metrics_valid: SegmentationMetrics,
    metrics_train: SegmentationMetrics,
    logger,
    prefix: str = "",
):
    metrics_valid.reduce_across_gpus()
    if metrics_train is not None:
        metrics_train.reduce_across_gpus()
    if global_rank == 0:
        metrics_valid.assign_train_result(
            metrics_train.matrix if metrics_train is not None else None,
            metrics_train.matrix_number if metrics_train is not None else None,
            metrics_train.score_distribution if metrics_train is not None else None
        )
        fig_cm, fig_cm_num = metrics_valid.plot_cm(normalize=True)
        logger.log({
            f"{prefix}segmentation/CM": wandb.Image(fig_cm),
            f"{prefix}segmentation/CM-number": wandb.Image(fig_cm_num),

        })
        fig_cm.clear()
        fig_cm_num.clear()

        for cluster_name in metrics_valid.clusters_label.keys():
            fig_score, metrics_score = metrics_valid.plot_score_distributions(cluster_name, logy=True)
            logger.log({f"{prefix}{cluster_name}-segmentation/distribution": wandb.Image(fig_score)})
            logger.log({f"{prefix}{cluster_name}-segmentation/{label}": value for label, value in metrics_score.items()})
            fig_score.clear()

        metrics, metrics_process = metrics_valid.compute_metrics()
        for cluster_name in metrics:
            for log, value in metrics[cluster_name].items():
                logger.log({f"{prefix}{cluster_name}-segmentation/{log}": value})

        for process_label in metrics_process.keys():
            for num_q in metrics_process[process_label].keys():
                for log, value in metrics_process[process_label][num_q].items():
                    if value > 0:
                        logger.log({f"{prefix}{process_label}-segmentation/{num_q}obj/{log}": value})


    metrics_valid.reset()
    if metrics_train is not None:
        metrics_train.reset()



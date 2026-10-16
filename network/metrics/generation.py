from typing import Callable, Union

import numpy as np
import torch
from evenet.utilities.diffusion_sampler import DDIMSampler
from evenet.utilities.debug_tool import debug_nonfinite_batch
from functools import partial

import matplotlib.pyplot as plt
from evenet.network.loss.generation import loss as gen_loss
from evenet.utilities.debug_tool import time_decorator
from typing import Dict
import copy
from scipy.spatial.distance import jensenshannon
import logging


logger = logging.getLogger(__name__)

# Figure style of the validation plots (matches the Ztautau evaluation figures): small sans-serif fonts,
# open axes, no legend frame. Applied per figure with plt.rc_context, so nothing else is affected.
PLOT_STYLE = {"font.family": "sans-serif",
              "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
              "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8, "axes.linewidth": 0.8,
              "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
              "xtick.labelsize": 6, "ytick.labelsize": 6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
              "xtick.direction": "in", "ytick.direction": "in", "figure.dpi": 150}
# Okabe-Ito first, then tab10: distinguishable for up to ~18 classes
CLASS_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000",
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"]


def _pretty(name: str) -> str:
    """Display name of a distribution key: 'neutrino-delta_theta' -> 'neutrino: delta_theta'."""
    for prefix in ("neutrino-", "point cloud-", "global-"):
        if name.startswith(prefix):
            return f"{prefix[:-1]}: {name[len(prefix):]}"
    return name


class GenerationMetrics:
    def __init__(
            self, device, class_names,
            sequential_feature_names,
            invisible_feature_names,
            target_global_names, target_global_index, target_event_index,
            hist_xmin=-15, hist_xmax=15, num_bins=60,
            global_generation=False,
            point_cloud_generation=False,
            neutrino_generation=False,
            use_generation_result=False,
            special_bin_configs: dict[str, list] = None
    ):

        self.sampler = DDIMSampler(device)
        self.device = device

        self.global_generation = global_generation
        self.point_cloud_generation = point_cloud_generation
        self.neutrino_generation = neutrino_generation
        self.use_generation_result = use_generation_result

        # Default values for histogram
        self.num_bins = num_bins
        self.hist_xmin = hist_xmin
        self.hist_xmax = hist_xmax

        self.sequential_feature_names = sequential_feature_names
        self.invisible_feature_names = invisible_feature_names
        self.target_global_names = target_global_names
        self.target_global_index = target_global_index
        self.target_event_index = target_event_index

        self.bins = np.linspace(self.hist_xmin, self.hist_xmax, self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])

        self.num_classes = len(class_names)
        self.class_names = class_names

        self.histogram = dict()
        self.truth_histogram = dict()
        self.reference_histogram = dict()

        self.histogram_2d = dict()
        self.residual_histogram = dict()   # prediction - truth per event, same bin width as the marginals
        self.pearson_stats = dict()
        self.reference_pearson_stats = dict()

        self.special_bins = dict()
        self.special_bins_centers = dict()
        if special_bin_configs is not None:
            for name, special_bins in special_bin_configs.items():
                self.special_bins[name] = np.linspace(special_bins[1], special_bins[2], special_bins[0])
                self.special_bins_centers[name] = 0.5 * (self.special_bins[name][:-1] + self.special_bins[name][1:])

    @time_decorator(name="[Generation] update metrics")
    def update(
            self,
            model,
            input_set,
            num_steps_global=20,
            num_steps_point_cloud=40,
            num_steps_neutrino=40,
            eta=1.0,
            schedules: Union[None, dict] = None,
            precomputed_neutrino: torch.Tensor = None,
            reference_neutrino: torch.Tensor = None,
            neutrino_only: bool = False,
    ):
        model.eval()

        predict_distribution = dict()
        truth_distribution = dict()
        reference_distribution = dict()
        process_id = input_set['classification'] if 'classification' in input_set else torch.zeros_like(
            input_set['conditions_mask']).long()  # (batch_size, 1)
        masking = dict()

        do_recon = True
        do_truth = True
        if schedules is not None:
            do_recon = schedules.get('generation', False)
            do_truth = schedules.get('neutrino_generation', False)

        if self.global_generation and not neutrino_only:
            ####################################
            ##  Step 1: Generate num vectors  ##
            ####################################

            predict_for_global = partial(
                model.predict_diffusion_vector,
                cond_x=input_set,
                mode="global"
            )

            data_shape = [input_set['num_sequential_vectors'].shape[0], 1 + len(self.target_global_names)]
            generated_distribution = self.sampler.sample(
                data_shape=data_shape,
                pred_fn=predict_for_global,
                normalize_fn=None,
                num_steps=num_steps_global,
                eta=eta,
                use_tqdm=False,
                process_name=f"Global",
            )

            generated_num_sequential_vectors = generated_distribution[..., 0]
            generated_num_sequential_vectors = model.num_point_cloud_normalizer.denormalize(
                generated_num_sequential_vectors)

            predict_distribution["num_vectors"] = torch.floor(generated_num_sequential_vectors.flatten() + 0.5)
            truth_distribution["num_vectors"] = input_set['num_sequential_vectors'].flatten()

            if len(self.target_global_names) > 0:
                generated_global = generated_distribution[..., 1:]
                generated_global = model.global_normalizer.denormalize(generated_global, index=self.target_global_index)
                for idx, name in enumerate(self.target_global_names):
                    predict_distribution[f"global-{name}"] = generated_global[..., idx].flatten()
                    truth_distribution[f"global-{name}"] = (
                        input_set['conditions'][..., self.target_global_index[idx]]).flatten()

                if self.use_generation_result:
                    input_set = copy.deepcopy(input_set)
                    input_set['conditions'][..., self.target_global_index] = generated_global

        if self.point_cloud_generation and do_recon and not neutrino_only:
            ####################################
            ##  Step 2: Generate point cloud  ##
            ####################################

            data_shape = input_set['x'].shape
            process_id = input_set['classification'] if 'classification' in input_set else torch.zeros_like(
                input_set['conditions_mask']).long()  # (batch_size, 1)

            predict_for_point_cloud = partial(
                model.predict_diffusion_vector,
                mode="event",
                cond_x=input_set,
                noise_mask=input_set["x_mask"].unsqueeze(-1)  # [B, T, 1] to match noise x
            )  # TODO: add stuff from previous step.

            generated_distribution = self.sampler.sample(
                data_shape=data_shape,
                pred_fn=predict_for_point_cloud,
                normalize_fn=model.sequential_normalizer,
                eta=eta,
                num_steps=num_steps_point_cloud,
                noise_mask=input_set["x_mask"].unsqueeze(-1),  # [B, T, 1] to match noise x
                use_tqdm=False,
                process_name=f"PointCloud",
            )

            for i in range(data_shape[-1]):
                if i in self.target_event_index:
                    masking[f"point cloud-{self.sequential_feature_names[i]}"] = input_set["x_mask"]
                    predict_distribution[f"point cloud-{self.sequential_feature_names[i]}"] = generated_distribution[
                        ..., i]
                    truth_distribution[f"point cloud-{self.sequential_feature_names[i]}"] = input_set['x'][..., i]


        if self.neutrino_generation and do_truth:
            #####################################
            ## Generate invisible point cloud  ##
            #####################################

            data_shape = input_set['x_invisible'].shape
            process_id = input_set['classification'] if 'classification' in input_set else torch.zeros_like(
                input_set['conditions_mask'].flatten()).long()  # (batch_size, 1)

            if precomputed_neutrino is not None:
                generated_distribution = precomputed_neutrino
            else:
                predict_for_neutrino = partial(
                    model.predict_diffusion_vector,
                    mode="neutrino",
                    cond_x=input_set,
                    noise_mask=input_set["x_invisible_mask"].unsqueeze(-1),
                )
                generated_distribution = self.sampler.sample(
                    data_shape=data_shape,
                    pred_fn=predict_for_neutrino,
                    normalize_fn=model.invisible_normalizer,
                    eta=eta,
                    num_steps=num_steps_neutrino,
                    use_tqdm=False,
                    process_name="Neutrino",
                    # remove_padding=(getattr(model, "invisible_padding", 0) > 0),
                    remove_padding=False,
                )
            if generated_distribution.shape != input_set['x_invisible'].shape:
                raise ValueError(
                    "Precomputed neutrino validation output does not match truth shape"
                )
            if reference_neutrino is not None and reference_neutrino.shape != generated_distribution.shape:
                raise ValueError(
                    "Reference neutrino validation output does not match prediction shape"
                )

            for i in range(data_shape[-1]):
                masking[f"neutrino-{self.invisible_feature_names[i]}"] = input_set["x_invisible_mask"]
                predict_distribution[f"neutrino-{self.invisible_feature_names[i]}"] = generated_distribution[..., i]
                truth_distribution[f"neutrino-{self.invisible_feature_names[i]}"] = input_set['x_invisible'][..., i]
                if reference_neutrino is not None:
                    reference_distribution[f"neutrino-{self.invisible_feature_names[i]}"] = reference_neutrino[..., i]

        # --------------- working line -----------------
        for distribution_name, distribution in predict_distribution.items():

            num_bins = self.num_bins
            if distribution_name in self.special_bins:
                num_bins = len(self.special_bins_centers[distribution_name])

            if distribution_name not in self.histogram:
                self.histogram[distribution_name] = {
                    class_name: np.zeros(num_bins)
                    for class_name in self.class_names
                }
            if distribution_name not in self.truth_histogram:
                self.truth_histogram[distribution_name] = {
                    class_name: np.zeros(num_bins)
                    for class_name in self.class_names
                }
            if distribution_name in reference_distribution and distribution_name not in self.reference_histogram:
                self.reference_histogram[distribution_name] = {
                    class_name: np.zeros(num_bins)
                    for class_name in self.class_names
                }

            if distribution_name not in self.histogram_2d:
                self.histogram_2d[distribution_name] = {
                    class_name: np.zeros((num_bins, num_bins))
                    for class_name in self.class_names
                }
            if distribution_name not in self.residual_histogram:
                self.residual_histogram[distribution_name] = {
                    class_name: np.zeros(num_bins)
                    for class_name in self.class_names
                }

            if distribution_name not in self.pearson_stats:
                self.pearson_stats[distribution_name] = {
                    class_name: {
                        'sum_x': 0.0, 'sum_y': 0.0,
                        'sum_xx': 0.0, 'sum_yy': 0.0,
                        'sum_xy': 0.0, 'n': 0
                    } for class_name in self.class_names
                }
            if distribution_name in reference_distribution and distribution_name not in self.reference_pearson_stats:
                self.reference_pearson_stats[distribution_name] = {
                    class_name: {
                        'sum_x': 0.0, 'sum_y': 0.0,
                        'sum_xx': 0.0, 'sum_yy': 0.0,
                        'sum_xy': 0.0, 'n': 0
                    } for class_name in self.class_names
                }

            for class_index, class_name in enumerate(self.class_names):
                class_mask = (process_id == class_index)
                if distribution_name in masking and (
                        predict_distribution[distribution_name].size() == masking[distribution_name].size()):
                    # Masking for point cloud
                    total_mask = masking[distribution_name][class_mask].flatten()
                    pred = predict_distribution[distribution_name][class_mask].flatten()[
                        total_mask].detach().cpu().numpy()
                    truth = truth_distribution[distribution_name][class_mask].flatten()[
                        total_mask].detach().cpu().numpy()
                else:
                    pred = predict_distribution[distribution_name][class_mask].detach().cpu().numpy()
                    truth = truth_distribution[distribution_name][class_mask].flatten().detach().cpu().numpy()

                hist_bins = self.bins
                if distribution_name in self.special_bins:
                    hist_bins = self.special_bins[distribution_name]

                hist, _ = np.histogram(pred, bins=hist_bins)
                self.histogram[distribution_name][class_name] += hist

                hist, _ = np.histogram(truth, bins=hist_bins)
                self.truth_histogram[distribution_name][class_name] += hist

                reference = None
                if distribution_name in reference_distribution:
                    if distribution_name in masking and (
                            reference_distribution[distribution_name].size() == masking[distribution_name].size()):
                        reference = reference_distribution[distribution_name][class_mask].flatten()[total_mask].detach().cpu().numpy()
                    else:
                        reference = reference_distribution[distribution_name][class_mask].flatten().detach().cpu().numpy()
                    hist, _ = np.histogram(reference, bins=hist_bins)
                    self.reference_histogram[distribution_name][class_name] += hist

                hist2d, _, _ = np.histogram2d(pred, truth, bins=[hist_bins, hist_bins])
                self.histogram_2d[distribution_name][class_name] += hist2d
                if pred.shape == truth.shape:  # residual bins: the marginal bins re-centred on zero
                    hist, _ = np.histogram(pred - truth, bins=hist_bins - 0.5 * (hist_bins[0] + hist_bins[-1]))
                    self.residual_histogram[distribution_name][class_name] += hist

                # Pearson stats
                stats = self.pearson_stats[distribution_name][class_name]
                stats['sum_x'] += pred.sum()
                stats['sum_y'] += truth.sum()
                stats['sum_xx'] += (pred ** 2).sum()
                stats['sum_yy'] += (truth ** 2).sum()
                stats['sum_xy'] += (pred * truth).sum()
                stats['n'] += pred.shape[0]
                if reference is not None:
                    stats = self.reference_pearson_stats[distribution_name][class_name]
                    stats['sum_x'] += reference.sum()
                    stats['sum_y'] += truth.sum()
                    stats['sum_xx'] += (reference ** 2).sum()
                    stats['sum_yy'] += (truth ** 2).sum()
                    stats['sum_xy'] += (reference * truth).sum()
                    stats['n'] += reference.shape[0]

    def reset(self):
        self.histogram = dict()
        self.truth_histogram = dict()
        self.reference_histogram = dict()

        self.histogram_2d = dict()
        self.residual_histogram = dict()
        self.pearson_stats = dict()
        self.reference_pearson_stats = dict()

    def reduce_across_gpus(self):
        if not torch.distributed.is_initialized():
            return

        # Helper function to reduce a nested dict
        def reduce_nested_histogram(nested_hist, dtype=torch.long):
            for name_, hist_group in nested_hist.items():
                for class_name_, data in hist_group.items():
                    tensor_ = torch.tensor(data, dtype=dtype, device=self.device)
                    torch.distributed.all_reduce(tensor_, op=torch.distributed.ReduceOp.SUM)
                    nested_hist[name_][class_name_] = tensor_.cpu().numpy()

        reduce_nested_histogram(self.histogram, dtype=torch.long)
        reduce_nested_histogram(self.truth_histogram, dtype=torch.long)
        reduce_nested_histogram(self.reference_histogram, dtype=torch.long)
        reduce_nested_histogram(self.histogram_2d, dtype=torch.long)
        reduce_nested_histogram(self.residual_histogram, dtype=torch.long)

        for name, stats_group in self.pearson_stats.items():
            for class_name, stats in stats_group.items():
                for key in ['sum_x', 'sum_y', 'sum_xx', 'sum_yy', 'sum_xy', 'n']:
                    tensor = torch.tensor(stats[key], dtype=torch.float32, device=self.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    stats[key] = tensor.item()
        for name, stats_group in self.reference_pearson_stats.items():
            for class_name, stats in stats_group.items():
                for key in ['sum_x', 'sum_y', 'sum_xx', 'sum_yy', 'sum_xy', 'n']:
                    tensor = torch.tensor(stats[key], dtype=torch.float32, device=self.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    stats[key] = tensor.item()

    @staticmethod
    def _class_grid(active, ratio=False):
        """Figure with one panel per active class (4 columns); with ratio=True each panel has a ratio axis below."""
        import matplotlib.gridspec as gridspec
        n = max(len(active), 1)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig = plt.figure(figsize=(2.4 * ncols, (2.3 if ratio else 1.9) * nrows + 0.4))
        outer = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55 if ratio else 0.5, wspace=0.3,
                                  left=0.07, right=0.99, top=0.9, bottom=0.11)
        axes = []
        for k in range(n):
            cell = outer[k // ncols, k % ncols]
            if ratio:
                inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=cell, height_ratios=[3, 1], hspace=0.06)
                ax = fig.add_subplot(inner[0])
                rax = fig.add_subplot(inner[1], sharex=ax)
                axes.append((ax, rax))
            else:
                axes.append((fig.add_subplot(cell), None))
        return fig, axes

    @staticmethod
    def _ratio_limits(r, re):
        dev = np.abs(r[np.isfinite(r) & np.isfinite(re) & (re < 0.5)] - 1)
        need = 1.1 * np.quantile(dev, 0.9) if len(dev) else 0.5
        span = next((x for x in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0) if x >= need), 1.0)
        return max(0.0, 1 - span), 1 + span

    def plot_histogram_func(
            self,
            truth_histogram,
            histogram,
            bin_widths,
            bin_centers,
            reference_histogram=None,
            name="",
    ):
        """One panel per class: the truth as filled reference with its statistical band, the prediction as
        points with errors, both normalised to unit area, and a prediction / truth ratio below."""
        edges = np.concatenate([bin_centers - bin_widths / 2, [bin_centers[-1] + bin_widths[-1] / 2]])
        jsd = dict()
        active = [c for c in self.class_names if np.sum(histogram[c]) > 0 or np.sum(truth_histogram[c]) > 0]
        with plt.rc_context(PLOT_STYLE):
            fig, axes = self._class_grid(active, ratio=True)
            for (ax, rax), cls_name in zip(axes, active):
                counts, truth_counts = histogram[cls_name].astype(float), truth_histogram[cls_name].astype(float)
                n_pred, n_truth = np.sum(counts), np.sum(truth_counts)
                truth_density = truth_counts / (n_truth * bin_widths) if n_truth > 0 else np.zeros_like(counts)
                truth_err = np.sqrt(truth_counts) / (n_truth * bin_widths) if n_truth > 0 else np.zeros_like(counts)
                pred_density = counts / (n_pred * bin_widths) if n_pred > 0 else np.zeros_like(counts)
                pred_err = np.sqrt(counts) / (n_pred * bin_widths) if n_pred > 0 else np.zeros_like(counts)
                if n_truth > 0:
                    ax.stairs(truth_density, edges, fill=True, color="#D9D9D9", lw=0, label="Truth")
                    ax.stairs(truth_density, edges, color="black", lw=0.6)
                    ax.stairs(truth_density + truth_err, edges, baseline=truth_density - truth_err, fill=True,
                              facecolor="none", edgecolor="#444444", hatch="//////", lw=0, label="Truth stat.")
                if n_pred > 0:
                    shown = counts > 0
                    ax.errorbar(bin_centers[shown], pred_density[shown], yerr=pred_err[shown], fmt="o", ms=2.0,
                                color="#0072B2", lw=0.7, capsize=0, label="Prediction")
                if reference_histogram is not None and np.sum(reference_histogram[cls_name]) > 0:
                    ref = reference_histogram[cls_name].astype(float)
                    ax.stairs(ref / (np.sum(ref) * bin_widths), edges, color="#D55E00", lw=0.9, ls="--", label="Reference")
                text = f"n = {int(n_pred):,}"
                if n_pred > 0 and n_truth > 0:
                    jsd[cls_name] = jensenshannon(truth_counts / n_truth, counts / n_pred)
                    text += f"\nJSD = {jsd[cls_name]:.3f}"
                top = max(truth_density.max() if n_truth > 0 else 0, (pred_density + pred_err).max() if n_pred > 0 else 0, 1e-9)
                ax.set_ylim(0, 1.45 * top)
                ax.set_xlim(edges[0], edges[-1])
                ax.set_title(cls_name, loc="left", fontsize=7, pad=3)
                ax.text(0.97, 0.95, text, transform=ax.transAxes, ha="right", va="top", fontsize=5.5, linespacing=1.3)
                ax.set_ylabel("Density")
                ax.tick_params(labelbottom=False)
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.where((truth_density > 0) & (counts > 0), pred_density / truth_density, np.nan)
                    re = pred_err / truth_density
                    band = np.where(truth_density > 0, truth_err / truth_density, 0.0)
                lo, hi = self._ratio_limits(r, re)
                rax.stairs(1 + band, edges, baseline=1 - band, fill=True, color="#D0D0D0", lw=0)
                rax.axhline(1, color="black", lw=0.5, ls="--")
                inside = np.isfinite(r) & (r >= lo) & (r <= hi)
                rax.errorbar(bin_centers[inside], r[inside], yerr=re[inside], fmt="o", ms=1.8, color="#0072B2", lw=0.6, capsize=0)
                pad = 0.04 * (hi - lo)
                up, down = np.isfinite(r) & (r > hi), np.isfinite(r) & (r < lo)
                rax.plot(bin_centers[up], np.full(up.sum(), hi - pad), "^", ms=2.5, color="#0072B2", mec="none")
                rax.plot(bin_centers[down], np.full(down.sum(), lo + pad), "v", ms=2.5, color="#0072B2", mec="none")
                rax.set_ylim(lo, hi)
                rax.set_ylabel("Pred. / Truth", fontsize=6)
                rax.set_xlabel(_pretty(name))
            if axes:
                axes[0][0].legend(fontsize=5.5, loc="upper left", handlelength=1.4, borderaxespad=0.2)
            fig.suptitle(f"{_pretty(name)} — validation, unit-area marginals per class", x=0.01, ha="left", fontsize=8)
        return fig, jsd

    def plot_residual_func(self, residual_histogram, bin_widths, bin_centers, name=""):
        """Per-event residual (prediction - truth), one panel per class, with the mean, RMS and 68 % half-width."""
        centre = 0.5 * (bin_centers[0] - bin_widths[0] / 2 + bin_centers[-1] + bin_widths[-1] / 2)
        centers = bin_centers - centre
        edges = np.concatenate([centers - bin_widths / 2, [centers[-1] + bin_widths[-1] / 2]])
        active = [c for c in self.class_names if np.sum(residual_histogram[c]) > 0]
        # common x range: 5 x the largest class RMS (the bins span the full target range, residuals are narrower)
        widest = 0.0
        for cls_name in active:
            counts = residual_histogram[cls_name].astype(float)
            mean = np.sum(counts * centers) / np.sum(counts)
            widest = max(widest, np.sqrt(np.sum(counts * (centers - mean) ** 2) / np.sum(counts)))
        half_range = min(edges[-1], max(5 * widest, 3 * bin_widths.max()))
        with plt.rc_context(PLOT_STYLE):
            fig, axes = self._class_grid(active)
            for (ax, _), cls_name in zip(axes, active):
                counts = residual_histogram[cls_name].astype(float)
                n = np.sum(counts)
                mean = np.sum(counts * centers) / n
                rms = np.sqrt(np.sum(counts * (centers - mean) ** 2) / n)
                cumulative = np.cumsum(counts) / n
                q16, q84 = (np.interp(q, cumulative, edges[1:]) for q in (0.16, 0.84))
                density = counts / (n * bin_widths)
                ax.stairs(density, edges, fill=True, color="#0072B2", alpha=0.18, lw=0)
                ax.stairs(density, edges, color="#0072B2", lw=1.0)
                ax.axvline(0.0, color="black", lw=0.5, ls="--")
                ax.axvspan(q16, q84, color="#0072B2", alpha=0.08, lw=0)
                ax.set_title(cls_name, loc="left", fontsize=7, pad=3)
                ax.text(0.97, 0.95, f"n = {int(n):,}\nmean = {mean:.3g}\nRMS = {rms:.3g}\n68 % = {0.5 * (q84 - q16):.3g}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=5.5, linespacing=1.3)
                ax.set_xlim(-half_range, half_range)
                ax.set_ylim(0, 1.45 * max(density.max(), 1e-9))
                ax.set_ylabel("Density")
                ax.set_xlabel("prediction − truth")
            fig.suptitle(f"{_pretty(name)} — validation residual per event", x=0.01, ha="left", fontsize=8)
        return fig

    def plot_histogram2d_func(self, histogram2d, x_centers, y_centers, title="2D Histogram"):
        """Prediction against truth (counts, log colour scale) with the diagonal."""
        import matplotlib.colors as mcolors
        dx, dy = np.diff(x_centers).mean(), np.diff(y_centers).mean()
        extent = [y_centers[0] - dy / 2, y_centers[-1] + dy / 2, x_centers[0] - dx / 2, x_centers[-1] + dx / 2]
        with plt.rc_context(PLOT_STYLE):
            fig, ax = plt.subplots(figsize=(3.6, 3.2))
            masked = np.ma.masked_less_equal(histogram2d, 0)
            image = ax.imshow(masked, origin="lower", extent=extent, aspect="auto", cmap="viridis",
                              norm=mcolors.LogNorm(vmin=1, vmax=max(float(histogram2d.max()), 1.0)))
            ax.plot([extent[0], extent[1]], [extent[2], extent[3]], color="#D55E00", lw=0.8, ls="--")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Events")
            ax.set_xlabel("truth")
            ax.set_ylabel("prediction")
            ax.set_title(title, loc="left")
            fig.tight_layout()
        return fig

    def plot_histogram(self):
        figs = dict()

        jsd_results = dict()
        for name in self.histogram:
            bin_centers = self.bin_centers
            bin_widths = np.diff(self.bins)

            if name in self.special_bins:
                bin_widths = np.diff(self.special_bins[name])
                bin_centers = self.special_bins_centers[name]

            figs[f"{name}-1d"], jsd = self.plot_histogram_func(
                self.truth_histogram[name],
                self.histogram[name],
                bin_widths=bin_widths,
                bin_centers=bin_centers,
                reference_histogram=self.reference_histogram.get(name),
                name=name,
            )
            if name in self.residual_histogram:
                figs[f"{name}-residual"] = self.plot_residual_func(
                    self.residual_histogram[name], bin_widths=bin_widths, bin_centers=bin_centers, name=name)
            for cls_name, score in jsd.items():
                jsd_results[f"{name}-{cls_name}"] = score
                if name in self.reference_histogram:
                    reference_counts = self.reference_histogram[name][cls_name]
                    truth_counts = self.truth_histogram[name][cls_name]
                    if np.sum(reference_counts) > 0 and np.sum(truth_counts) > 0:
                        jsd_results[f"reference-{name}-{cls_name}"] = jensenshannon(
                            truth_counts / np.sum(truth_counts),
                            reference_counts / np.sum(reference_counts),
                        )

            for class_name in self.class_names:
                if class_name not in jsd:
                    continue
                if 'neutrino' not in name:
                    continue

                fig = self.plot_histogram2d_func(
                    self.histogram_2d[name][class_name],
                    x_centers=bin_centers,
                    y_centers=bin_centers,
                    title=f"{_pretty(name)} — {class_name}, validation"
                )
                figs[f"2D_{name}_{class_name}"] = fig

        # Pearson correlation
        pearson_results = dict()
        for name in self.pearson_stats:
            if 'neutrino' not in name:
                continue

            pearson_results[name] = dict()
            for class_name in self.class_names:

                if f"{name}-{class_name}" not in jsd_results:
                    continue

                stats = self.pearson_stats[name][class_name]
                n = stats['n']
                numerator = n * stats['sum_xy'] - stats['sum_x'] * stats['sum_y']
                denominator = np.sqrt(
                    (n * stats['sum_xx'] - stats['sum_x'] ** 2) *
                    (n * stats['sum_yy'] - stats['sum_y'] ** 2)
                )
                if denominator == 0:
                    r = 0.0
                else:
                    r = numerator / denominator
                pearson_results[name][class_name] = r

                if name in self.reference_pearson_stats:
                    stats = self.reference_pearson_stats[name][class_name]
                    n = stats['n']
                    numerator = n * stats['sum_xy'] - stats['sum_x'] * stats['sum_y']
                    denominator = np.sqrt(
                        (n * stats['sum_xx'] - stats['sum_x'] ** 2) *
                        (n * stats['sum_yy'] - stats['sum_y'] ** 2)
                    )
                    pearson_results[f"reference-{name}"] = pearson_results.get(f"reference-{name}", {})
                    pearson_results[f"reference-{name}"][class_name] = 0.0 if denominator == 0 else numerator / denominator

        return figs, pearson_results, jsd_results


@time_decorator(name="[Generation] shared_step")
def shared_step(
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        gen_metrics: GenerationMetrics,
        model: torch.nn.Module,
        global_loss_scale: float,
        event_loss_scale: float,
        invisible_loss_scale: float,
        device: torch.device,
        loss_head_dict: dict,
        num_steps_global=20,
        num_steps_point_cloud=100,
        num_steps_neutrino=100,
        diffusion_on: bool = False,
        invisible_padding: int = 0,
        update_metric: bool = True,
        event_weight: torch.Tensor = None,
        schedules: Union[None, dict] = None,
        debug_nonfinite: bool = True,
):
    generation_loss = dict()

    global_gen_loss = torch.tensor(0.0, device=device, requires_grad=True)
    recon_gen_loss = torch.tensor(0.0, device=device, requires_grad=True)
    truth_gen_loss = torch.tensor(0.0, device=device, requires_grad=True)
    for generation_target, generation_result in outputs.items():
        feature_dim = generation_result["vector"].shape[-1]
        if generation_target == "point_cloud":
            masking = generation_result["mask"]
        elif generation_target == "neutrino":
            masking = generation_result["mask"] # (B, N, 1)

            if invisible_padding > 0:
                B, N, _ = masking.shape
                # Expand to (B, N, F), with True for real features, False for padded ones
                masking = masking.expand(B, N, feature_dim).clone()
                masking[:, :, -invisible_padding:] = False  # mask out padded features
                feature_dim = 1
        elif generation_target == "num_point_cloud":
            masking = None
            feature_dim = None
        else:
            masking = None
            feature_dim = None

        generation_loss[generation_target] = gen_loss(
            predict=generation_result["vector"],
            target=generation_result["truth"],
            mask=masking,
            feature_dim=feature_dim,
            event_weight=event_weight
        )

        if debug_nonfinite:
            debug_nonfinite_batch(
                {
                    "predict": generation_result["vector"],
                    "truth": generation_result["truth"],
                    "mask": masking,
                    "weight": event_weight,
                },
                batch_dim=0,  # change if your batch axis differs
                name=f"gen/{generation_target}",
                logger=logger,
            )

        # if generation loss is nan, then print all details
        if debug_nonfinite and torch.isnan(generation_loss[generation_target]):
            logger.warning(
                f"NaN in generation loss for {generation_target} "
                f"predict: {generation_result['vector']}, truth: {generation_result['truth']}, "
            )

            generation_loss[generation_target] = 0.0

        if generation_target == "global":
            global_gen_loss = global_gen_loss + generation_loss[generation_target]
            loss_head_dict["generation-global"] = global_gen_loss
        elif generation_target == "neutrino":
            truth_gen_loss = truth_gen_loss + generation_loss[generation_target]
            loss_head_dict["generation-truth"] = truth_gen_loss

        elif generation_target == "point_cloud":
            recon_gen_loss = recon_gen_loss + generation_loss[generation_target]
            loss_head_dict["generation-recon"] = recon_gen_loss

        if diffusion_on and update_metric:
            gen_metrics.update(
                model=model,
                input_set=batch,
                num_steps_global=num_steps_global,
                num_steps_point_cloud=num_steps_point_cloud,
                num_steps_neutrino=num_steps_neutrino,
                schedules=schedules,
            )

    loss = (global_gen_loss * global_loss_scale + recon_gen_loss * event_loss_scale + truth_gen_loss * invisible_loss_scale) / len(
        outputs)
    # print(f"Training: {model.training}, loss scale: {event_loss_scale}, total sum: {torch.sum(masking) if masking is not None else masking}, Global loss: {global_gen_loss.item()}, Recon loss: {recon_gen_loss.item()}, Truth loss: {truth_gen_loss.item()}, loss: {loss.item()}")

    return loss, generation_loss


@time_decorator(name="[Generation] shared_epoch_end")
def shared_epoch_end(
        global_rank,
        metrics_valid: GenerationMetrics,
        metrics_train: GenerationMetrics,
        logger,
):
    import wandb

    metrics_valid.reduce_across_gpus()
    if metrics_train:
        metrics_train.reduce_across_gpus()

    if global_rank == 0:
        category_map = {
            "neutrino-": "generation-invisible",
            "point cloud-": "generation-event",
            "global-": "generation-global"
        }
        figs, extra, jsd_results = metrics_valid.plot_histogram()
        for name, fig in figs.items():

            for prefix, category in category_map.items():
                if prefix in name:
                    tag = name.replace(prefix, "")
                    logger.log({f"{category}/{tag}": wandb.Image(fig)})
                    break

            plt.close(fig)

        for name in extra:
            for class_name, value in extra[name].items():
                # logger.log({f"generation/pearson_{name}_{class_name}": value})

                for prefix, category in category_map.items():
                    if prefix in name:
                        tag = name.replace(prefix, "")
                        logger.log({f"{category}/pearson/{tag}_{class_name}": value})
                        break

        for _ in jsd_results:
            for jsd_name, jsd_score in jsd_results.items():
                # logger.log({f"generation/jsd_{jsd_name}": jsd_score})

                for prefix, category in category_map.items():
                    if prefix in jsd_name:
                        tag = jsd_name.replace(prefix, "")
                        logger.log({f"{category}/jsd/{tag}": jsd_score})
                        break

    metrics_valid.reset()
    if metrics_train:
        metrics_train.reset()

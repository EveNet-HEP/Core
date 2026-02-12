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
import wandb
import copy
from scipy.spatial.distance import jensenshannon
import logging


logger = logging.getLogger(__name__)

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

        self.histogram_2d = dict()
        self.pearson_stats = dict()

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
            schedules: Union[None, dict] = None
    ):
        model.eval()

        predict_distribution = dict()
        truth_distribution = dict()
        process_id = input_set['classification'] if 'classification' in input_set else torch.zeros_like(
            input_set['conditions_mask']).long()  # (batch_size, 1)
        masking = dict()

        do_recon = True
        do_truth = True
        if schedules is not None:
            do_recon = schedules.get('generation', False)
            do_truth = schedules.get('neutrino_generation', False)

        if self.global_generation:
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

        if self.point_cloud_generation and do_recon:
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

            predict_for_neutrino = partial(
                model.predict_diffusion_vector,
                mode="neutrino",
                cond_x=input_set,
                noise_mask=input_set["x_invisible_mask"].unsqueeze(-1)  # [B, T, 1] to match noise x
            )

            generated_distribution = self.sampler.sample(
                data_shape=data_shape,
                pred_fn=predict_for_neutrino,
                normalize_fn=model.invisible_normalizer,
                eta=eta,
                num_steps=num_steps_neutrino,
                use_tqdm=False,
                process_name=f"Neutrino",
                remove_padding=True,
            )

            for i in range(data_shape[-1]):
                masking[f"neutrino-{self.invisible_feature_names[i]}"] = input_set["x_invisible_mask"]
                predict_distribution[f"neutrino-{self.invisible_feature_names[i]}"] = generated_distribution[..., i]
                truth_distribution[f"neutrino-{self.invisible_feature_names[i]}"] = input_set['x_invisible'][..., i]

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

            if distribution_name not in self.histogram_2d:
                self.histogram_2d[distribution_name] = {
                    class_name: np.zeros((num_bins, num_bins))
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

                hist2d, _, _ = np.histogram2d(pred, truth, bins=[hist_bins, hist_bins])
                self.histogram_2d[distribution_name][class_name] += hist2d

                # Pearson stats
                stats = self.pearson_stats[distribution_name][class_name]
                stats['sum_x'] += pred.sum()
                stats['sum_y'] += truth.sum()
                stats['sum_xx'] += (pred ** 2).sum()
                stats['sum_yy'] += (truth ** 2).sum()
                stats['sum_xy'] += (pred * truth).sum()
                stats['n'] += pred.shape[0]

    def reset(self):
        self.histogram = dict()
        self.truth_histogram = dict()

        self.histogram_2d = dict()
        self.pearson_stats = dict()

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
        reduce_nested_histogram(self.histogram_2d, dtype=torch.long)

        for name, stats_group in self.pearson_stats.items():
            for class_name, stats in stats_group.items():
                for key in ['sum_x', 'sum_y', 'sum_xx', 'sum_yy', 'sum_xy', 'n']:
                    tensor = torch.tensor(stats[key], dtype=torch.float32, device=self.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    stats[key] = tensor.item()

    def plot_histogram_func(
            self,
            truth_histogram,
            histogram,
            bin_widths,
            bin_centers,
    ):

        colors = [
            "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD",
            "#CD5582", "#E1BE6A", "#E1BE6A", "#E89A7A", "#EC6B2D"
        ]

        fig, ax = plt.subplots()

        jsd = dict()
        for cls, cls_name in enumerate(self.class_names):
            # Plot training histogram (bars)
            counts = histogram[cls_name]
            if np.sum(counts) > 0:
                density = counts / (np.sum(counts) * bin_widths)
                color = colors[cls % len(colors)]
                label = f"{cls_name} (Pred)"
                plt.plot(
                    bin_centers,
                    density,
                    color=color,
                    label=label,
                    linestyle='--',
                    marker='o',
                    linewidth=2,
                    markersize=6
                )
            truth_counts = truth_histogram[cls_name]
            if np.sum(truth_counts) > 0:
                truth_density = truth_counts / (np.sum(truth_counts) * bin_widths)
                color = colors[cls % len(colors)]
                label = f"{cls_name} (Truth)"
                plt.bar(
                    bin_centers,
                    truth_density,
                    width=bin_widths,
                    color=color,
                    alpha=0.7,
                    label=f"{cls_name} (Truth)", edgecolor=color, fill=False
                )

            if (np.sum(counts) > 0) and (np.sum(truth_counts) > 0):
                p = truth_counts / np.sum(truth_counts)
                q = counts / np.sum(counts)
                jsd[cls_name] = (jensenshannon(p, q))

        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        # plt.show()

        return fig, jsd

    def plot_histogram2d_func(self, histogram2d, x_centers, y_centers, title="2D Histogram"):
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
        pcm = ax.pcolormesh(X, Y, histogram2d, shading='auto', cmap='viridis')
        fig.colorbar(pcm, ax=ax, label="Counts")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Truth')
        ax.set_title(title)
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
            )
            for cls_name, score in jsd.items():
                jsd_results[f"{name}-{cls_name}"] = score

            for class_name in self.class_names:
                if class_name not in jsd:
                    continue
                if 'neutrino' not in name:
                    continue

                fig = self.plot_histogram2d_func(
                    self.histogram_2d[name][class_name],
                    x_centers=bin_centers,
                    y_centers=bin_centers,
                    title=f"2D Histogram {name} - {class_name}"
                )
                figs[f"2D_{name}_{class_name}"] = fig

        # Pearson correlation
        pearson_results = dict()
        for name in self.pearson_stats:
            if 'neutrino' not in name:
                continue

            pearson_results[name] = dict()
            for class_name in self.class_names:

                if class_name not in jsd_results:
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
        schedules: Union[None, dict] = None
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
        if torch.isnan(generation_loss[generation_target]):
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

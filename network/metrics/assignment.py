import re

import torch.nn

from evenet.network.loss.assignment import convert_target_assignment
from evenet.utilities.debug_tool import time_decorator, debug_nonfinite_batch
from evenet.utilities.group_theory import complete_indices, symmetry_group
from evenet.control.event_info import EventInfo
from evenet.network.metrics.predict_assignment import extract_predictions

from typing import List, Dict
from torch import Tensor

from functools import reduce
from itertools import permutations, product
import warnings
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import wandb
from matplotlib.lines import Line2D
import logging

logger = logging.getLogger(__name__)


@time_decorator(name="[Assignment] reconstruct_mass_peak")
def reconstruct_mass_peak(Jet, assignment_indices, padding_mask, log_energy=True):
    """
    *** input Jet with log pt and log mass ***
    Jet: [batch_size, num_jets, 4]
    assignment_indices: [batch_size, num_targets]
    """
    jet_pt = Jet[..., 0]
    jet_eta = Jet[..., 1]
    jet_phi = Jet[..., 2]
    # jet_mass = Jet[..., 3]
    jet_energy = Jet[..., 3]

    if log_energy:
        jet_pt = torch.expm1(jet_pt)
        jet_energy = torch.expm1(jet_energy)

    def gather_jets(jet_tensor):
        assert (assignment_indices >= 0).all(), f"assignment_indices has negative values! {assignment_indices}"

        return torch.gather(jet_tensor.unsqueeze(1), 2, assignment_indices.unsqueeze(1)).squeeze(1)

    pt = gather_jets(jet_pt)
    eta = gather_jets(jet_eta)
    phi = gather_jets(jet_phi)
    energy = gather_jets(jet_energy)

    selected_mask = torch.gather(padding_mask.unsqueeze(1), 2, assignment_indices.unsqueeze(1)).squeeze(1)
    is_valid_event = selected_mask.all(dim=1)

    # 4-vector components
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    E = energy

    total_e = E.sum(dim=1)
    total_px = px.sum(dim=1)
    total_py = py.sum(dim=1)
    total_pz = pz.sum(dim=1)

    mass_squared = total_e ** 2 - (total_px ** 2 + total_py ** 2 + total_pz ** 2)
    mass_squared = torch.clamp(mass_squared, min=0.0)
    invariant_mass = torch.sqrt(mass_squared)

    invariant_mass[~is_valid_event] = float(-999)
    return invariant_mass


def get_assignment_necessaries(
        event_info: EventInfo,
) -> dict:
    permutation_indices = dict()
    num_targets = dict()
    event_particles = dict()
    for process in event_info.process_names:
        permutation_indices[process] = []
        num_targets[process] = []
        for event_particle_name, product_symmetry in event_info.product_symmetries[process].items():
            topology_name = ''.join(event_info.product_particles[process][event_particle_name].names)
            topology_name = f"{event_particle_name}/{topology_name}"
            topology_name = re.sub(r'\d+', '', topology_name)
            topology_category_name = event_info.pairing_topology[topology_name][
                "pairing_topology_category"]
            permutation_indices_tmp = complete_indices(
                event_info.pairing_topology_category[topology_category_name][
                    "product_symmetry"].degree,
                event_info.pairing_topology_category[topology_category_name][
                    "product_symmetry"].permutations
            )
            permutation_indices[process].append(permutation_indices_tmp)
            event_particles[process] = [p for p in event_info.event_particles[process].names]
            num_targets[process].append(event_info.pairing_topology_category[topology_category_name][
                                            "product_symmetry"].degree)

    return {
        'loss': {
            'num_targets': num_targets,
            'event_particles': event_particles,
            'event_permutations': event_info.event_permutations,
        },
        'step': {
            'num_targets': num_targets,
            'event_particles': event_particles,
            'event_permutations': event_info.event_permutations,
            'product_symbolic_groups': event_info.product_symbolic_groups,
            'process_to_topology': event_info.process_to_topology
        }
    }


@time_decorator(name="[Assignment] predict")
def predict(assignments: List[Tensor],
            detections: List[Tensor],
            product_symbolic_groups,
            event_permutations):
    device = assignments[0].device
    assignments_indices = extract_predictions(
        [
            torch.nan_to_num(assignment, nan=-float('inf'))
            for assignment in assignments
        ]
    )

    assignment_probabilities = []
    dummy_index = torch.arange(assignments_indices[0].shape[0])
    for assignment_probability, assignment, symmetries in zip(
            assignments,
            assignments_indices,
            product_symbolic_groups.values()
    ):
        # Get the probability of the best assignment.
        # Have to use explicit function call here to construct index dynamically.
        assignment_probability = assignment_probability.__getitem__((dummy_index, *assignment.T))
        # Convert from log-probability to probability.
        assignment_probability = torch.exp(assignment_probability)

        # Multiply by the symmetry factor to account for equivalent predictions.
        assignment_probability = symmetries.order() * assignment_probability

        # Convert back to cpu and add to database.
        assignment_probabilities.append(assignment_probability)

    final_assignments_indices = []
    final_assignments_probabilities = []
    final_detections_probabilities = []
    for symmetry_group in event_permutations:
        for symmetry_element in symmetry_group:
            symmetry_element = np.sort(np.array(symmetry_element))
            detection_result = detections[symmetry_element[0]]
            softmax = torch.nn.Softmax(dim=-1)

            detection_prob = softmax(detection_result)

            assignment_tmp = torch.stack([assignments_indices[element] for element in symmetry_element])
            assignment_probability_tmp = torch.stack(
                [assignment_probabilities[element] for element in symmetry_element])

            sort_index = torch.argsort(-1 * assignment_probability_tmp, dim=0)
            expanded_sort_index = sort_index.unsqueeze(2)
            expanded_sort_index = expanded_sort_index.expand(-1, -1, assignment_tmp.shape[2])
            assignment_sorted = torch.gather(assignment_tmp, dim=0, index=expanded_sort_index)
            assignment_probability = torch.gather(assignment_probability_tmp, dim=0, index=sort_index)

            init_probabilities = torch.ones_like(assignment_probability[0])
            for iorder in range(len(symmetry_element)):
                final_assignments_indices.append(assignment_sorted[iorder])
                final_assignments_probabilities.append(assignment_probability[iorder])
                detections_probabilities = 1.0 - (detection_prob[:, iorder] / init_probabilities)
                init_probabilities = detections_probabilities
                final_detections_probabilities.append(detections_probabilities)

    return {
        "best_indices": final_assignments_indices,
        "assignment_probabilities": final_assignments_probabilities,
        "detection_probabilities": final_detections_probabilities
    }


class SingleProcessAssignmentMetrics:
    def __init__(
            self,
            device,
            event_permutations,
            event_symbolic_group,
            event_particles,
            product_symbolic_groups,
            ptetaphienergy_index,
            process,
            detection_WP=[0.0, 0.5, 0.8],
            hist_xmin=0,
            hist_xmax=250,
            num_bins=125
    ):

        self.device = device
        self.event_permutations = event_permutations
        self.event_particles = event_particles
        self.event_group = event_symbolic_group
        self.target_groups = product_symbolic_groups
        self.hist_xmin = hist_xmin
        self.hist_xmax = hist_xmax
        self.num_bins = num_bins
        self.detection_WP = detection_WP
        self.ptetaphienergy_index = ptetaphienergy_index
        self.process = process
        self.detection_cut = 0.0

        clusters = []
        cluster_groups = []

        for orbit in self.event_group.orbits():
            orbit = tuple(sorted(orbit))
            names = [self.event_particles[i] for i in orbit]

            names_clean = [name.replace('/', '') for name in names]

            # cluster_name = map(dict.fromkeys, names_clean)
            # cluster_name = map(lambda x: x.keys(), cluster_name)
            # cluster_name = ''.join(reduce(lambda x, y: x & y, cluster_name))

            # Build a list of sets of characters
            char_sets = [set(name) for name in names_clean]

            # Find the common characters
            common_chars = reduce(lambda x, y: x & y, char_sets)

            # Preserve order based on the first name in names_clean
            cluster_name = ''.join([c for c in names_clean[0] if c in common_chars])

            clusters.append((cluster_name, names, orbit))  # ['t', ['t1', 't2'], Orbit]

            cluster_group = self.target_groups[names[0]]
            for name in names:
                assert self.target_groups[name] == cluster_group, (
                    f"Invalid symmetry group for '{name}': expected {self.target_groups[name]}, "
                    f"but got {cluster_group}."
                )

            cluster_groups.append((cluster_name, names, cluster_group))  # ['t', ['t1', 't2'], Group]

        self.clusters = clusters
        self.cluster_groups = cluster_groups

        self.bins = np.linspace(self.hist_xmin, self.hist_xmax, self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])

        self.bins_score = np.linspace(0, 1, self.num_bins + 1)
        self.bin_centers_score = 0.5 * (self.bins_score[:-1] + self.bins_score[1:])

        self.truth_metrics = dict(
            {f"{i + 1}{cluster_name}": {
                "mass": np.zeros(self.num_bins),
                "detection_score": np.zeros(self.num_bins),
                "assignment_score": np.zeros(self.num_bins)
            }
                for cluster_name, particle_name, orbit in self.clusters
                for i in range(len(particle_name))
            })

        self.predict_metrics_correct = dict(
            {f"{i + 1}{cluster_name}": {
                "mass": np.zeros(self.num_bins),
                "detection_score": np.zeros(self.num_bins),
                "assignment_score": np.zeros(self.num_bins),
            }
                for cluster_name, particle_name, orbit in self.clusters
                for i in range(len(particle_name))
            })

        self.predict_metrics_wrong = dict(
            {f"{i + 1}{cluster_name}": {
                "mass": np.zeros(self.num_bins),
                "detection_score": np.zeros(self.num_bins),
                "assignment_score": np.zeros(self.num_bins),
            }
                for cluster_name, particle_name, orbit in self.clusters
                for i in range(len(particle_name))
            })

        self.train_metrics_correct = None
        self.train_metrics_wrong = None

        self.full_log = dict()
        self.particle_max = [len(cluster_indices) for _, _, cluster_indices in self.clusters]
        self.particle_ranges = [list(range(-1, pmax + 1)) for pmax in self.particle_max]

        for event_counts in product(*self.particle_ranges):
            event_mask_name = ""
            cluster_candidate = []
            for mask_count, (cluster_name, _, _) in zip(event_counts, self.clusters):
                mask_count = "*" if mask_count < 0 else str(mask_count)
                event_mask_name = event_mask_name + mask_count + cluster_name
                if not (mask_count == 0):
                    cluster_candidate.append(cluster_name)
            self.full_log[f"{self.process}/Purity/{event_mask_name}/event_proportion"] = {
                "num": 0,
                "den": 0
            }
            self.full_log[f"{self.process}/Purity/{event_mask_name}/event_purity"] = {
                "num": 0,
                "den": 0
            }
            for cluster in cluster_candidate:
                self.full_log[f"{self.process}/Purity/{event_mask_name}/{cluster}_purity"] = {
                    "num": 0,
                    "den": 0,
                }

    @time_decorator(name="[Assignment] update step")
    def update(
            self,
            best_indices,
            assignment_probabilities,
            detection_probabilities,
            truth_indices,
            truth_masks,
            inputs,
            inputs_mask,
            detection_cut=0.5
    ):

        self.detection_cut = detection_cut  # Update the self property

        best_indices, truth_indices = self.sort_outputs(best_indices, truth_indices)  # Remove intra-particle symmetries

        correct_assigned = self.check_correct_assignment(
            best_indices,
            truth_indices,
            truth_masks
        )

        # Log purity
        total_particle_counts, particle_counts, _ = self.particle_count_info(truth_masks)
        for event_counts in product(*self.particle_ranges):
            event_mask = total_particle_counts >= 0

            for particle_count, event_count in zip(particle_counts, event_counts):
                if event_count >= 0:
                    event_mask = event_mask & (particle_count == event_count)
                else:
                    event_mask = event_mask & (total_particle_counts > 0)

            masked_predictions_correct = [p[event_mask] for p in correct_assigned]
            masked_target_masks = [p[event_mask] for p in truth_masks]

            event_mask_name = ""
            for mask_count, (cluster_name, _, _) in zip(event_counts, self.clusters):
                mask_count = "*" if mask_count < 0 else str(mask_count)
                event_mask_name = event_mask_name + mask_count + cluster_name

            event_purity_num, event_purity_den = self.event_purity(masked_predictions_correct, masked_target_masks)
            cluster_purity_num, cluster_purity_den = self.cluster_purity(masked_predictions_correct,
                                                                         masked_target_masks)
            self.full_log[f"{self.process}/Purity/{event_mask_name}/event_proportion"]["num"] += event_mask.sum().item()
            self.full_log[f"{self.process}/Purity/{event_mask_name}/event_proportion"]["den"] += event_mask.size()[0]
            self.full_log[f"{self.process}/Purity/{event_mask_name}/event_purity"]["num"] += event_purity_num
            self.full_log[f"{self.process}/Purity/{event_mask_name}/event_purity"]["den"] += event_purity_den

            for mask_count, (cluster_name, _, _), purity_num, purity_den in zip(event_counts, self.clusters,
                                                                                cluster_purity_num, cluster_purity_den):
                if mask_count == 0:
                    continue
                self.full_log[f"{self.process}/Purity/{event_mask_name}/{cluster_name}_purity"]["num"] += purity_num
                self.full_log[f"{self.process}/Purity/{event_mask_name}/{cluster_name}_purity"]["den"] += purity_den

        # Log mass distribution
        for cluster_name, names, orbit in self.clusters:
            truth_count = torch.stack([truth_masks[iorbit] for iorbit in list(sorted(orbit))], dim=0).int().sum(dim=0)
            truth = torch.stack([truth_indices[iorbit] for iorbit in list(sorted(orbit))], dim=0)
            truth_masking = torch.stack([truth_masks[iorbit] for iorbit in list(sorted(orbit))], dim=0)
            prediction = torch.stack([best_indices[iorbit] for iorbit in list(sorted(orbit))], dim=0)
            predict_detection = torch.stack([detection_probabilities[iorbit] for iorbit in list(sorted(orbit))], dim=0)
            predict_assign_score = torch.stack(
                [assignment_probabilities[iorbit] for iorbit in list(sorted(orbit))], dim=0
            )
            correct_reco = torch.stack([correct_assigned[iorbit] for iorbit in list(sorted(orbit))], dim=0)

            for num_resonance in range(len(names)):
                truth_mask = (truth_count == (num_resonance + 1))
                hist_name = f"{num_resonance + 1}{cluster_name}"
                for local_resonance in range(len(names)):
                    truth_local = truth[local_resonance, :, :]
                    truth_mask_local = truth_mask
                    truth_local = truth_local[truth_mask_local]
                    truth_masking_local = truth_masking[local_resonance, truth_mask_local]
                    if not (truth_local.size()[0] > 0):
                        continue

                    input = inputs[truth_mask_local]
                    input_mask = inputs_mask[truth_mask_local]
                    jet = input[:, :, self.ptetaphienergy_index]

                    # Fill truth metrics
                    truth_mass = reconstruct_mass_peak(
                        jet[truth_masking_local],
                        truth_local[truth_masking_local],
                        input_mask[truth_masking_local])
                    hist, _ = np.histogram(truth_mass.detach().cpu().numpy(), bins=self.bins)
                    self.truth_metrics[hist_name]["mass"] += hist

                    prediction_local = prediction[local_resonance, :, :][truth_mask_local]
                    detection_local = predict_detection[local_resonance, :][truth_mask_local]
                    correct_local = correct_reco[local_resonance, :][truth_mask_local]
                    assign_score_local = predict_assign_score[local_resonance, :][truth_mask_local]

                    # Fill correct metrics
                    predict_correct = prediction_local[correct_local]
                    detection_correct = detection_local[correct_local]
                    assign_score_correct = assign_score_local[correct_local]

                    if prediction_local.size()[0] > 0:
                        reco_mass_correct = reconstruct_mass_peak(
                            jet[correct_local], predict_correct, input_mask[correct_local]
                        )
                        hist, _ = np.histogram(
                            reco_mass_correct[detection_correct > detection_cut].detach().cpu().numpy(), bins=self.bins)
                        self.predict_metrics_correct[hist_name]["mass"] += hist

                        hist, _ = np.histogram(detection_correct.detach().cpu().numpy(), bins=self.bins_score)
                        self.predict_metrics_correct[hist_name]["detection_score"] += hist

                        hist, _ = np.histogram(
                            assign_score_correct[detection_correct > detection_cut].detach().cpu().numpy(),
                            bins=self.bins_score)
                        self.predict_metrics_correct[hist_name]["assignment_score"] += hist

                    prediction_false = prediction_local[~correct_local]
                    detection_false = detection_local[~correct_local]
                    assign_score_false = assign_score_local[~correct_local]
                    if (prediction_false.size()[0] > 0) and (prediction_false >= 0).all():
                        reco_mass_false = reconstruct_mass_peak(
                            jet[~correct_local], prediction_false, input_mask[~correct_local]
                        )
                        hist, _ = np.histogram(
                            reco_mass_false[detection_false > detection_cut].detach().cpu().numpy(),
                            bins=self.bins
                        )
                        self.predict_metrics_wrong[hist_name]["mass"] += hist

                        hist, _ = np.histogram(detection_false.detach().cpu().numpy(), bins=self.bins_score)
                        self.predict_metrics_wrong[hist_name]["detection_score"] += hist

                        hist, _ = np.histogram(
                            assign_score_false[detection_false > detection_cut].detach().cpu().numpy(),
                            bins=self.bins_score)
                        self.predict_metrics_wrong[hist_name]["assignment_score"] += hist

    def check_correct_assignment(
            self,
            prediction,
            target_indices,
            target_masks,
    ):

        result = [torch.zeros_like(prediction[i]).bool() for i in range(len(prediction))]
        for cluster_name, cluster_particles, cluster_indices in self.clusters:
            cluster_target_masks = torch.stack([target_masks[i] for i in cluster_indices])
            cluster_target_indices = torch.stack([target_indices[i] for i in cluster_indices])
            cluster_predictions = torch.stack([prediction[i] for i in cluster_indices])
            correct_predictions = torch.zeros_like(cluster_target_masks, dtype=torch.int64)
            for target_permutation in permutations(range(len(cluster_indices))):
                target_permutation = torch.tensor(
                    target_permutation, dtype=torch.int64,
                    device=cluster_target_masks.device
                )
                prediction_correct = (cluster_predictions == cluster_target_indices[target_permutation])

                prediction_correct = prediction_correct.all(-1) * cluster_target_masks[target_permutation]
                correct_predictions = torch.maximum(prediction_correct, correct_predictions)

            for ilocal, iglobal in enumerate(cluster_indices):
                result[iglobal] = correct_predictions[ilocal, :].bool()

        return result

    def reset(self):
        for name, hist in self.truth_metrics.items():
            for key in hist.keys():
                self.truth_metrics[name][key] = np.zeros(self.num_bins)

        for name, hist in self.predict_metrics_correct.items():
            for key in hist.keys():
                self.predict_metrics_correct[name][key] = np.zeros(self.num_bins)

        for name, hist in self.predict_metrics_wrong.items():
            for key in hist.keys():
                self.predict_metrics_wrong[name][key] = np.zeros(self.num_bins)

        for name in self.full_log:
            for key in self.full_log[name].keys():
                self.full_log[name][key] = 0

        self.train_metrics_correct = None
        self.train_metrics_wrong = None

    def reduce_across_gpus(self):
        if torch.distributed.is_initialized():
            for name, hist in self.truth_metrics.items():
                for key in hist.keys():
                    tensor = torch.tensor(hist[key], dtype=torch.long, device=self.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    self.truth_metrics[name][key] = tensor.cpu().numpy()

            for name, hist in self.predict_metrics_correct.items():
                for key in hist.keys():
                    tensor = torch.tensor(hist[key], dtype=torch.long, device=self.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    self.predict_metrics_correct[name][key] = tensor.cpu().numpy()

            for name, hist in self.predict_metrics_wrong.items():
                for key in hist.keys():
                    tensor = torch.tensor(hist[key], dtype=torch.long, device=self.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    self.predict_metrics_wrong[name][key] = tensor.cpu().numpy()

            for name, log in self.full_log.items():
                for key in log.keys():
                    tensor = torch.tensor(log[key], dtype=torch.long, device=self.device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    self.full_log[name][key] = tensor.cpu().numpy()

    def assign_train_result(self,
                            train_metrics_correct=None,
                            train_metrics_wrong=None):
        self.train_metrics_correct = train_metrics_correct
        self.train_metrics_wrong = train_metrics_wrong

    def plot_mass_spectrum_func(self,
                                truth,
                                predict_correct,
                                predict_wrong,
                                train_predict_correct=None,
                                train_predict_wrong=None,
                                ):

        logs = dict()

        fig, ax = plt.subplots(figsize=(9, 6))
        base_colors = plt.cm.Set2(np.linspace(0.2, 0.8, 2))
        lighter = lambda c: tuple(min(1.0, x + 0.3) for x in c)
        efficiency_detection = (predict_correct.sum() + predict_wrong.sum()) / (truth.sum() + 1e-8)
        text = f"* reco eff (WP: {self.detection_cut}): {efficiency_detection:.2f}"
        # if train_predict_correct is not None:
        #     efficiency_detection_train = (train_predict_correct.sum() + train_predict_wrong.sum())/truth.sum()
        #     text += f"(train: {efficiency_detection_train:.2f})"
        ## Self plot

        total_pred = np.zeros_like(predict_wrong)

        predict_accuracy = predict_correct.sum() / (predict_correct.sum() + predict_wrong.sum() + 1e-8)
        logs["pred_accuracy"] = predict_accuracy

        bin_widths = np.diff(self.bins)
        total_prediction_sum = predict_correct + predict_wrong
        predict_correct = predict_correct / np.maximum(1, total_prediction_sum.sum() * bin_widths)
        predict_wrong = predict_wrong / np.maximum(1, total_prediction_sum.sum() * bin_widths)

        ax.bar(
            self.bin_centers,
            predict_correct,
            width=np.diff(self.bins),
            bottom=total_pred,
            color=base_colors[0],
            alpha=0.6,
            label=f'Reco Success [acc: {predict_accuracy:.2f}]',
        )

        ax.bar(
            self.bin_centers,
            predict_wrong,
            width=np.diff(self.bins),
            bottom=total_pred + predict_correct,
            color=lighter(base_colors[0]),
            alpha=0.5,
            label='Reco False'
        )

        truth = truth / np.maximum(1, truth.sum() * bin_widths)
        ax.plot(self.bin_centers,
                truth,
                linestyle='None',
                marker='o',
                markersize=6,
                label='Truth',
                color='black')

        def gauss(x, a, mu, sigma):
            return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        try:
            popt_truth, _ = curve_fit(gauss, self.bin_centers, truth, p0=[np.max(truth), 100, 10])
            ax.plot(self.bin_centers, gauss(self.bin_centers, *popt_truth), 'k--',
                    label=f'Truth Fit: μ={popt_truth[1]:.2f}, σ={popt_truth[2]:.2f}')
        except RuntimeError:
            print("Truth fit failed")

        try:
            popt_pred, _ = curve_fit(
                gauss, self.bin_centers, predict_correct + predict_wrong,
                p0=[np.max(predict_correct + predict_wrong), 100, 10]
            )
            ax.plot(self.bin_centers, gauss(self.bin_centers, *popt_pred), 'r--',
                    label=f'Pred Fit: μ={popt_pred[1]:.2f}, σ={popt_pred[2]:.2f}')

            logs["pred_mass"] = popt_pred[1]
            logs["pred_resolution"] = popt_pred[2]
        except RuntimeError:
            print("Prediction fit failed")
            logs["pred_mass"] = None
            logs["pred_resolution"] = None

        if train_predict_correct is not None:
            train_accuracy = train_predict_correct.sum() / (train_predict_correct.sum() + train_predict_wrong.sum() + 1e-8)
            logs["train_accuracy"] = train_accuracy
            train_predict_sum = train_predict_correct + train_predict_wrong
            train_predict_correct = train_predict_correct / np.maximum(1, (train_predict_sum).sum() * bin_widths)
            train_predict_wrong = train_predict_wrong / np.maximum(1, (train_predict_sum).sum() * bin_widths)
            ax.plot(
                self.bin_centers,
                train_predict_correct,
                linestyle='None',
                marker='o',
                markersize=6,
                markerfacecolor=base_colors[0],
                markeredgecolor=base_colors[0],
                label=f'Reco Success (train) [acc: {train_accuracy:.2f}]',
            )

            ax.plot(
                self.bin_centers,
                train_predict_correct + train_predict_wrong,
                color=base_colors[0],
                linestyle='None',
                markersize=6,
                marker='o',
                markerfacecolor='none',
                markeredgecolor=base_colors[0],
                label='Total (train)'
            )

            try:
                popt_train, _ = curve_fit(
                    gauss, self.bin_centers, train_predict_correct + train_predict_wrong,
                    p0=[np.max(train_predict_correct + train_predict_wrong), 100, 10]
                )
                ax.plot(self.bin_centers, gauss(self.bin_centers, *popt_train), 'b--',
                        label=f'Pred Fit (Train): μ={popt_train[1]:.2f}, σ={popt_train[2]:.2f}')

                logs["train_mass"] = popt_train[1]
                logs["train_resolution"] = popt_train[2]

            except RuntimeError:
                print("Prediction fit failed")
                logs["train_mass"] = None
                logs["train_resolution"] = None

        custom_text = Line2D([], [], color='none', label=text)
        ax.add_line(custom_text)

        ax.legend()
        fig.tight_layout()
        return fig, logs

    def plot_mass_spectrum(self):
        return_plot = dict()
        return_log = dict()
        for name, hist in self.truth_metrics.items():
            return_plot[f"{name}"], return_log_tmp = self.plot_mass_spectrum_func(
                hist["mass"],
                self.predict_metrics_correct[name]["mass"],
                self.predict_metrics_wrong[name]["mass"],
                self.train_metrics_correct[name]["mass"] if self.train_metrics_correct is not None else None,
                self.train_metrics_wrong[name]["mass"] if self.train_metrics_wrong is not None else None,
            )

            for log_name, log in return_log_tmp.items():
                variable_name = log_name.replace("train_", "").replace("pred_", "")
                if variable_name not in return_log:
                    return_log[variable_name] = dict()
                return_log[variable_name][f"{self.process}/{name}"] = log

        return return_plot, return_log

    def plot_score_func(self,
                        correct_score,
                        false_score,
                        train_correct_score=None,
                        train_false_score=None
                        ):
        fig, ax = plt.subplots(figsize=(8, 6))
        bin_widths = np.diff(self.bins_score)
        ax.bar(
            self.bin_centers_score,
            correct_score / np.maximum(1.0, np.sum(correct_score) * bin_widths),
            width=bin_widths,
            color='C0',
            alpha=0.85,
            label='Correct assign',
            edgecolor='black'
        )
        ax.bar(
            self.bin_centers_score,
            false_score / np.maximum(1.0, np.sum(false_score) * bin_widths),
            width=bin_widths,
            color='C1',
            alpha=0.65,
            label='Wrong assign',
            edgecolor='black'
        )
        # Plot training histogram (b
        if train_correct_score is not None:
            train_bin_widths = np.diff(self.bins_score)
            ax.plot(self.bin_centers_score,
                    train_correct_score / np.maximum(1, train_correct_score.sum() * train_bin_widths),
                    linestyle='None',
                    marker='o',
                    linewidth=3,
                    markersize=6,
                    label='Correct assign (train)',
                    color='C0')
            ax.plot(self.bin_centers_score,
                    train_false_score / np.maximum(1, train_false_score.sum() * train_bin_widths),
                    linestyle='None',
                    marker='o',
                    linewidth=3,
                    markersize=3,
                    label='Wrong assign (train)',
                    color='C1'
                    )

        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_title("Score Distribution")
        ax.legend(loc="best")
        ax.grid(True)
        fig.tight_layout()

        return fig

    def plot_score(self, target="detection_score"):
        return_plot = dict()
        for name, _ in self.truth_metrics.items():
            return_plot[f"{name}"] = self.plot_score_func(
                self.predict_metrics_correct[name][target],
                self.predict_metrics_wrong[name][target],
                self.train_metrics_correct[name][target] if self.train_metrics_correct is not None else None,
                self.train_metrics_wrong[name][target] if self.train_metrics_wrong is not None else None,
            )
        return return_plot

    def summary_log(self):
        return_log = dict()

        for name, log in self.full_log.items():
            purity = log["num"] / log["den"] if log["den"] > 0 else 0
            return_log[name] = purity
        return return_log

    @staticmethod
    def permute_arrays(self, array_list, permutation):
        return [array_list[index] for index in permutation]

    def sort_outputs(self, predictions, targets):
        """
        :param predictions:
        :param targets:
        :return:
        Sort all of the targets and predictions to avoid any intra-particle symmetries
        """

        predictions = [torch.clone(p) for p in predictions]
        targets = [torch.clone(p) for p in targets]
        for i, (_, particle_group) in enumerate(self.target_groups.items()):
            for orbit in particle_group.orbits():
                orbit = tuple(sorted(orbit))

                targets[i][:, orbit] = torch.sort(targets[i][:, orbit], dim=1)[0]
                predictions[i][:, orbit] = torch.sort(predictions[i][:, orbit], dim=1)[0]
        return predictions, targets

    def particle_count_info(self, target_masks):
        """
        return:
            total_particle_counts: [B,]: total valid resonance particles
            particle_counts: [[B,] x num_clusters]: valid resonance particles in each cluster
            particle_max: [x num_clusters] : max resonance particles in each cluster
        """
        target_masks = torch.stack(target_masks, dim=0)

        total_particle_counts = target_masks.sum(0)

        particle_counts = [target_masks[list(cluster_indices)].sum(0)
                           for _, _, cluster_indices in self.clusters]
        particle_max = [len(cluster_indices) for _, _, cluster_indices in self.clusters]

        return total_particle_counts, particle_counts, particle_max

    def cluster_purity(self, prediction_correct, target_masks):

        results_num = []
        results_den = []
        for cluster_name, cluster_particles, cluster_indices in self.clusters:
            cluster_prediction_correct = torch.stack([prediction_correct[i] for i in cluster_indices])
            cluster_target_masks = torch.stack([target_masks[i] for i in cluster_indices])
            total_particles = cluster_target_masks.sum()
            if total_particles > 0:
                results_num.append(cluster_prediction_correct.sum().item())
                results_den.append(total_particles.sum().item())
            else:
                results_num.append(0)
                results_den.append(0)

        return results_num, results_den

    def event_purity(self, prediction_correct, target_masks):
        target_masks = torch.stack(target_masks, dim=0)
        num_particles_in_event = target_masks.sum(0)
        event_prediction_correct = torch.stack(prediction_correct, dim=0)
        event_accuracy_num = (event_prediction_correct.sum(0) == num_particles_in_event)
        event_accuracy_den = num_particles_in_event.size()[0]
        return event_accuracy_num.sum().item(), event_accuracy_den


@time_decorator(name="[Assignment] shared_step")
def shared_step(
        ass_loss_fn,
        loss_dict,
        loss_detailed_dict,
        assignment_loss_scale,
        detection_loss_scale,
        process_names,
        assignments,
        detections,
        targets,
        targets_mask,
        product_symbolic_groups,
        event_permutations,
        batch_size,
        device,
        event_particles,
        num_targets,
        point_cloud,
        point_cloud_mask,
        subprocess_id,
        metrics: dict[str, SingleProcessAssignmentMetrics],
        process_to_topology: dict[str, dict[str, float]],
        update_metric: bool = True,
        event_weight = None
):
    num_processes = len(event_permutations)

    loss_detailed_dict.update({
        "assignment": {},
        "detection": {},
    })

    symmetric_losses = ass_loss_fn(
        assignments=assignments,
        detections=detections,
        targets=targets,
        targets_mask=targets_mask,
        process_id=subprocess_id,
        event_weight=event_weight
    )

    assignment_predict = dict()
    ass_target_metric, ass_mask_metric = convert_target_assignment(
        targets=targets,
        targets_mask=targets_mask,
        event_particles=event_particles,
        num_targets=num_targets
    )

    assignment_loss = torch.zeros(1, device=device, requires_grad=True)
    detected_loss = torch.zeros(1, device=device, requires_grad=True)
    active_heads_sum = {k: 0 for k in loss_dict.keys() if 'assignment_' in k}
    for process in process_names:

        # debug_nonfinite_batch(
        #     {
        #         "assignments": assignments[process],
        #         "detections": detections[process],
        #     },
        #     batch_dim=0, name=f"ass/{process}", logger=logger
        # )

        assignment_predict[process] = predict(
            assignments=assignments[process],
            detections=detections[process],
            product_symbolic_groups=product_symbolic_groups[process],
            event_permutations=event_permutations[process],
        )

        if update_metric:
            metrics[process].update(
                best_indices=assignment_predict[process]["best_indices"],
                assignment_probabilities=assignment_predict[process]["assignment_probabilities"],
                detection_probabilities=assignment_predict[process]["detection_probabilities"],
                truth_indices=ass_target_metric[process],
                truth_masks=ass_mask_metric[process],
                inputs=point_cloud,
                inputs_mask=point_cloud_mask,
            )

        loss_detailed_dict["assignment"][process] = symmetric_losses["assignment"][process]
        loss_detailed_dict["detection"][process] = symmetric_losses["detection"][process]

        assignment_loss = assignment_loss + symmetric_losses["assignment"][process]
        detected_loss = detected_loss + symmetric_losses["detection"][process]

        for topology, topo_weight in process_to_topology[process].items():
            ass_name = f'assignment_{topology}'
            ass_total = symmetric_losses["assignment"][process] + symmetric_losses["detection"][process]
            loss_dict[ass_name] = loss_dict[ass_name] + topo_weight * ass_total
            active_heads_sum[ass_name] += topo_weight

    loss_dict['assignment'] = (assignment_loss / num_processes * assignment_loss_scale).mean()
    loss_dict['detection'] = (detected_loss / num_processes * detection_loss_scale).mean()

    for actives, active_sum in active_heads_sum.items():
        loss_dict[actives] = loss_dict[actives] / active_sum
        # del loss_dict[actives]
        # loss_dict[actives.replace("assignment_", "")] = value

    total_loss = loss_dict['assignment'] + loss_dict['detection']

    return total_loss, assignment_predict


@time_decorator(name="[Assignment] shared_epoch_end")
def shared_epoch_end(
        global_rank,
        metrics_valid,
        metrics_train,
        logger,
):
    for process in metrics_valid:
        metrics_valid[process].reduce_across_gpus()
    if metrics_train:
        for process in metrics_train:
            metrics_train[process].reduce_across_gpus()
    if global_rank == 0:

        for process in metrics_valid:

            logs = metrics_valid[process].summary_log()
            logger.log(logs)

            # if metrics_train[process] is not None:
            #     training_logs = metrics_train[process].summary_log()
            #     training_logs = {
            #         f"{name}-train": log
            #         for name, log in training_logs.items()
            #     }
            #     logger.log(training_logs)

            metrics_valid[process].assign_train_result(
                metrics_train[process].predict_metrics_correct,
                metrics_train[process].predict_metrics_wrong,
            )

            figs, logs = metrics_valid[process].plot_mass_spectrum()
            for name, log_set in logs.items():
                for set_name, log in log_set.items():
                    logger.log({
                        f"{set_name}/{name}": log}
                    )
            logger.log({
                f"assignment_reco_mass/{process}/{name}": wandb.Image(fig)
                for name, fig in figs.items()
            })
            for _, fig in figs.items():
                plt.close(fig)

            figs = metrics_valid[process].plot_score(target="detection_score")
            wandb.log({
                f"assignment_reco_detection/{process}/{name}": wandb.Image(fig)
                for name, fig in figs.items()
            })
            for _, fig in figs.items():
                plt.close(fig)

            figs = metrics_valid[process].plot_score(target="assignment_score")
            wandb.log({
                f"assignment_score/{process}/{name}": wandb.Image(fig)
                for name, fig in figs.items()
            })
            for _, fig in figs.items():
                plt.close(fig)

    for _, metric in metrics_valid.items():
        metric.reset()

    if metrics_train:
        for _, metric in metrics_train.items():
            metric.reset()

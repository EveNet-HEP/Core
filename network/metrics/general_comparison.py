import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed
from collections import defaultdict


class GenericMetrics:
    def __init__(self, style_config=None):
        self.valid = defaultdict(lambda: defaultdict(list))
        self.train = defaultdict(lambda: defaultdict(list))

        self.epoch_buffer = {
            "train": defaultdict(lambda: defaultdict(list)),
            "valid": defaultdict(lambda: defaultdict(list)),
        }

        self.style_config = style_config or {
            "train": {"linestyle": "-", "marker": "x", "alpha": 0.75, "linewidth": 2},
            "valid": {"linestyle": "--", "marker": "o", "alpha": 0.9, "linewidth": 1.5},
            "colors": [
                "#2D8875", "#52AADC", "#7C7979", "#7FABD1",
                "#91ccc0", "#963B79", "#97D0C5", "#B5CE4E",
                "#BD7795", "#C7C1DE", "#C89736", "#D75B4E",
                "#EC6E66", "#EEB6D4", "#F39865", "#F7AC53"
            ]
        }

    def update(self, new_data: dict, is_train=False):
        buffer = self.epoch_buffer["train" if is_train else "valid"]

        for METRIC, values_dict in new_data.items():
            for label, value in values_dict.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                buffer[METRIC][label].append(value)

    def finalize_epoch(self, is_train=False):
        buffer = self.epoch_buffer["train" if is_train else "valid"]
        target = self.train if is_train else self.valid

        for METRIC, values_dict in buffer.items():
            for label, values in values_dict.items():
                values = np.array(values)
                mean_value = np.mean(values)
                target[METRIC][label].append(np.array([mean_value]))

        # Clear the buffer
        self.epoch_buffer["train" if is_train else "valid"] = defaultdict(lambda: defaultdict(list))

    def reduce_across_gpus(self, device):
        if not torch.distributed.is_initialized():
            return

        def reduce_storage(storage):
            reduced = defaultdict(lambda: defaultdict(list))
            for metric, values_dict in storage.items():
                for label, arrays in values_dict.items():
                    stacked = np.stack(arrays)
                    tensor = torch.tensor(stacked, dtype=torch.float32, device=device)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    tensor /= torch.distributed.get_world_size()
                    reduced[metric][label] = [tensor.cpu().numpy()]
            return reduced

        self.valid = reduce_storage(self.valid)
        self.train = reduce_storage(self.train)

    def reset(self):
        self.valid.clear()
        self.train.clear()

    def plot_all(self):
        results = {}
        for metric in sorted(set(self.valid) | set(self.train)):
            fig, ax = plt.subplots(figsize=(18, 12), dpi=100)
            color_map = self.style_config["colors"]

            labels = sorted(set(self.valid.get(metric, {})) | set(self.train.get(metric, {})))
            for idx, label in enumerate(labels):
                color = color_map[idx % len(color_map)]

                # Plot training data
                if label in self.train.get(metric, {}):
                    y = np.concatenate(self.train[metric][label])
                    x = np.arange(len(y))
                    ax.plot(
                        x, y,
                        label=f"{label}",
                        color=color,
                        linestyle=self.style_config["train"]["linestyle"],
                        marker=self.style_config["train"]["marker"],
                        alpha=self.style_config["train"]["alpha"]
                    )

                # Plot validation data
                if label in self.valid.get(metric, {}):
                    y = np.concatenate(self.valid[metric][label])
                    x = np.arange(len(y))
                    ax.plot(
                        x, y,
                        # label=f"{label} (Valid)",
                        color=color,
                        linestyle=self.style_config["valid"]["linestyle"],
                        marker=self.style_config["valid"]["marker"],
                        alpha=self.style_config["valid"]["alpha"]
                    )

            ax.set_title(f"{metric} Comparison")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.legend(
                loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False
            )
            ax.grid(True)
            # set log y
            ax.set_yscale('log')
            fig.tight_layout(rect=[0, 0, 0.98, 1])  # leave space on the right
            results[metric] = fig

        return results


if __name__ == '__main__':
    # Instantiate the plotter
    plotter = GenericMetrics()

    # Simulate dummy updates
    for step in range(100):
        # Create dummy data
        train_data = {
            "loss": {
                "process_A": np.array(1.0 / (step + 1) + np.random.randn() * 0.01),
                "process_B": np.array(1.2 / (step + 1) + np.random.randn() * 0.01),
            },
            "resolution": {
                "process_A": np.array(0.5 + np.random.randn() * 0.01),
                "process_B": np.array(0.4 + np.random.randn() * 0.01),
            }
        }

        valid_data = {
            "loss": {
                "process_A": np.array([1.1 / (step + 1) + np.random.randn() * 0.02]),
                "process_B": np.array([1.3 / (step + 1) + np.random.randn() * 0.02]),
            },
            "resolution": {
                "process_A": np.array([0.52 + np.random.randn() * 0.02]),
                "process_B": np.array([0.42 + np.random.randn() * 0.02]),
            }
        }

        # Update the plotter
        plotter.update(train_data, is_train=True)
        plotter.update(valid_data, is_train=False)

        if step % 10 == 0:
            plotter.finalize_epoch(is_train=True)
            plotter.finalize_epoch(is_train=False)

    plotter.finalize_epoch(is_train=False)
    plotter.finalize_epoch(is_train=True)

    # Plot the results
    figs = plotter.plot_all()

    # Display the plots
    for metric, fig in figs.items():
        print(f"Showing plot for {metric}")
        fig.show()
        # plt.close(fig)

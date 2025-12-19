import torch
import torch.nn as nn


class GradNormController(nn.Module):
    def __init__(self, task_names: list[str], alpha: float = 0.12, learning_rate: float = 1e-3):
        """
        Args:
            task_names (list[str]): Names of loss heads/tasks
            alpha (float): GradNorm alpha parameter (controls correction strength)
            lr (float): Learning rate for updating loss weights
        """
        super().__init__()
        self.alpha: float = alpha
        self.task_names: list[str] = task_names

        self.loss_weights: nn.ParameterDict = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
            for name in task_names
        })

        self.loss_weight_optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.loss_weights.parameters(), lr=learning_rate
        )

        self.initial_losses: dict[str, torch.Tensor] = {}

    def get_weighted_losses(self, loss_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: self.loss_weights[name] * loss_dict[name]
            for name in self.task_names
        }

    def compute_gradnorm_loss(
            self,
            loss_dict: dict[str, torch.Tensor],
            head_modules: dict[str, nn.Module],
            step: int,
            epoch: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            loss_dict (dict[str, Tensor]): Unweighted per-task losses
            head_modules (dict[str, nn.Module]): Dict of task heads (used to compute gradient norms)
            step (int): Global step
            epoch (int): Current epoch

        Returns:
            Tuple of:
                - total weighted loss
                - GradNorm loss
                - Dictionary of gradient norms per task
        """
        if epoch == 0 and step == 0:
            self.initial_losses = {
                k: v.detach().clone() for k, v in loss_dict.items()
            }

        print(self.initial_losses)
        print(head_modules.keys())

        grads: dict[str, torch.Tensor] = {}
        weighted_losses: dict[str, torch.Tensor] = {
            name: self.loss_weights[name] * loss_dict[name]
            for name in head_modules.keys()
        }

        print(weighted_losses)

        for name, weighted_loss in weighted_losses.items():
            head_modules[name].zero_grad(set_to_none=True)
            weighted_loss.backward(retain_graph=True)

            norm: torch.Tensor = torch.norm(torch.stack([
                p.grad.norm(2) for p in head_modules[name].parameters()
                if p.grad is not None and p.requires_grad
            ]), 2)

            grads[name] = norm.detach()

        total_weighted_loss: torch.Tensor = sum(weighted_losses.values())

        current_losses: dict[str, torch.Tensor] = {
            k: v.detach() for k, v in loss_dict.items() if k in head_modules
        }
        avg_loss_ratio: torch.Tensor = torch.stack([
            current_losses[k] / self.initial_losses[k] for k in head_modules
        ]).mean()

        targets: dict[str, torch.Tensor] = {
            k: grads[k] * ((current_losses[k] / self.initial_losses[k]) / avg_loss_ratio) ** self.alpha
            for k in head_modules
        }

        G: torch.Tensor = torch.stack([grads[k] for k in head_modules])
        T: torch.Tensor = torch.stack([targets[k] for k in head_modules])
        grad_norm_loss: torch.Tensor = nn.functional.l1_loss(G, T)

        return total_weighted_loss, grad_norm_loss, grads

    def step(self, grad_norm_loss: torch.Tensor) -> None:
        self.loss_weight_optimizer.zero_grad()
        grad_norm_loss.backward()
        self.loss_weight_optimizer.step()

    def get_weights(self) -> dict[str, float]:
        return {k: v.item() for k, v in self.loss_weights.items()}

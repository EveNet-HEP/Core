import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, List


class FAMO(nn.Module):
    # https://github.com/Cranial-XIX/FAMO/tree/main
    def __init__(
            self,
            task_list: List[str],
            device: torch.device,
            lr: float = 0.025,
            gamma: float = 0.01,
            turn_on: bool = True,
            logits_bound: float = 1.0,
            eps: float = 1e-8
    ):
        super().__init__()
        self.device = device
        self.task_list = task_list
        self.turn_on = turn_on
        self.logits_bound = logits_bound
        self.eps = eps

        # Parameters (logits) for each task
        self.w = nn.ParameterDict({
            k: nn.Parameter(torch.tensor([0.0], device=device))
            for k in task_list
        })

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=gamma)
        self.min_losses = {k: torch.tensor([0.0], device=device) for k in task_list}

        # Cache for update step
        self.prev_task_list = []
        self.prev_losses = None

    def step(self, loss_dict: Dict[str, torch.Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        default_log = {
            **{f'logits-{task}': self.w[task] for task in self.w.keys()},
            **{f'weights-{task}': torch.tensor(0.0, device=self.device) for task in self.w.keys()},
            'entropy': torch.tensor(0.0, device=self.device),
        }

        # Filter out zero-loss tasks
        filtered_loss_dict = {
            k: v for k, v in loss_dict.items() if v.detach().flatten()[0] > self.eps
        }

        if not filtered_loss_dict:  # all losses are 0
            return torch.tensor(0.0, device=self.device), default_log

        self.prev_task_list = [k for k in self.task_list if k in filtered_loss_dict]
        losses = torch.stack([filtered_loss_dict[k].flatten()[0] for k in self.prev_task_list])  # [N]

        if not self.turn_on:
            return losses.sum(), default_log

        logits = torch.stack([self.w[k] for k in self.prev_task_list]).squeeze(-1)
        logits = self.logits_bound * torch.tanh(logits)
        weights = F.softmax(logits, dim=0)  # [N]

        min_vec = torch.stack([self.min_losses[k].flatten()[0] for k in self.prev_task_list])
        D = losses - min_vec + 1e-8
        c = (weights / D).sum().detach()

        self.prev_losses = losses.detach()

        # prevent model backward from updating FAMO params
        weights_for_loss = weights.detach()
        weighted_loss = (D.log() * weights_for_loss / c).sum()

        # print(f"raw losses: {loss_dict}")
        # print(f"losses: {losses} min_vec: {min_vec} D: {D} c: {c} weighted_loss: {weighted_loss}")
        # print(f"weights: {weights} logits: {logits}")

        # Overwrite active tasks only
        for i, task in enumerate(self.prev_task_list):
            default_log[f'logits-{task}'] = logits[i]
            default_log[f'weights-{task}'] = weights[i]

        # Update entropy
        default_log['entropy'] = -(weights * weights.log()).sum()

        return weighted_loss, default_log


    def update(self, current_loss_dict: Dict[str, torch.Tensor]) -> None:

        if not self.turn_on or not self.prev_task_list:
            return

        delta = []
        for k in self.prev_task_list:
            prev = self.prev_losses[self.prev_task_list.index(k)]
            curr = current_loss_dict[k].detach().flatten()[0]
            delta.append(
                torch.log(prev - self.min_losses[k].flatten()[0] + 1e-8)
                - torch.log(curr - self.min_losses[k].flatten()[0] + 1e-8)
            )

            # print(f"prev: {prev} curr: {curr} delta: {delta[-1]}")
        delta = torch.stack(delta)  # [N]

        with torch.enable_grad():
            logits = torch.stack([self.w[k] for k in self.prev_task_list]).squeeze(-1)
            logits = self.logits_bound * torch.tanh(logits)

            weights = F.softmax(logits, dim=0)

            # print(f"logits: {logits} weights: {weights}")

            grads = torch.autograd.grad(weights, [logits], grad_outputs=delta, retain_graph=False)[0]

        self.optimizer.zero_grad()
        for i, k in enumerate(self.prev_task_list):
            self.w[k].grad = grads[i].unsqueeze(0)  # [1]

        self.optimizer.step()

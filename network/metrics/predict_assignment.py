import torch
from typing import List, Tuple


def compute_strides(num_partons: int, max_jets: int, max_partons: int, device) -> torch.Tensor:
    strides = torch.full((max_partons,), -1, dtype=torch.long, device = device)
    strides[-1 - (max_partons - num_partons)] = 1
    for i in range(num_partons - 2, -1, -1):
        strides[i] = strides[i + 1] * max_jets
    return strides


def unravel_index(index: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:

    """
    index: (batch_size,)
    """
    batch_size = index.shape[0]
    num_partons = strides.shape[1]
    result = torch.zeros((batch_size, num_partons), dtype=torch.long, device=strides.device)
    remainder = index
    for i in range(num_partons):
        result[:, i] = remainder // strides[:, i]
        remainder %= strides[:, i]

    result = torch.where(strides > 0, result, -1)
    return result


def ravel_index(index: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
    raw_index = index * strides
    raw_index = torch.where(strides > -1, raw_index, 0)
    return torch.sum(raw_index, dim=1)


def maximal_prediction(predictions: List[torch.Tensor]) -> Tuple[int, int, float]:
    batch_size = predictions[0].shape[0]
    device = predictions[0].device
    best_value = torch.full((batch_size,), -float('inf'), device = device, dtype = torch.float)
    best_prediction = torch.full((batch_size,), -1, device = device, dtype = torch.long)
    best_jet = torch.full((batch_size,), -1, device = device, dtype = torch.long)


    for i, pred in enumerate(predictions):
        max_value, max_jet = torch.max(pred, dim=1) # max_value.shape = (batch_size,)
        best_prediction = torch.where(max_value > best_value,  torch.full_like(best_prediction, i), best_prediction)
        best_jet = torch.where(max_value > best_value, max_jet, best_jet)
        best_value = torch.where(max_value > best_value, max_value, best_value)

    return best_jet, best_prediction, best_value


def mask_jet(data: torch.Tensor, num_partons: int, max_jets: int, index: torch.Tensor, value: float):
    batch_size = data.shape[0]
    if num_partons == 1:
        data[torch.arange(batch_size), index] = value
    elif num_partons == 2:
        data = data.reshape((batch_size, max_jets, max_jets))
        data[torch.arange(batch_size), index, :] = value
        data[torch.arange(batch_size), :, index] = value
    elif num_partons == 3:
        data = data.reshape((batch_size, max_jets, max_jets, max_jets))
        data[torch.arange(batch_size), index, :, :] = value
        data[torch.arange(batch_size), :, index, :] = value
        data[torch.arange(batch_size), :, :, index] = value
    else:
        raise NotImplementedError("num_partons > 3 not yet implemented in PyTorch version")


def extract_prediction(predictions: List[torch.Tensor], num_partons: torch.Tensor, max_jets: int) -> torch.Tensor:
    float_neg_inf = -float('inf')
    max_partons = num_partons.max().item()
    num_targets = len(predictions)
    batch_size = predictions[0].shape[0]

    strides = [compute_strides(num_partons[i].item(), max_jets, max_partons, predictions[0].device) for i in range(num_targets)]
    strides = torch.stack([s.unsqueeze(0).expand(batch_size, -1) for s in strides])

    results = torch.full((num_targets,batch_size, max_partons), -2, dtype=torch.long, device = strides.device)

    for _ in range(num_targets):
        best_jet, best_prediction, best_value = maximal_prediction(predictions)


        best_jets = unravel_index(best_jet, strides[best_prediction, torch.arange(batch_size), :])
        results[best_prediction, torch.arange(batch_size), :] = best_jets
        results[:,~torch.isfinite(best_value), :] = -2 # Do not return good predictions if invalid jets comes in

        for i in range(num_targets):
            predictions[i] = predictions[i] + (torch.where(best_prediction == i, float_neg_inf, 0)).unsqueeze(1)
            for i_parton in range(num_partons[i].item()):
                jet = best_jets.reshape(batch_size, -1)[:, i_parton] # (batch_size,)
                mask_jet(predictions[i], num_partons[i].item(), max_jets, jet, float_neg_inf)

    return results


def _extract_predictions(predictions: List[torch.Tensor], num_partons: torch.Tensor, max_jets: int,
                         batch_size: int) -> torch.Tensor:
    num_targets = len(predictions)
    max_partons = num_partons.max().item()
    output = torch.zeros((batch_size, num_targets, max_partons), dtype=torch.long)

    output = extract_prediction(predictions, num_partons, max_jets)

    return output.contiguous()


def extract_predictions(predictions: List[torch.Tensor]) -> List[torch.Tensor]:
    flat_predictions = [p.reshape(p.shape[0], -1) for p in predictions]
    num_partons = torch.tensor([p.ndim - 1 for p in predictions], dtype=torch.long, device=predictions[0].device)
    max_jets = max(max(p.shape[1:])  for p in predictions)
    batch_size = max(p.shape[0] for p in predictions)

    results = _extract_predictions(flat_predictions, num_partons, max_jets, batch_size)
    return [results[i, :, :num_partons[i]] for i in range(len(predictions))]

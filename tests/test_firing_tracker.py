import torch

from model_diffing.scripts.firing_tracker import FiringTracker


def test_firing_tracker():
    A = 10
    L = 100
    firing_tracker = FiringTracker(A, L, torch.device("cpu"))

    baseline_BA = []
    threshold = torch.rand((A,))
    print(f"threshold: {threshold}")
    for _ in range(5):
        x_BA = torch.rand((32, A)) > threshold
        firing_tracker.add_batch(x_BA)
        baseline_BA.append(x_BA)

    actual_thresholds_LA = torch.cat(
        [
            baseline_BA[-4][-4:],
            baseline_BA[-3],
            baseline_BA[-2],
            baseline_BA[-1],
        ],
        dim=0,
    )
    assert actual_thresholds_LA.shape == (L, A)

    mean_1 = actual_thresholds_LA.float().mean(dim=0)
    mean_2 = firing_tracker.firing_percentage_A()

    assert torch.allclose(mean_1, mean_2), f"max diff: {torch.max(torch.abs(mean_1 - mean_2))}. {mean_1} != {mean_2}"

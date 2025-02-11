import torch


class FiringTracker:
    # shapes:
    # L: length
    # A: activation_size

    def __init__(self, activation_size: int, length: int, device: torch.device):
        """
        Args:
            activation_size (int): Number of features (activations) per record.
            length (int): The number of time steps to keep (the window size).
            device (torch.device): Device to allocate tensors on.
        """
        self._activation_size = activation_size
        self._max_length = length
        """max length of the buffer"""
        self._ptr = 0
        """Pointer to the next slot in the circular buffer. Between 0 and `_max_length - 1`."""
        self._count = 0
        """number of activations recorded. Saturates at `_max_length`."""
        self._buffer_LA = torch.zeros(length, activation_size, dtype=torch.bool, device=device)
        """holds the last L boolean activation vectors."""
        self._running_window_sum_A = torch.zeros(activation_size, dtype=torch.int64, device=device)
        """Running sum for each activation over the last L activations."""

    def add_batch(self, firing_BA: torch.Tensor) -> None:
        firing_BA = firing_BA.bool().long()
        batch_size = firing_BA.shape[0]

        if self._ptr + batch_size > self._max_length:
            # we would wrap. No need to handle this, just split and handle as 2 batches
            break_idx = self._max_length - self._ptr
            self.add_batch(firing_BA[:break_idx])
            self.add_batch(firing_BA[break_idx:])
            return

        section_to_overwrite_BA = self._buffer_LA[self._ptr : self._ptr + batch_size]
        self._running_window_sum_A -= section_to_overwrite_BA.sum(dim=0)
        self._running_window_sum_A += firing_BA.sum(dim=0)
        section_to_overwrite_BA[:] = firing_BA

        self._ptr = (self._ptr + batch_size) % self._max_length

        if self._count < self._max_length:  # hasn't wrapped yet
            self._count += batch_size

    def firing_percentage_A(self) -> torch.Tensor:
        """
        Returns a float tensor of shape (activation_size,) where each element is the
        fraction of times the corresponding activation was True in the current window.
        """
        if self._count == 0:
            raise ValueError("No activations have been recorded yet.")
        return self._running_window_sum_A.float() / self._count




class SimpleFiringTracker:
    def __init__(self, activation_size: int, device: torch.device):
        self._activation_size = activation_size
        self.steps_since_fired_A = torch.zeros(activation_size, dtype=torch.int64, device=device)

    def add_batch(self, firing_BA: torch.Tensor) -> None:
        firing_A = firing_BA.bool().any(dim=0)
        self.steps_since_fired_A[firing_A] = 0
        self.steps_since_fired_A[~firing_A] += 1
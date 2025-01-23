import random
from collections.abc import Iterator

import torch

from model_diffing.log import logger


# B = "batch"
# Bf = "buffer size"
# X = "arbitrary shape"
def batch_shuffle_tensor_iterator_BX(
    tensor_iterator_X: Iterator[torch.Tensor],  # X here means "arbitrary shape"
    shuffle_buffer_size: int,
    yield_batch_size: int,
) -> Iterator[torch.Tensor]:
    if shuffle_buffer_size < yield_batch_size:
        raise ValueError(
            f"shuffle_buffer_size must be greater than yield_batch_size, but got {shuffle_buffer_size=} and {yield_batch_size=}"
        )
    if shuffle_buffer_size < yield_batch_size * 4:
        logger.warning("shuffle_buffer_size is less than 4x yield_batch_size, this may lead to poor shuffling")

    first_tensor_X = next(tensor_iterator_X)  # this "wastes" an example. This is ok.

    buffer_BfX = torch.empty(
        (shuffle_buffer_size, *first_tensor_X.shape),
        device=first_tensor_X.device,
        dtype=first_tensor_X.dtype,
    )
    buffer_size_bytes = buffer_BfX.numel() * buffer_BfX.element_size()
    logger.info(f"shuffle buffer size: {buffer_size_bytes / 1e9:.2f} GB")

    buffer_BfX[0] = first_tensor_X
    available_indices = {0}
    stale_indices = set(range(1, shuffle_buffer_size))

    def sample_BX():
        batch_indices = random.sample(list(available_indices), yield_batch_size)
        available_indices.difference_update(batch_indices)
        stale_indices.update(batch_indices)
        return buffer_BfX[batch_indices]

    while True:
        # refill buffer
        for stale_idx, example_X in zip(list(stale_indices), tensor_iterator_X, strict=False):
            buffer_BfX[stale_idx] = example_X
            available_indices.add(stale_idx)
            stale_indices.remove(stale_idx)

        if len(available_indices) <= shuffle_buffer_size // 2:
            # This means the buffer wasn't refilled above. therefore the iterator is exhausted
            # so we yield the remaining activations
            while len(available_indices) >= yield_batch_size:
                yield sample_BX()
            break

        # yield batches until buffer is half empty
        while len(available_indices) > shuffle_buffer_size // 2:
            yield sample_BX()

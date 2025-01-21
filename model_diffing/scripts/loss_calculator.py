# import torch
# from transformer_lens import HookedTransformer

# from model_diffing.dataloader.activations import ModelRunner


# class LossCalculator:
#     def __init__(self, model_runner: ModelRunner):
#         self.model_runner = model_runner

#     def calculate_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return self.model_runner.get_activations_BSMLD(input) - target

from .jumprelu import JumpReLUActivation
from .relu import ReLUActivation
from .topk import BatchTopkActivation, TopkActivation

ACTIVATIONS = {
    "TopkActivation": TopkActivation,
    "BatchTopkActivation": BatchTopkActivation,
    "ReLU": ReLUActivation,
    "JumpReLU": JumpReLUActivation,
}

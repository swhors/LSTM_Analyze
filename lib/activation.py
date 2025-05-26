from enum import Enum
class ActivationOutput(Enum):
    linear = "linear" # No activation, output is directly passed through.
    relu = "relu" # Rectified Linear Unit, max(x, 0).
    sigmoid = "sigmoid" # Sigmoid function, output between 0 and 1.
    tanh = "tanh" # Hyperbolic tangent, output between -1 and 1.
    softmax = "softmax" # Normalizes output to a probability distribution.
    elu = "elu" # Exponential Linear Unit.
    selu = "selu" # Scaled Exponential Linear Unit.


class RecurrentActivation(Enum):
    sigmoid = "sigmoid" # Commonly used for gates in LSTM.
    sard_sigmoid = "hard_sigmoid" # hard_sigmoid
    tanh = "tanh" # Can be used, but sigmoid is more typical for gates.
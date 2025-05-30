"""
common_env_sets.py
"""
from lib.activation import ActivationOutput, RecurrentActivation
from datetime import datetime


"""
Activation (Output):
    linear: No activation, output is directly passed through.
    relu: Rectified Linear Unit, max(x, 0).
    sigmoid: Sigmoid function, output between 0 and 1.
    tanh: Hyperbolic tangent, output between -1 and 1.
    softmax: Normalizes output to a probability distribution.
    elu: Exponential Linear Unit.
    selu: Scaled Exponential Linear Unit.

Recurrent Activation:
    sigmoid: Commonly used for gates in LSTM.
    hard_sigmoid: A faster, less computationally expensive version of sigmoid.
    tanh: Can be used, but sigmoid is more typical for gates.
"""


lstm_units=[[(45, ActivationOutput.selu.name),   # 0
             (45, ActivationOutput.selu.name),
             (45, ActivationOutput.sigmoid.name),
             (45, ActivationOutput.elu.name)
             ],
            [(45, ActivationOutput.selu.name),   # 1
             (45, ActivationOutput.selu.name)
             ],
            [(45, ActivationOutput.selu.name),   # 2
             (45, ActivationOutput.selu.name),
             (45, ActivationOutput.selu.name),
             (45, ActivationOutput.selu.name),
             (45, ActivationOutput.selu.name),
             (45, ActivationOutput.selu.name),
             (45, ActivationOutput.selu.name),
             (45, ActivationOutput.sigmoid.name),
             (45, ActivationOutput.sigmoid.name),
             (45, ActivationOutput.elu.name)
             ],
            [(45, ActivationOutput.tanh.name),   # 3
             (45, ActivationOutput.tanh.name),
             (45, ActivationOutput.tanh.name),
             (45, ActivationOutput.tanh.name),
             (45, ActivationOutput.tanh.name),
             (45, ActivationOutput.tanh.name),
             (45, ActivationOutput.tanh.name)
             ],
            [(45, ActivationOutput.tanh.name),   # 4
             (45, ActivationOutput.tanh.name),
             (45, ActivationOutput.tanh.name),
             ],
            [(45, ActivationOutput.tanh.name)    # 5
             ],
            ]
dense_units = [[(45, ActivationOutput.selu.name),   # 0
                (45, ActivationOutput.elu.name),
                (45, ActivationOutput.elu.name),],
               [(45, ActivationOutput.selu.name),], # 1
               [(45, ActivationOutput.selu.name),   # 2
                (45, ActivationOutput.elu.name),
                (45, ActivationOutput.elu.name),],
               [(45, ActivationOutput.tanh.name),   # 3
               ],
               []                                   # 4
              ]
sel_date = datetime(2025, 5, 24, 20, 35)
sel_date_ts = (sel_date.timestamp() / 10000000)
steps = [45, 45,]
metrics = [["accuracy"], []]
dropout = [0, 0]
learning_rate = [0.01, 0.05]
last_lstm_return_sequences = [False, True]
loss = ["binary_crossentropy", "mse"]
output_dense_activation = [ActivationOutput.elu.name, #0
                           ActivationOutput.selu.name, #1
                           ActivationOutput.sigmoid.name, #2
                           ActivationOutput.tanh.name, #3
                           ActivationOutput.softmax.name, #4
                           ActivationOutput.relu.name, #5
                           ActivationOutput.linear.name #6
                          ]
epochs = [10, #0
          20, #1
          25, #2
          50, #3
          100] #4

rand_seed=[sel_date_ts, datetime.now().timestamp(), 0]
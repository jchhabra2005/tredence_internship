# Self-Pruning Neural Network on CIFAR-10

## Introduction

In this project, a self-pruning neural network was implemented using PyTorch on the CIFAR-10 dataset. Instead of using standard fully connected layers, custom `PrunableLinear` layers were designed. These layers learn gate values for every connection in the network, allowing the model to automatically determine which connections are important and which can be suppressed during training.

This project focuses on designing a **differentiable pruning mechanism** that allows the model to adapt its structure dynamically during training, eliminating the need for post-training pruning steps.

---

## Approach

A simple feedforward neural network was used with three fully connected layers. Each layer was replaced with a custom `PrunableLinear` layer.

Every weight in the layer had an associated gate score. The gate scores were passed through a sigmoid activation to obtain gate values between 0 and 1.

The effective weight used during forward propagation was:

[
W_{effective} = W \times \sigma(G)
]

where:

* (W) is the original weight matrix
* (G) is the gate score matrix
* (\sigma(G)) is the sigmoid output of gate scores

This mechanism allows the network to softly control the importance of each connection.

To encourage sparsity, an L1 penalty was added on the gate values during training.

The total loss function used was:

[
Loss = ClassificationLoss + \lambda \times SparsityLoss
]

where:

* ClassificationLoss = CrossEntropyLoss
* SparsityLoss = sum of all gate values
* (\lambda) controls the sparsity strength

---

## Why L1 Regularization Encourages Sparsity

L1 regularization adds a penalty proportional to the absolute value of parameters. In this project, the penalty was applied to gate values.

Since the optimizer minimizes the total loss, it pushes many gate values toward very small numbers to reduce the sparsity penalty.

Unlike L2 regularization, which reduces parameter magnitude smoothly, L1 regularization strongly encourages many values to become close to zero. This naturally creates sparse networks because connections with very small gate values contribute minimally during computation.

As a result, unnecessary connections are automatically suppressed by the model during training.

---

## Results from Lambda Experiments

Different values of the sparsity coefficient (\lambda) were tested to study its effect on sparsity and accuracy.

| Lambda | Accuracy (%) | Sparsity (%) |
| ------ | ------------ | ------------ |
| 0.0    | 54.86        | 0.00         |
| 1e-05  | 57.10        | 1.72         |
| 0.0001 | 55.65        | 36.17        |
| 0.001  | 50.66        | 94.79        |
| 0.01   | 42.84        | 99.83        |

### Observations

* Very small lambda values caused almost no pruning.
* Moderate lambda values introduced noticeable sparsity while maintaining reasonable accuracy.
* Large lambda values forced most gate values close to zero, creating extremely sparse models.
* Excessive sparsity reduced model performance because too many useful connections were removed.

Interestingly, a very small sparsity penalty ((1e^{-5})) slightly improved accuracy, likely due to mild regularization reducing overfitting.

---

## Histogram Observations

The histogram of gate values showed a clear **bimodal tendency**. A large number of gates were pushed close to zero, indicating successful pruning, while another group remained significantly above zero, representing important connections retained by the model.

As the value of (\lambda) increased, the distribution shifted more aggressively toward zero, confirming that the sparsity regularization effectively controlled pruning strength.

This visualization validates that the model learned sparse representations automatically without requiring explicit pruning steps.

---

## Trade-off Between Sparsity and Accuracy

One of the key observations from this project was the trade-off between sparsity and accuracy.

* Higher sparsity reduces the number of active connections.
* Fewer active connections reduce model complexity and memory usage.
* However, excessive pruning removes important information pathways inside the network.

As a result:

* Small or moderate sparsity can maintain good accuracy.
* Extremely high sparsity eventually hurts performance.

This trade-off is crucial in practical machine learning systems, especially when deploying models on resource-constrained environments such as mobile devices or embedded systems.

---

## Conclusion

This project demonstrated how self-pruning neural networks can be implemented using trainable gates and L1 regularization.

Unlike traditional pruning methods applied after training, this approach integrates pruning directly into the training process. The use of differentiable gates allows the network to learn which connections to remove in an end-to-end manner.

The experiments showed that:

* L1 regularization effectively encourages sparsity.
* Gate-based pruning can automatically suppress unnecessary connections.
* Sparsity and accuracy must be carefully balanced.

Overall, this project provided a practical understanding of neural network pruning, sparse learning, and regularization techniques in deep learning.

# Afterburner - Models Experimental Acceleration Library 

Hi!

This library is very much experimental, the goal is to create an easy way to compress neural networks as they train.
Some of really weird stuff can be pushed (and removed, lol).

## DryingLinearLayer

This approach uses the fact that higher-dimension neural network being trained easier, but then it's required to prune it.
Drying approach is a way of applying using plasticity during training.
1. At the beginning the model converge faster on larger number of weights
2. At some point, we start applying weights decay (similarly how it's implemented in Adam optimizer) to a set of neurons, which we mark "drying" neurons.
3. The decay will gradually push weights to zero, and by that time, due to plasticity, all the energy is moved to "wet" neurons.
4. Once the model is trained, "dried" neurons can be pruned without performance impact (as their values are nearly zero).

### Usage

1. Replace your linear layers with DryingLinearLayer
2. At the beginning, the model will train as usual
3. Later, weight drying will kick in (similarly to weight decay with Adam optimizer)
4. Drying neurons means the values will be gradually suppressed to zero
5. During training process, the neural network will gradually adjust to the smaller linear layer
6. Once the model is trained, only a small fraction of neurons will have weights
7. The dried model can be pruned (the "dried" weights can be removed without impact on the accuracy)

### Drying Rates
![image](https://github.com/user-attachments/assets/30f8a7f6-7961-4494-8769-3673aa877090)

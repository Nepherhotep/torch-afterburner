# Afterburner - Models Experimental Acceleration Library 

1. Replace your linear layers with DryingLinearLayer
2. At the beginning, the model will train as usual
3. Later, weight drying will kick in (similarly to weight decay with Adam optimizer)
4. Drying neurons means the values will be gradually suppressed to zero
5. During training process, the neural network will gradually adjust to the smaller linear layer
6. Once the model is trained, only a small fraction of neurons will have weights
7. The dried model can be pruned (the "dried" weights can be removed without impact on the accuracy)

Drying Rate
![image](https://github.com/user-attachments/assets/30f8a7f6-7961-4494-8769-3673aa877090)

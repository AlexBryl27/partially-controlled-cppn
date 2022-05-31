# Partially controlled CPPN with audio signal

Classic CPPN is a NN which contains of several fully connected layers and activations. 
The NN weights are initialized randomly, and when we make the model predict on a random grid, we get an abstract image. Profit.

For animated transformation we can add a hidden dimension to the meshgrid and transform it with a sinusoidal signal. This signal may also be an audio signal.

Based on this idea, I improved this approach a bit.

1. First of all, I didn't wanted to use random weights. For more control, I got the weights by approximating a specific image. This gave me "partially controlled CPPN". 

2. By using of FFT on the audio signal (it was my own track) I got eight amplitudes. I split the hidden meshgrid dimension and added all those amplitudes separately. This gave me eight pulsed areas in the image. The center is static.

Slow random transformations are obtained by a classical cosine signal.

## Result
![Example](test.gif)


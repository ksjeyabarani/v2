##### Name all the layers in the network, describe what they do.
```layer_defs = [];
layer_defs.push({type:'input', out_sx:24, out_sy:24, out_depth:1});
layer_defs.push({type:'conv', sx:5, filters:8, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'pool', sx:3, stride:3});
layer_defs.push({type:'softmax', num_classes:10});

net = new convnetjs.Net();
net.makeLayers(layer_defs);

trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:20, l2_decay:0.001});
```
First layer
- is an input layer with input size declarations.

- Parameters : 
   Size : out_sx:24, out_sy:24, out_depth:1 (24x24 RGB image)

Second layer 
- is an Convolution layer. In this layer, the neurons are connected only locally to a few neurons in the layer before it.

- Parameters : 
   filter size (sx:5)
   number of filters (filters:8)
   stride at which they are applied in the input volume( stride:1)
   pad the input by some amount of pixels with zeros (pad:2)
   activation:'relu'

Third layer 
- is an Pooling layer. The parameters passed to this layer are  same as to the convolutional layer except for 'activation'.

- Parameters : 
   sx:2, stride:2

Fourth layer 
- is a Convolution layer. 

- Parameters: 
   sx:5, filters:16, stride:1, pad:2, activation:'relu'


Fifth layer 
- is an Pooling layer.

- Parameters : 
   sx:3, stride:3

Sixth layer 
- is the last layer 'softmax' loss layer to predict a set of 10 discrete classes for the data.
- the outputs are probabilities that sum to 1. 

- Parameters:
   num_classes:10


##### Experiment with the number and size of filters in each layer. Does it improve the accuracy?
Increasing or decresing the number of filters does not help with increasing the accuracy.

##### Remove the pooling layers. Does it impact the accuracy?
Without pooling layers, the accuracy dropped significantly.

##### Add one more conv layer. Does it help with accuracy?
Adding one more conv layer dropped the accuracy dropped significantly.

##### Increase the batch size. What impact does it have?
No significant change in accuracy with increasing the batch size to twice the size previously.

##### What is the best accuracy you can achieve? Are you over 99%? 99.5%?
With base configuration, I see 99% accuracy a few times before it fluctuates back to below 99%.
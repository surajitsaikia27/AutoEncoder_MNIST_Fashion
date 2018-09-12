# AutoEncoder_MNIST_Fashion
A basic example of an Autoenconder to visualize the concept.
In the past decade Image retrieval is the advancing Field of research in the computer vision domain. Apart from the CNNs, the Autoencoders 
plays a significant role for such retrieval systems. Autoencoder is a Neural Network and also an unsupervised Learning Algorithm. 
It applies Backpropogation algorithm by settiing the same input as  the target value.

# Installation and running
Install Tensorflow for GPU machine
```
pip install tensorflow-gpu
```
If you don't have a GPU
```
pip install tensorflow
```
Download the MNIST Fashion Dataset
https://www.kaggle.com/zalando-research/fashionmnist

Put the proper path of the dataset, and type
```
python autoencoder.py
```
Now, training will start, and at the end you can observe the output generated by the decoder, which will be similar to the input image provided.

![Decoder_ouput](https://github.com/surajitsaikia27/AutoEncoder_MNIST_Fashion/blob/master/decoded.png?raw=true)

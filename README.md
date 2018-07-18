# Modeling action-conditional dynamics of objects from video

In this project, we consider the problem of extracting states for modeling the dynamics of moving objects in
videos. First, with a simple example, we show that a previously proposed architecture, has an undesirable
property: it does not preserve the spatial information of the features in a frame. To mitigate this, we propose
a new architecture. Like previous work, this network takes the form of an autoencoder. The novel property
of this network is that, unlike the previous networks, here the input image is reconstructed at the output
of the network using the weighted sum of radial basis kernels. In contrast to the conventional radial basis
function (RBF) networks the weights are not trainable parameters. Instead, they are computed for each
frame as the output of a network consisted of convolutional and fully connected layers. In this network, since
the center of the RBFs are fixed, the spatial information is preserved.


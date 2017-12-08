# CapsNet

This is the implementation of Capsule Net (https://arxiv.org/pdf/1710.09829.pdf) for MNIST on TensorFlow. 
The architecture, notations and all the details are kept the same as in the above mentioned paper. 

Dynamic routing keeps the same notations, reconstruction loss is added, Adam 
optimizer with exponentially decaying learning rate is used.

This network without much hyperparameter tuning easily achieved ~99.5% on test set in about 70 epochs.

To run the code, type in the terminal
  python run.py

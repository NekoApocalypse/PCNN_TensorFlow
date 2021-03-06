# PCNN-Tensorflow
TensorFlow implementation of PCNN network for relation extraction.

# Alternative Implementation
Use train_PCNN_mask.py & test_PCNN_mask.py for 'masked pooling' implementation.

In the original version we slice sentences into three parts and pad each part to the length of original sentences. This 'trick' affacts the convolution outputs at the ends of each slice.

In the 'masked pooling' version, we do not slice the input sentence. Instead, we use a zero-one masks to split outputs of convolution layer.

Theoratically speaking the later version should be the correct implementation of PCNN, but we keep the original for comparison.

# Data
Dataset is available as 'origin_data.tar.gz'. Extract this file and run 'initial.py' to get training data.

# Requirement
* Tensorflow 1.4
* Python 3.5.2

# Reference
Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf

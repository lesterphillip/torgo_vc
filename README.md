# TORGO Dataset Preprocessor

The official TORGO dataset gave out free samples for their dysarthric speech corpus. 

The dataset samples can be downloaded here: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html

One use of this dataset is Voice Conversion (VC). Using different Neural Network Architectures, we can convert a voice from either dysarthric-to-healthy or healthy-to-dysarthric.
To do parallel voice conversion though, we need a dataset that labels the same transcript from a dysarthric to healthy. 
However, the free samples are not labelled in a way that we can directly do VC.

This script preprocesses and filters the TORGO samples so that parallel voice conversion is possible.
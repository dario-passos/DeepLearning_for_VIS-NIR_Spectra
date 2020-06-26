# DeepLearning for VIS-NIR Spectral Analysis

This repository contains some of my research test about the application of 
Deep Learning architectures (NN models) applied to the analysis of VIS-NIR spectral data. Here are some of my personal 
implementation of interesting NN architectures that can be found in the literature about DeepLearning for spectral analysis.
In some cases I'll try to reproduce the results presented by the original paper authors as a way to validate my
implementation before applying the model to my onw datasets.
Please feel free to take a look and see if you find anything usefull for your own research. Be aware of possible 
bugs in the codes!! 

There are/will be also some links to interesting online resources...


This is part of my research at [CEOT@UAlg](https://www.ceot.ualg.pt/research-groups/sensing-and-biology]).
For some exchange of ideas, drop me an email (dmpassos @ uagl.pt)

<hr>

## MODELS
### Bjerrum et al 2017 CNN

In this notebook I try to reproduce the spectral analysis pipeline that was proposed in Bjerrum et al 2017 ( [paper here](https://arxiv.org/abs/1710.01927) ). 

This is a regression problem. Basically we use the spectra information (X) to train a CNN to predict the ammount of 
some chemical compound (Y). The error metrics obtained with an "optimal" PLS model is used as a baseline for comparisons with the 
CNN model. The pipeline includes spectra pre-processing, outlier removal, implementation and optimization of a PLS model 
(that serves as error metric baseline), implementation of a CNN model, bayesian optimization of the CNN hyperparameters, 
TPE optimization of the CNN hyperparameters and a brief study of the interpretability of the CNN activations in terms
of spectral features

Check the .ipynb notebook for details [Bjerrum2017_CNN/BayesOpt_CNN1.2.ipynb](/notebooks/Bjerrum2017_CNN/BayesOpt_CNN1.2.ipynb)
The notebook might seem long but this is due to the fact that on github there is no clipping of the output of the computation
cells.


### more in the near future...

# DeepLearning for VIS-NIR Spectral Analysis

This repository contains some of my research test about the application of 
Deep Learning architectures (NN models) and other machine learning models applied to the analysis of VIS-NIR spectral data. 
Here are some of my personal implementation of interesting NN architectures that can be found in the literature about 
DeepLearning for spectral analysis. In some cases I'll try to reproduce the results presented by the original paper authors as 
a way to validate my implementation before applying the model to my own datasets. 

During my research I tend to do a lot of exploratory analysis, during which I learn and implement many new things 
for the first time. However, often these exploratory studies/tests get "lost is translation" and I tend to forget 
about some of the stuff I did (python functions, tricks, algorithms...). This repository is also an attempt to discipline 
myself and be a bit more organized research-wise. I include many comments in the notebooks so that I can remember what 
I did in the past. I also find this useful for collaborations since its easier for colleagues and other researchers to 
understand what and why some stuff was done.

Please feel free to take a look around and see if you find anything useful for your own research. Be aware of possible 
bugs in the codes!! 

At some point I'll build a list of interesting online resources on these topics...


This is part of my research at [CEOT@UAlg](https://www.ceot.ualg.pt/research-groups/sensing-and-biology]).
For some exchange of ideas, drop me an email (dmpassos @ ualg.pt)

<hr>

## MODELS
### Bjerrum et al 2017 CNN

In this notebook I try to reproduce the spectral analysis pipeline that was proposed in Bjerrum et al 2017 ( [paper here](https://arxiv.org/abs/1710.01927) ). 

This is a regression problem. Basically we use the spectra information (X) to train a CNN to predict the amount of 
some chemical compound (Y). The error metrics obtained with an "optimal" PLS model is used as a baseline for comparisons with the 
CNN model. The pipeline includes spectra pre-processing, outlier removal, implementation and optimization of a PLS model 
(that serves as error metric baseline), implementation of a CNN model, Bayesian optimization of the CNN hyper-parameters, 
TPE optimization of the CNN hyper-parameters and a brief study of the interpretability of the CNN activations in terms
of spectral features.

Check the .ipynb notebook for details [Bjerrum2017_CNN/BayesOpt_CNN1.2.ipynb](/notebooks/Bjerrum2017_CNN/BayesOpt_CNN1.2.ipynb).
The notebook might seem long but this is due to the fact that on GitHub there is no clipping of the output of the computation
cells (the training steps outputs are visible).


### more in the near future...

![Logo](https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra/blob/master/images/github_card.png)

# DeepLearning for VIS-NIR Spectral Analysis
This repository is part of my research at [CEOT@UAlg](https://www.ceot.ualg.pt/research-groups/sensing-and-biology]) and 
contains some of my research tests about the application of 
Deep Learning architectures (NN models) and other machine learning models applied to the analysis of VIS-NIR spectral data. 
Here are some of my personal implementation of interesting NN architectures that can be found in the literature about 
DeepLearning for spectral analysis. These implementations are done mainly in <code>python</code> and <code>tensorflow.keras</code> in Jupyter notebooks. In some cases I'll try to reproduce the results presented by the original paper authors as 
a way to validate my implementation before applying the model to my own datasets. The folder "images" contains schematics of the NN models implemented in the notebooks. 

During my research I tend to do a lot of exploratory analysis, during which I learn and implement many new things 
for the first time. However, often these exploratory studies/tests get "lost is translation" and I tend to forget 
about some of the stuff I did (python functions, tricks, algorithms...). This repository is also an attempt to discipline 
myself and be a bit more organized research-wise. I try to include many comments in the notebooks so that I can remember what 
I did in the past. This is also useful for collaborations since its easier for colleagues and other researchers to 
understand what and why some stuff was done. That is one of the reasons I thought of sharing these notebooks here in GitHub. 

Please feel free to take a look around and see if you find anything useful for your own research. Be aware of possible 
bugs in the codes!! 

This repository will be continuoulsy updated. At some point I'll also introduce a list of interesting online resources on these topics...


My [ResearchGate profile](https://www.researchgate.net/profile/Dario_Passos).<br>
For some exchange of ideas, drop me an email (dmpassos @ ualg.pt)<br>

<hr>

## MODELS
### Bjerrum et al 2017 CNN

In this notebook I try to reproduce the spectral analysis pipeline that was proposed by Bjerrum et al 2017 in "*Data Augmentation of Spectral Data for Convolutional Neural Network (CNN) Based Deep Chemometrics*" ( [paper here](https://arxiv.org/abs/1710.01927) ). 

This is a regression problem. Basically we use the spectra information (X) to train a CNN to predict the amount of 
some chemical compound (Y). The error metrics obtained with an "optimal" PLS model is used as a baseline for comparisons with the 
CNN model. The pipeline includes spectra pre-processing, outlier removal, implementation and optimization of a PLS model 
(that serves as error metric baseline), implementation of a CNN model, Bayesian optimization of the CNN hyper-parameters, 
TPE optimization of the CNN hyper-parameters and a brief study of the interpretability of the CNN activations in terms
of spectral features.

Check the .ipynb notebook for details [Bjerrum2017_CNN/BayesOpt_CNN1.2.ipynb](/notebooks/Bjerrum2017_CNN/BayesOpt_CNN1.2.ipynb).
The notebook might seem long but this is due to the fact that on GitHub there is no clipping of the output of the computation
cells (the training steps outputs are visible).


<br>

### Cui, Fearn 2018 CNN

This notebook attempts to replicate section section 5.3 of "*Modern practical convolutional neural networks for multivariate regression: Applications to NIR calibration*" ( [paper here](https://www.sciencedirect.com/science/article/pii/S0169743918301382?via%3Dihub) ) by Cui, C. and Fearn, T. 2018. 
I implemented the CNN architecture that the authors suggest for spectral analysis and applied it to a "small dataset". The data used here (named data set 3 by the authors) was downloaded from the [original source](http://www.models.kvl.dk/wheat_kernels). For details and references about the data check the source website or the "Data description.txt" file included.

This is a regression problem. Basically we use the spectra information (our X) to predict the ammount of some chemical compound (our Y). A PLS model is used as baseline for error metrics comparison. 
From a Physics/Chemistry point of view, many researchers need to find which spectral features (absorption bands) are being more used to predict whatever they need to predict. This is helpful because absorption bands allow to identify chemical compounds and in last instance to help understand what physical or even biological processes are in play. So, there is a need to explore the interpretability of NN models. Cui and Fearn 2018, suggest looking at the "regression coefficients of the NN" (see section 2.7 of the paper for details). This is also implemented in this notebook.

Check the .ipynb notebook for details [CuiFearn2018_CNN/Cui_CNN.ipynb](/notebooks/CuiFearn2018_CNN/Cui_CNN.ipynb).


### more in the near future...

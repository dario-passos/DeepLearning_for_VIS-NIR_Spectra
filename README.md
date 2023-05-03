![Logo](https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra/blob/master/images/github_card.png)

# <b>Deep Learning for VIS-NIR Spectral Analysis and Chemometrics</b>
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

## Deep Learning models in the literature
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

<br>

## TUTORIALS
### SVM hyperparameters optimization, Kaneko and Funatsu 2015

In this notebook I'll show one way of performing hyperparameter optimization for Support Vector Machines (with rbf kernel) models when applied to regression problems. From experience I can say that the method works relatively well (depending on the data set of course) when compared to classical grid search and random grid search methods.

The method is based on the theoretical foundations of
Cherkassky, V.; Ma, Y. Practical selection of SVM parameters and noise estimation for SVM regression. Neural Networks 2004, 17, 113 – 126

and further expanded by

Kaneko, H.; Funatsu, K. Fast optimization of hyperparameters for support vector regressionmodels with highly predictive ability. Chemometr. Intell. Lab. 2015, 142, 64 – 69. Kaneko and Funatsu (2015)

Input: x - vector feature data y - vector label data Output: Optimum parameters C, gamma and epsilon. NOTE: Be sure to import all relevant libs before running. Also take into consideration that this methods optimizes the hyperparameters for scaled x in the interval [0,1]. This is done using MinMaxScaler(). The input x values must already be in that interval.

Support Vector Machines for regression has been used in recent years with great success in data analysis and is considered a very versatile technique in machine learning. The efficiency of the SVM model is highly dependent of its hyper-parameters optimization, which means that in the case of using a gaussian kernel (RBF-kernel), hyper-parameters  𝐶 ,  𝛾  and  𝜖  have to be optimized (for each data set individually).  𝐶  is the regularization parameter and it controls a tradeoff between the allowed error deviations and the complexity of the decision function,  𝜖  is related to the error penalty of the loss function and  𝛾  represents the width of the RBF kernel. In what follows I implement the the hyper-parameter optimization proposed by Kaneko and Funatsu 2015. This method uses a combination of theoretical hyper-parameter determination (from Cherkassky and Ma 2004) coupled with a ’restrained’ grid-search approach.

In Kaneko and Funatsu 2015, the authors show that a good estimate of parameter  𝐶  can be obtained analytically from the statistical distribution of the calibration data and, parameter  𝜖  can be obtained directly from a noise estimation in the data based on the mean squared error of a k Nearest Neighbors regressor algorithm. The theoretical value of  𝛾  is determined by maximizing the variance of kernel similarity diversity as detailed in Cherkassky and Ma 2004. However, these theoretical parameters, are not optimized for prediction because of the statistical fluctuations that can exist in different sections of the calibration/training data. Fine-tuned values can be found in by probing an interval around these theoretical values. This part of the optimization process takes two hyper-parameters (e.g.  𝐶  and  𝛾 ) as constant (equal to their theoretical values) and optimizes the 3rd one (e.g.  𝜖 ) by Random Grid Search (RGS) and minimization of RMSE using 5k cross-validation. This is done 3 times optimizing all hyper-parameters. It seems that for this method to work reliably the input data should lie in the [0,1] range.

Check the .ipynb notebook for details [KanekoFunatsu2015_SVROpt/SVM_optimization.ipynb](/notebooks/KanekoFunatsu2015_SVROpt/SVM_optimization.ipynb).

### Tutorial on automated optimization of deep spectral modelling for regression and classification (update .v2)
In this tutorial we show how to implement and optimized the hyperparameteres of a deep learning model (CNN) using advanced Baeysian Optimization techniques. The models are implemented in Tensorflow 2 and python and are optimized using the implementations of TPE (Tree-structured Parzen Estimator) and Hyperband algorithms available in [Optuna](https://optuna.org/). The repository files contain notebooks that are complementary to the official publication <b>*Passos, D., Mishra, P., "A tutorial on automatic hyperparameter tuning of deep spectral modelling for regression and classification tasks"*, Chemometrics and Intelligent Laboratory Systems, Volume 223, 15 April 2022, 104520. [Open access Article](https://doi.org/10.1016/j.chemolab.2022.104520).</b> The tutorial files are self-contained but some theoretical concepts that are explained in the published manuscript might be useful to read. You are free to use the contained information and adapt the accompanying code to your own work, but if you do so, we appreciate that you cite the above mentioned paper. Enjoy...

Check [the tutorial files here](https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra/tree/master/notebooks/Tutorial_on_DL_optimization)


> __March 2023 update__  
The tutorial files were updated to work with newer versions of Optuna (3.0.5), Tensorflow (2.9.1 and up) and python (3.10.5). The main changes come from the Optuna library. All modifications are implemented into 2 new notebooks versions (that end with _v2) and are highlighted in red text, describing what has changed.

> __May 2023 update__
BUG UPDATE!!
**Bug detected in the notebook: "2) optimization_tutorial_classification_v2.ipynb". I detected a typo in the function <code>create_model()</code> defined in section 5.1). The range for the number of DENSE layers starts at index=1 (it should start at index=0). The loop for the hyperparameter optimization (HPO) generated NUM_FC_UNITS and DROPOUT_RATE lists that start at index=0. This means that the HP in these lists that have index=0 were never considered by the CNN during the optimization because only indices>0 are used. In practice this means that when we instantiate a new model, i.e. <code>create_model(num_FC_layers, num_FC_units, filter_size, DROPOUT, reg_beta)</code> with <code>(5, [360,350,132,442,334], 3, [0.555,0.165,0.13,0.46,0.085], 1e-05)</code> the first values for lists num_FC_units (i.e. 360) and DROPOUT (i.e. 0.555) are not considered by the model. We can confirm this in section 6.1. If we change the first items on these lists when we define "model454" the prediction results is exactly the same, i.e. the CNN is not using those hyperparameters because the corresponding DENSE_0/DROPOUT_0 layer pair was never created in the first place.**
**In terms of practical effects this means that the reported model complexity is higher than the real model. Our conclusion that the optimized model <code>model454 = create_model(5, [360,350,132,442,334], 3, [0.555,0.165,0.13,0.46,0.085], 1e-05)</code> had 5 optimized DENSE/DROPOUT layers means that in fact it had 4 optimized DENSE/DROPOUT layers with [350,132,442,334] / [0.165,0.13,0.46,0.085] respectively. That being said, the optimization pipeline suggested here remains valid and a useful way of dealing with CNN HPO for spectral analysis. This notebook will remain as is for the mean time for warning purposes and will be completely substituted with a corrected version after some time, after dealing with the correction of the online accompanying paper. Do a local search for #BUG keyword/comments in the present code to know where things will change**


### more in the near future...

# <b>Deep Tutti Frutti: exploring CNN architectures for dry matter prediction in fruit from multi-fruit near-infrared spectra</b>

In this repository I share the python code used to produce the results of our paper [Passos, D., Mishra, P.,"Deep Tutti Frutti: exploring CNN architectures for dry matter prediction in fruit from multi-fruit 
near-infrared spectra", Chemometrics and Intelligent Laboratory Systems 105023, (2023)](https://doi.org/10.1016/j.chemolab.2023.105023).

The results should be reproducible by using the same version of the libraries. If reproduction of the 
results is your goal, please import and install the requirement.txt file in attach in your conda environment
(python 3.10.9):

<code> $ pip install -r requirements.txt </code>

All notebooks can run autonomously. The file structure in this folder is the following:

**Deep-TuttiFrutti_HPO_analisys.ipynb** contains the analysis of the hyperparameter optimization (HPO) done
for the several tested base architectures. The results shown in the paper are derived from this file.
(If the file is not properly displayed by Github, simply download it).

**data**: folder containing the data sets used in this work

**optimization_files**: The notebooks used for the HPO of the individual architectures, the 
resulting optimization databases generated by Optuna and additional folders with the best 
pre-trained models (in the HPO).

**chemometrics_analysis_help.py**: this is a generic help file containing some functions that I 
usually use in chemometric tasks. Some are general, some are highly specialized for certain data sets
I used in the past. 


Be aware of possible typos in the comments of the code (due to the nasty habbit of copy/paste from other sections to
save time) and of the path to access some files that need to be loaded.

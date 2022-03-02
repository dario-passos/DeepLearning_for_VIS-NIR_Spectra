
# Tutorial on automated optimization of deep spectral modelling for regression and classification
This repository contains the companion notebooks for *Passos, D., Mishra, P., "A tutorial on automatic hyperparameter tuning of deep spectral modelling for regression and classification tasks"*, Chemometrics and Intelligent Laboratory Systems, Volume 223, 15 April 2022, 104520. [Open access Article](https://doi.org/10.1016/j.chemolab.2022.104520). You are free to use the contained information and adapt the accompanying code to your own work, but if you do so, we appreciate that you cite the above mentioned paper. Thanks.

1) Notebook on optimization for a <a href="https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra/blob/master/notebooks/Tutorial_on_DL_optimization/1)%20optimization_tutorial_regression.ipynb" target="_top">regression DL model</a>

2) Notebook on optimization for a <a href="https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra/blob/master/notebooks/Tutorial_on_DL_optimization/2)%20optimization_tutorial_classification.ipynb" target="_top">classification DL model</a>
 
 

Suggestion about how to install the necessary software to run the tutorial for automated spectral modelling

1) On Windows PCs, you can install Miniconda3 from https://docs.conda.io/en/latest/miniconda.html
2) Launch the conda prompt
3) In the shell, create a new conda environment using and setting it to python 3.6:
user> conda create --name tutorial_env python=3.6
During this process, anaconda will create an enviroment with the python base files.
4) After the new environment is create, activate it:
user> conda activate tutorial_env
In principle the pip package manager should be installed by default when the new environment is created. If you do a "pip" command and you get an error back, install it using:
user> conda install pip
5) We will use the pip package manager to install all the other necessary packages:<br>
	user> pip install tensorflow==2.5.0<br>
	user> pip install tensorflow_addons==0.13.0<br>
	user> pip install pandas==1.1.0<br>
	user> pip install tqdm==4.46.0<br>
	user> pip install scipy==1.5.4<br>
	user> pip install scikit-learn==0.24.2<br>
	user> pip install seaborn==0.11.0<br>
	user> pip install matplotlib==3.3.4<br>
	user> pip install jupyter<br>
	user> pip install optuna=2.9.1<br>
	user> pip install livelossplot<br>
If you have a GPU available you might opt for installing the tensorflow GPU version by following the instructions at: https://www.tensorflow.org/install/gpu
6) Assuming everything when ok during the install process, we can now launch the jupyter app
	user> jupyter-notebook
this will open an instance of the Jupyter Notebook on your browser.
7) Browse to where you saved the tutorial .ipynb file, open it and start running it cell by cell.


*Update:*
The notebooks were sucessfully tested with a more up to date version of the packages, on a linux Ubuntu-Mate 21.1 (DL models running on CPU)
Everything worked well using python 3.8 and packages: Tensorflow (2.7.0), Tensorflow add-ons (0.15.0), tqdm (4.60.0), Numpy (1.19.5),Pandas (1.3.5), Optuna  2.10.0, Scikit-Learn (1.0.2)

16/02/2022, Dário Passos, dmpassos@ualg.pt

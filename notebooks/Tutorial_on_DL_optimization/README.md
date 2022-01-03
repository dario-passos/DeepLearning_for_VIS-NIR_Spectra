# Tutorial on automated optimization of deep spectral modelling for regression and classification
 This repository contains the companion notebooks for Passos, Mishra 2021, "A tutorial on automatic hyperparameter tuning of deep spectral modelling for regression and classification tasks". This work is in progress and the main manuscript is still under peer review process (October 2021). If by any chance you stumble uppon this repo, you are free to use the contained information but I kindly ask not to spread the word around until the peer review is completed. I will update this page properly when it happens. Thanks.
 
 
Suggestion about how to install the necessary software to run the tutorial for automated spectral modelling

1) Install Miniconda3 from https://docs.conda.io/en/latest/miniconda.html
2) Launch the conda prompt
3) In the shell, create a new conda environment using and setting it to python 3.6:
user> conda create --name tutorial_env python=3.6
During this process, anaconda will create an enviroment with the python base files.
4) After the new environment is create, activate it:
user> conda activate tutorial_env
In principle the pip package manager should be installed by default when the new environment is created. If you do a "pip" command and you get an error back, install it using:
user> conda install pip
5) We will use the pip package manager to install all the other necessary packages:
user> pip install tensorflow==2.5.0
user> pip install tensorflow_addons==0.13.0
user> pip install pandas==1.1.0
user> pip install tqdm==4.46.0
user> pip install scipy==1.5.4
user> pip install scikit-learn==0.24.2
user> pip install seaborn==0.11.0
user> pip install matplotlib==3.3.4
user> pip install jupyter
user> pip install optuna=2.9.1
user> pip install livelossplot
If you have a GPU available you might opt for installing the tensorflow GPU version by following the instructions at: https://www.tensorflow.org/install/gpu
6) Assuming everything when ok during the install process, we can now launch the jupyter app
	user> jupyter-notebook
this will open an instance of the Jupyter Notebook on your browser.
7) Browse to where you saved the tutorial .ipynb file, open it and start running it cell by cell.

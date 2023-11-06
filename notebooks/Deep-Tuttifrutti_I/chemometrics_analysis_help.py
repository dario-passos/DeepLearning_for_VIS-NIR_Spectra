## Help functions to help tune the PLS models, etc... 
## D.PASSOS
import os
import sys
import shutil
from sys import stdout
import glob
import natsort
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import math
from datetime import datetime

from scipy.signal import savgol_filter
from scipy import stats, interpolate
from sklearn.preprocessing import StandardScaler, Binarizer, MinMaxScaler, RobustScaler
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA
# from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score , KFold, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, explained_variance_score

from joblib import Parallel, delayed


## Compute correlation matrices between x and y quantities
def get_corrs(x,y_col):
    """
    Compute correlations between DataFrame x and dataseries (column) y_col
    -----------------------------------------
    x: pandas DataFrame (n x m), e.g. spectra of n samples with m features
    y_col: column name of a pandas DataFrame with n samples and multiplte columns
    -----------------------------------------
    """
    y_label=y_col.name
    corr=x.corrwith(y_col)
    return corr, y_label 





def plot_corrs(x_data, y_data, l, title=None):
    """
    Plot correlations between a spectra matrix, x_data and a reference matrix quantities in 
    y_data as a function of l (the wavelenght).
    -------------------------------------------
    x_data: pandas DataFrame (n x m), e.g. spectra of n samples with m features
    y_data: pandas DataFrame with n samples and multiplte columns, e.g. wet chemistry or responses
    l: wavelenght range of the spectra
    -------------------------------------------
    """
    plt.figure(figsize=(15,3))
    for col in y_data.columns[1:4]:
        corr, y_label = get_corrs(x_data,y_data[col])
        plt.plot(l,corr, label=y_label)
        plt.xlabel(r'$\lambda$ nm')
        plt.ylabel(r'$\rho$')
        plt.xlim(l[0],l[-1])
        plt.legend(bbox_to_anchor=(1.1,1), loc="upper right")
    plt.title('Pearson correlations with wavelength for '+str(title))
    plt.show()





def vip(model):
    """
    Compute the VIP scores of a trained PLS model
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips




def snv(x_data):
    """ 
    Computes the Standard Normal Variate (SNV) from the full range of the spectrum
    -----------------------------------------
    x_data: numpy array (n x m) with n rows (samples) and m columns (features)
    -----------------------------------------
    """
    # Define a new array and populate it with the corrected data  
    data_snv = np.zeros_like(x_data)
    for i in range(x_data.shape[0]):
         # Apply correction
        data_snv[i,:] = (x_data[i,:] - np.mean(x_data[i,:])) / np.std(x_data[i,:])
    return data_snv





def snv2(data, coi, cof):
    """ 
    Computes the Standard Normal Variate based on a specified range of the spectrum
    -----------------------------------------
    x_data: numpy array (n x m) with n rows (samples) and m columns (features)
    coi: column number where the normalization range starts
    cof: column number where the normalization range stops
    NOTE: if needed check wavelenghts vector to find column number/index
    -----------------------------------------
    """
    # Define a new array and populate it with the corrected data  
    data_snv = np.zeros_like(data)
    for i in range(data.shape[0]):
         # Apply correction
        data_snv[i,:] = (data[i,:] - np.mean(data[i,coi:cof])) / np.std(data[i,coi:cof])
    return data_snv





# Multiplicative Scattering Correction using mean spectra as reference
def msc(input_data, reference=None):
    ''' 
    Perform Multiplicative Scattering Correction (MSC).
    If no reference is introduced, the mean spectra is used as reference.
    The function can be modified to return the mean spectra for use on other data (see last line)
    -----------------------------------------
    input_data: numpy array (n x m), with n rows (samples) and m columns (features)
    reference: numpy array (1 x m)
    -----------------------------------------
    '''
    # mean centre correction
    input_data_centered = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        input_data_centered[i,:] = input_data[i,:] - input_data[i,:].mean()
    # Get the reference spectrum. If not given, estimate it from the mean    
    if reference is None:    
        # Calculate mean
        ref = np.mean(input_data_centered, axis=0)
    else:
        ref = reference
    # Define a new array and populate it with the corrected data    
    data_msc = np.zeros_like(input_data_centered)
    for i in range(input_data_centered.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data_centered[i,:], 1, full=True)
        # Apply correction
        data_msc[i,:] = (input_data_centered[i,:] - fit[0][1]) / fit[0][0] 
    return data_msc
#     return data_msc, ref ## return the reference spectrum as well






def error_metrics(y_true0, y_pred0):
    y_true=np.ravel(y_true0)
    y_pred=np.ravel(y_pred0)
    ## R squared R2 (based on the Pearson correlation) 
    R2 = stats.pearsonr(y_true.squeeze(),y_pred.squeeze())[0]**2
    ## Root Mean Squared Error (RMSE)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    ## Prediction Gain (PG)
     # initialize PG vector with the mean value
    PG0 = np.zeros(len(y_pred)) + np.mean(y_true)
     # Now we compute the rms error between this preciction (mean value) and the validation set
    PG0_MSE= np.sqrt(mean_squared_error(y_true, PG0))
    PG= PG0_MSE / RMSE    
    ## Coefficient of Variation (CVAR) 
    CVAR = np.round(100.*RMSE/np.mean(y_true),2)
    SDR = np.std(y_true) / RMSE
    return np.round(R2,3), np.round(RMSE,3), np.round(PG,3), np.round(CVAR,3), np.round(SDR,3)



## Function to help find the best number of components of the PLS based on CV MSE Loss of the train set
def pls_optimization_cv(x_train, y_train, nmax=20, Nfolds=5, plot_opt=False):
    """
    This function finds the optimal number of PLS components (a.k.a Latent Variables, #LV) that best models the data
    based on mean squared error (MSE) loss and NFold Cross Validation (CV) and returns the best LV number
    X_train - The training data X
    Y_train - The training data Y
    nmax - maximum number of PLS latent variables to test (default=20)
    Nfolds - number of splits for on train set cross-validation (default=5)
    plot_components - Plot the model's optimization and optimized model
    """
    ## Run PLS for a variable number of components (LVs), up to nmax, and compute mean of 'Nfolds' CV error metrics
    cv_rmse=[]
    
    print('\nComputing optimal number of LVs for PLS model in the range 1 to {}...\n'.format(nmax))
    component = np.arange(1, nmax+1)
    for i in component:
        pls = PLSRegression(n_components=i, scale=True)
        
        cv_score=cross_val_score(pls, x_train, y_train, cv=KFold(Nfolds, shuffle = True, random_state=42),\
                        n_jobs=-1, scoring='neg_mean_squared_error', error_score=0)
        ## This scikit-learn scorer returns a negative MSE so we multiply it by -1 when saving its sqrt mean value
        cv_rmse.append(np.round(np.sqrt(-np.mean(cv_score)),3))
        ## Trick to update computation status on the same line (like a progress bar / counter)
        counter = 100*(i+1)/(nmax+1)
        stdout.write("\r%d%% completed" % counter)
        stdout.flush()
    stdout.write("\n")
    
    ## Calculate and print the position of mse minimum (indices start on 0, LV starts on 1)
    cv_rmsemin = np.argmin(cv_rmse)
    bestLV = cv_rmsemin+1
    RMSE_min = np.round(np.min(cv_rmse),3)
    print(f'Suggested number of LV based in Mean of {Nfolds} CV RMSE: {bestLV}')
    print(f'Minimum found in {Nfolds} CV RMSE: {RMSE_min}')
    stdout.write("\n")
    
    # Plot position of bestLV in RMSE score
    if plot_opt is True:
        plt.figure(figsize=(9,3))
        ax1=plt.subplot()
        ax1.plot(component, np.array(cv_rmse), '-v', color = 'blue', mfc='blue')
        ax1.plot(component[cv_rmsemin], np.array(cv_rmse)[cv_rmsemin], 'P', ms=10, mfc='red',label='LV with lowest RMSE')
        plt.xlabel('Number of PLS components')
        plt.ylabel('Mean of '+str(Nfolds)+'CV RMSE ')
        ax1.set_xticks(component)
        plt.xlim(0, nmax+1)
        plt.title('# PLS components')
        plt.legend()
        plt.grid(alpha=0.33)
        plt.show()
    return bestLV, RMSE_min
    

    
## Function to help find the best number of components of the PLS based on CV MSE Loss of the train set
## and returns the error metrics of the optimal model on the test set.
def pls_optimization_cv2(x_train, y_train, Nfolds=5, standardize=False, plot_opt=False):
    """
    This function finds the optimal number of PLS components (a.k.a Latent Variables, #LV) that best models the data
    based on mean squared error (MSE) loss and NFolds Cross Validation (CV) and returns the best LV number
    X_train - The training data X
    Y_train - The training data Y
    standardize - Standardize the data (default=False)
    Nfolds - number of splits for on train set cross-validation
    plot_components - Plot the model's optimization and optimized model
    """
    ## Run PLS for a variable number of components (LVs), up to nmax, 
    ## and compute mean of 'Nfolds' CV error metrics
    cv_mse=[]
    nmax=30
    
    if standardize:
        x_train_scaled = StandardScaler().fit_transform(x_train)
    else:
        x_train_scaled = x_train
        
    print('\nComputing optimal number of LVs for PLS model in the range 1 to {}...\n'.format(nmax))
    component = np.arange(1, nmax)
    for i in component:
        pls = PLSRegression(n_components=i, scale=True, max_iter=500)
        
        cv_score=cross_val_score(pls, x_train, y_train, cv=KFold(Nfolds, shuffle = True, random_state=123),\
                        n_jobs=-1, scoring='neg_mean_squared_error', error_score=0)
        ## This scikit-learn scorer returns a negative MSE so we multiply it by -1 when saving its mean value
        cv_mse.append(np.sqrt(-np.mean(cv_score)))
        ## Trick to update computation status on the same line (like a progress bar / counter)
        counter = 100*(i+1)/nmax
        stdout.write("\r%d%% completed" % counter)
        stdout.flush()
    stdout.write("\n")
    
    ## Calculate and print the position of mse minimum (indices start on 0, LV starts on 1)
    cv_msemin = np.argmin(cv_mse)
    ## Alternatively compute the first inflection of monoticity in cme
    mse_difs=[]
    for i in np.arange(1,len(cv_mse)):
        # compute differences between consecutive mse
        dif=cv_mse[i]-cv_mse[i-1]
        mse_difs.append(dif)
    # find where the signs change in mse_difs    
    indSigChange = np.where(np.sign(mse_difs[:-1]) != np.sign(mse_difs[1:]))[0] +1 +1
    # define the "expert" chosen LV where we have the first change in sign
    if len(indSigChange)==0:
        ## if there is no sign change in different then set it = to where mse is minimal
        expertLV=cv_msemin
    else:
        expertLV = indSigChange[0]
    # print(mse_difs)
    # print(expertLV)
    print("Suggested number of components based in Mean of {} CV MSE loss: {}".format(Nfolds, cv_msemin+1))
    print('Minimum found in Mean of {} CV MSE: {}'.format(Nfolds, np.min(cv_mse)))
    stdout.write("\n")
    bestLV=cv_msemin+1
    # Plot position of bestLV in MSE score
    if plot_opt is True:
        plt.figure(figsize=(9,3))
        ax1=plt.subplot()
        ax1.plot(component, np.array(cv_mse), '-v', color = 'blue', mfc='blue')
        ax1.plot(component[expertLV-1], np.array(cv_mse)[expertLV-1], '*', ms=12, mfc='lightgreen', label='LV of 1st MSE inflection')
        ax1.plot(component[cv_msemin], np.array(cv_mse)[cv_msemin], 'P', ms=10, mfc='red',label='LV with lowest MSE')
        plt.xlabel('Number of PLS components')
        plt.ylabel('Mean of '+str(Nfolds)+'CV RMSE loss')
        plt.title('# PLS components')
        plt.legend()
#         plt.axvline(expertLV)
        plt.xlim(left=-1)
        plt.show()
    return bestLV, expertLV    
    

    
## Function to help find the best number of components of the PLS based on CV MSE Loss of the train set subjected to the
## size of the input data
def pls_optimization_cv3(x_train, y_train, Nfolds=5):
    """
    This function finds the optimal number of PLS components (a.k.a Latent Variables, #LV) that best models the data
    based on mean squared error (MSE) loss and NFold Cross Validation (CV) and returns the best LV number.
    NEW!!--> It tests the lenght of the X vector and adapts the max lv of the search to that match that length.
    X_train - The training data X
    Y_train - The training data Y
    nmax - maximum number of PLS latent variables to test (default=20)
    Nfolds - number of splits for on train set cross-validation (default=5)
    plot_components - Plot the model's optimization and optimized model
	########
	Outputs: best #LVs and min RMSECV
    """
    ## Run PLS for a variable number of components (LVs), up to nmax, and compute mean of 'Nfolds' CV error metrics
    cv_mse=[]
    
    ## Adapt the maximum number of LS of the search space based on the lenght of the X vector
    if np.shape(x_train)[1] <= 15:
        nmax = np.shape(x_train)[1]
    else:
        nmax = 15
        
    print('\nComputing optimal number of LVs for PLS model in the range 1 to {}...\n'.format(nmax))
    component = np.arange(1, nmax+1)
    for i in component:
        pls = PLSRegression(n_components=i, scale=True)
        
        cv_score=cross_val_score(pls, x_train, y_train, cv=KFold(Nfolds, shuffle = True, random_state=123),\
                        n_jobs=-1, scoring='neg_mean_squared_error', error_score=0)
        ## This scikit-learn scorer returns a negative MSE so we multiply it by -1 when saving its mean value
        cv_mse.append(-np.mean(cv_score))
    
    ## Calculate and print the position of mse minimum (indices start on 0, LV starts on 1)
    cv_msemin = np.argmin(cv_mse)
    bestLV = cv_msemin+1
	## Get RMSE from min(cv_mse)
    RMSE_min = np.round(np.sqrt(np.min(cv_mse)),3)
    print(f'Suggested number of LV based in Mean of {Nfolds} CV RMSE: {bestLV}')
    print(f'Minimum found in {Nfolds} CV RMSE: {RMSE_min}')
    stdout.write("\n")

    return bestLV, RMSE_min    
 
    
    
    
## Function to help find the best number of components of the PLS based on MSE Loss of the val set
## and returns the error metrics of the optimal model on the test set.
def pls_optimization_val(x_train, x_val, y_train, y_val, plot_opt=False):
    """
    This function finds the optimal number of PLS components (a.k.a Latent Variables, #LV) that best models the data
    based on mean squared error (MSE) loss and 10 Cross Validation (CV) and returns the best LV number
    X_train - The training data X
    Y_train - The training data Y
    X_val - The validation data X 
    Y_val - The validation data Y 
    plot_components - Plot the model's optimization and optimized model
    """
    ## Run PLS for a variable number of components (LVs), up to nmax, and compute mean of 'Nfolds' CV error metrics
    val_mse=[]
    nmax=30
    
    print('Computing optimal number of LVs for PLS model in the range 1 to {}...\n'.format(nmax))
    component = np.arange(1, nmax)
    for i in component:
        pls = PLSRegression(n_components=i, scale=True, max_iter=1000)
        # train the model
        pls.fit(x_train, y_train)
        # compute score on val dataset
        y_val_pred=pls.predict(x_val)
        R2_test, RMSE_test, PG_test, CVAR_test, SDR_test = error_metrics(y_val,y_val_pred)      
        val_mse.append(RMSE_test**2.)
        ## Trick to update computation status on the same line (like a progress bar / counter)
        counter = 100*(i+1)/nmax
        stdout.write("\r%d%% completed" % counter)
        stdout.flush()
    stdout.write("\n")
    
    ## Calculate and print the position of mse minimum (indices start on 0, LV starts on 1)
    val_msemin = np.argmin(val_mse)
    ## Alternatively compute the first inflection of monoticity in cme
    mse_difs=[]
    for i in np.arange(1,len(val_mse)):
        # compute differences between consecutive mse
        dif=val_mse[i]-val_mse[i-1]
        mse_difs.append(dif)
    # find where the signs change in mse_difs    
    indSigChange = np.where(np.sign(mse_difs[:-1]) != np.sign(mse_difs[1:]))[0] +1 +1
    # define the "expert" chosen LV where we have the first change in sign
    if len(indSigChange)==0:
        ## if there is no sign change in different then set it = to where mse is minimal
        expertLV=cv_msemin
    else:
        expertLV = indSigChange[0]
    # print(mse_difs)
    # print(expertLV)
    print("Suggested number of components based in validation set MSE loss: ", val_msemin+1)
    print('Minimum found in valdation set MSE: {}'.format(np.min(val_mse)))
    stdout.write("\n")
    bestLV=val_msemin+1
    # Plot position of bestLV in MSE score
    if plot_opt is True:
        plt.figure(figsize=(9,3))
        ax1=plt.subplot()
        ax1.plot(component, np.array(val_mse), '-v', color = 'blue', mfc='blue')
        ax1.plot(component[expertLV-1], np.array(val_mse)[expertLV-1], '*', ms=12, mfc='lightgreen', label='LV of 1st MSE inflection')
        ax1.plot(component[val_msemin], np.array(val_mse)[val_msemin], 'P', ms=10, mfc='red',label='LV with lowest MSE')
        plt.xlabel('Number of PLS components')
        plt.ylabel('Mean of val set MSE loss')
        plt.title('# PLS components')
        plt.legend()
#         plt.axvline(expertLV)
        plt.xlim(left=-1)
        plt.show()
    return bestLV, expertLV    






    
def pls_prediction_metrics(l, x_train, y_train, x_test, y_test, xname, yname, lv, verbose=True, plot_pred=False, plot_vip=False):
    """
    USE: pls_prediction_metrics(x_train, y_train, x_test, y_test, yname, lv, plot_pred=False)
    Data X and Y should be numpy arrays...
    #################
    l: vector with wavelenghts for plots x scale
    x_train, y_train: train dataset
    x_test, y_test:   test dataset
    xname:  name of x data (for plots)
    yname:  name of y data (for plots)
    lv:     number of latent variables for PLS model
    verbose: True (False) for output text with results
    plot_pred: False (True) for plot predictions
    plot_vip: False (True) for plot of VIP scores
    ##################
    OUTPUT: several error metrics for train and test sets...
    """
    ## Define PLS with suggested optimal number of components and fit train data
    pls1 = PLSRegression(n_components=lv, scale=True)
    
    ## Fit PLS model to train data
    pls1.fit(x_train, y_train)
    
    ## Get predictions for train and test sets
    y_train_pred = pls1.predict(x_train)
    y_test_pred = pls1.predict(x_test)
    
    ## Compute error metrics
    R2_train, RMSE_train, PG_train, CVAR_train, SDR_train = error_metrics(y_train, y_train_pred)
    R2_test, RMSE_test, PG_test, CVAR_test, SDR_test = error_metrics(y_test, y_test_pred)
    
    if verbose == True:
        print('\nError metrics for best PLS model with LV = {}'.format(lv))
        print('METRIC \t TRAIN \t TEST ')
        print('R2     \t {:0.3f}\t {:0.3f}'.format(R2_train,R2_test))
        print('RMSE   \t {:0.3f}\t {:0.3f}'.format(RMSE_train,RMSE_test))
        print('PG   \t {:0.3f}\t {:0.3f}'.format(PG_train,PG_test))
        print('CVAR   \t {:0.3f}\t {:0.3f}'.format(CVAR_train,CVAR_test))
        print('SDR  \t {:0.3f}\t {:0.3f}'.format(SDR_train,SDR_test))
    
    ## Plots: MSE vs. PLS LV and regression for best PLS model 
    # Get plot limits
    Y = np.concatenate([y_train, y_test])

    rangey = np.max(Y) - np.min(Y)
    rangex = np.max(Y) - np.min(Y)
    
    # x=y line and +- 1std upper and lower bowndaries
    xy_x=np.ravel([np.min(Y)-0.1*rangex, np.max(Y)+0.1*rangex])
    xy_y=np.ravel([np.min(Y)-0.1*rangey, np.max(Y)+0.1*rangey])

    xy_y_up=xy_y+np.std(Y)
    xy_y_down=xy_y-np.std(Y)
   
    if plot_pred is True:
        ## linear fit to predicted test data
        plt.figure(figsize=(5,5))
        plt.title(yname+' prediction using '+xname+' data')
       
        ## fit the test data
        z = np.polyfit(np.ravel(y_test), np.ravel(y_test_pred), 1)

        ax = plt.subplot()
        ax.plot(xy_x, xy_y, 'k--', linewidth=2, label='y=x')
        # plt.fill_between(xy_x, xy_y_down, xy_y_up, alpha=0.2)
        ax.scatter(y_train,y_train_pred,c='gray',s=26, marker='o', alpha=0.66)
        ax.scatter(y_test,y_test_pred,s=40, marker='o', facecolors='None', edgecolors='r')
        ax.plot(y_test, z[1]+z[0]*y_test, c='darkgreen', linewidth=3,label='test')
        plt.xlim(xy_x)
        plt.ylim(xy_y)
        plt.ylabel('Predicted '+yname, fontsize=10)
        plt.xlabel('Measured '+yname, fontsize=10)
        plt.legend(loc=4)
        # Print the test error metrics on the plot
        plt.text(np.min(xy_x)+0.05*rangex, np.max(xy_y)-0.1*rangey, 'R$^{2}=$ %5.2f'  % R2_test, fontsize=13)
        plt.text(np.min(xy_x)+0.05*rangex, np.max(xy_y)-0.15*rangey, 'RMSE: %5.2f' % RMSE_test, fontsize=13)
        plt.text(np.min(xy_x)+0.05*rangex, np.max(xy_y)-0.2*rangey, 'PG: %5.2f' % PG_test, fontsize=13)
        plt.text(np.min(xy_x)+0.05*rangex, np.max(xy_y)-0.25*rangey, 'CVar: %5.2f%%' % CVAR_test, fontsize=13)
        plt.show()    

    if plot_vip is True:
        pls_vip=vip(pls1)
        fig, ax = plt.subplots(figsize=(12,3))
        plt.title('PLS VIP scores ')
        plt.ylabel('VIP score', fontsize=14)
        plt.xlabel('Wavelength (nm)', fontsize=14)
        ax.plot(l,pls_vip,'k',label='VIP scores')
        ax.set_ylim(np.min(pls_vip), np.max(pls_vip))
        plt.axhline(1,color='k', linestyle='--',linewidth=0.75)
        plt.legend()
        plt.show()
        
    return R2_train, RMSE_train, PG_train, CVAR_train, SDR_train, R2_test, RMSE_test, PG_test, CVAR_test, SDR_test



def pls_prediction_metrics2(x_train, y_train, x_test, y_test, yname, lv):
    
    ## Define PLS with suggested optimal number of components and fit train data
    pls1 = PLSRegression(n_components=lv, scale=True, max_iter=1000)
    
    ## Fit PLS model to train data
    pls1.fit(x_train, y_train)
    
    ## Get predictions for test set
    y_test_pred = pls1.predict(x_test)
    
    ## Compute error metrics
    R2_test, RMSE_test, PG_test, CVAR_test, SDR_test=error_metrics(y_test,y_test_pred)
    
    ## Print error metrics
    print('\nError metrics for best PLS model with LV = {}'.format(lv))
    print('METRIC \t TEST')
    print('R2     \t {:0.3f}'.format(R2_test))
    print('RMSE   \t {:0.3f}'.format(RMSE_test))
    print('PG     \t {:0.3f}'.format(PG_test))
    print('CVAR   \t {:0.3f}'.format(CVAR_test))
    print('SDR    \t {:0.3f}'.format(SDR_test))
    
    return [R2_test, RMSE_test, PG_test, CVAR_test, SDR_test]



def plot_traintest_dists(l, X_name, X_train, X_test, Y_name, Y_train, Y_test):
    print('X data = {}'.format(X_name))
    print('x_train: {}{} \t\t  x_test:{} {}'.format(np.shape(X_train),type(X_train),np.shape(X_test),type(X_test)))
    print('\nY data={}'.format(Y_name))
    print('y_train: {}{} \t\t  y_test:{} {}'.format(np.shape(Y_train),type(Y_train),np.shape(Y_test),type(Y_test)))
    plt.figure(figsize=(15,3))
    plt.subplot(131)
    plt.title(Y_name+' (train/test distributions)')
    sns.histplot(data = Y_train, kde=True, label='Train', color='black', legend=True)
    sns.histplot(data = Y_test, kde=True, label='Test', color='red', legend=True)
    plt.legend()
    plt.subplot(132)
    plt.title('Y data')
    if isinstance(Y_train,pd.DataFrame):
        plt.plot(np.arange(len(Y_train)),Y_train.values,'k.',label='Train')
        plt.plot(np.arange(len(Y_test))+len(Y_train),Y_test.values,'r.',label='Test')
        plt.ylabel(Y_name)
        plt.xlabel('Sample number')
    else:
        ## plot of numpy arrays
        plt.plot(np.arange(len(Y_train)),Y_train,'k.',label='Train')
        plt.plot(np.arange(len(Y_test))+len(Y_train),Y_test,'r.',label='Test')
        plt.ylabel(Y_name)
        plt.xlabel('Sample number')
    plt.legend()
    ax3 = plt.subplot(133)
    plt.plot(l,X_train.values.T, 'k', alpha=0.66)
    plt.plot(l,X_test.values.T,'r')
    plt.title('X data')
    plt.ylabel(X_name)
    plt.xlabel('Wavelength')
    ax3.ticklabel_format(axis='y', style='sci',scilimits=[0,1], useMathText=True)
    plt.show()
    
    
def pls_analysis(XNAME,XTRAIN0, XTEST,YNAME,YTRAIN0, YTEST, XTEST_SINGLE, YTEST_SINGLE):
    """
    Input name of the dataset, train set and test sets.
    Computes optimal PLS in 5CV on the train set, computes metrics 
    on the train and test sets, plot Latent space (lv, lv2) and
    VIP scores for best model.
    """
    ############ DATASET DISTRIBUTIONS #########################
    ## Plot train test set distributions
    plot_traintest_dists(XNAME,XTRAIN0, XTEST,YNAME,YTRAIN0, YTEST)

    ############ MODELS #########################
    ## Find best lv for PLS using 10CV
    BestLV_cv, ExpLV_cv = pls_optimization_cv(XTRAIN0, YTRAIN0, Nfolds=5, plot_opt=True)

    print('Optimal PLS LVs -> ', BestLV_cv)
    ## compute final metrics on test set
    pls_prediction_metrics(XTRAIN0.values, YTRAIN0.values, XTEST.values, YTEST.values, YNAME, BestLV_cv, plot_pred=True)
    # print('Expert PLS LVs -> ', ExpLV_cv)
    ## compute final metrics on test set
    # pls_prediction_metrics(XTRAIN0.values, YTRAIN0.values, XTEST.values, YTEST.values, YNAME, ExpLV_cv, plot_pred=True)

    # ## Find best lv for PLS using the validation set
    # BestLV_val, ExpLV_val = pls_optimization_val(XTRAIN, XVAL, YTRAIN, YVAL, plot_opt=True)
    
    # print('Optimal PLS LVs -> ', BestLV_val)
    # ## compute final metrics on test set
    # pls_prediction_metrics(XTRAIN0.values, YTRAIN0.values, XTEST.values, YTEST.values, YNAME,BestLV_val, plot_pred=True)
    # print('Expert PLS LVs -> ', ExpLV_val)
    # ## compute final metrics on test set
    # pls_prediction_metrics(XTRAIN0.values, YTRAIN0.values, XTEST.values, YTEST.values, YNAME,ExpLV_val, plot_pred=True)
    
    
    ############ LATENT SPACE PROJECTION (lv0 and lv1) #########################
    pls=PLSRegression(n_components=2, scale=True, max_iter=1000)
    pls.fit(XTRAIN0.values, YTRAIN0.values)
    
    plt.figure(figsize=(8,6))
    plt.title('PLS scores')
    plt.scatter(pls.x_scores_[:,0],pls.x_scores_[:,1], marker='.', c=YTRAIN0.values.flatten(), cmap='RdYlGn',s=100, alpha=0.66)
    plt.ylabel('LV(1)', fontsize=16)
    plt.xlabel('LV(0)', fontsize=16)
    cbar=plt.colorbar()
    cbar.ax.set_ylabel('Firmness', fontsize=16)
    plt.show()
    
    
    ############ VIP SCORE PLOT #########################
    ## Computing VIP scores for best model
    BEST_PLS=PLSRegression(n_components=BestLV_cv)
    BEST_PLS.fit(XTRAIN0.values, YTRAIN0.values)
    
    YPRED=BEST_PLS.predict(XTEST.values)
    ## Compute the prediction metrics when this model is applied to single spectra values.
    ## In the case of single spectra training, this YPRED_SINGLE is equal to YPRED.
    YPRED_SINGLE=BEST_PLS.predict(XTEST_SINGLE.values)
        
    R2, RMSE, PG, CVAR, SDR = error_metrics(YTEST, YPRED)
    R2_SINGLE, RMSE_SINGLE, PG_SINGLE, CVAR_SINGLE, SDR_SINGLE = error_metrics(YTEST_SINGLE, YPRED_SINGLE)
    ## Compute simple correlations
    # corr, y_label = get_corrs(XTRAIN0,YTRAIN0)
    
    pls_vip=vip(BEST_PLS)
    fig, ax = plt.subplots(figsize=(14,4))
    plt.title('PLS VIP scores ')
    plt.ylabel('VIP score', fontsize=14)
    plt.xlabel('Wavelength (nm)', fontsize=14)
    ax.plot(l[75:-75],pls_vip,'k',label='VIP scores')
    # plt.plot(l[75:-75],np.abs(corr), label='abs(corr)')
    # plt.axvspan(l[75:-75][349], l[75:-75][359], alpha=0.3, color='red')
    plt.axhline(1,color='k', linestyle='--',linewidth=0.75)
    # plt.axvline(715) 
    # plt.xlim(720,1105)
    plt.legend()
    plt.show()
    print('Final metrics on test set:\n R2={}, RMSE={}, PG={}, CVAR={}'.format(R2, RMSE, PG, CVAR))
    print('Final metrics on single samples test set:\n R2={}, RMSE={}, PG={}, CVAR={}'.format(R2_SINGLE, RMSE_SINGLE, PG_SINGLE, CVAR_SINGLE))
    return [YNAME, XNAME, BestLV_cv, R2, RMSE, PG, CVAR, R2_SINGLE, RMSE_SINGLE, PG_SINGLE, CVAR_SINGLE]


## Function to compute metrics and make prediction plots
def plot_prediction(X_train, Y_train, X_test, Y_test, Y_train_pred, Y_test_pred, savefig=False, figname=None):
    
    ## Compute train error scores 
    score_p0 = r2_score(Y_train, Y_train_pred)
    mse_p0 = mean_squared_error(Y_train, Y_train_pred)
    rmse_p0 = np.sqrt(mse_p0)
        
    ## Compute test error scores 
    score_p2 = r2_score(Y_test, Y_test_pred)
    mse_p2 = mean_squared_error(Y_test, Y_test_pred)
    rmse_p2 = np.sqrt(mse_p2)

  
    print('ERROR METRICS: \t TRAIN  \t TEST')
    print('------------------------------------------------------')
    print('R2  : \t\t %5.3f \t\t %5.3f'  % (score_p0,  score_p2 ))
    print('RMSE: \t\t %5.3f \t\t %5.3f' % (rmse_p0,   rmse_p2))
    
 
    ## Plot regression for PLS predicted data
#     rangey = max(Y_train) - min(Y_train)
#     rangex = max(Y_train_pred) - min(Y_train_pred)

    fig=plt.figure(figsize=(6,6))
    z = np.polyfit(np.ravel(Y_test), np.ravel(Y_test_pred), 1)
    ax = plt.subplot()
    ax.scatter(Y_test,Y_test_pred,c='k',marker='o',s=20, alpha=0.6)
    ax.plot(Y_test, z[1]+z[0]*Y_test, c='blue', linewidth=2,label='linear fit')
    ax.plot(Y_test, Y_test, 'k--', linewidth=1.5, label='y=x')
    plt.ylabel('Predicted')
    plt.xlabel('Measured')
    plt.title('Prediction from model')
    plt.legend(loc=4)
#     plt.tight_layout()
#     plt.show()
    
#     # Print the scores on the plot
#     plt.text(min(Y_train_pred)+0.02*rangex, max(Y_test)-0.1*rangey, 'R$^{2}=$ %5.3f'  % score_p2)
#     plt.text(min(Y_train_pred)+0.02*rangex, max(Y_test)-0.15*rangey, 'RMSE: %5.3f' % rmse_p2)

    if savefig==True:
        plt.savefig(figname, dpi=150)
        print('Figure saved')
    else:    
        plt.show() 
    return 


## Adapted From DeepChemometrics (Bjerrum et al 2017)
from numpy import newaxis as nA
class EmscScaler(object):
    '''Extended Multiplicative Scatter Correction'''

    def __init__(self,order=1):
        self.order = order
        self._mx = None
        
    def mlr(self,x,y):
        """Multiple linear regression fit of the columns of matrix x 
        (dependent variables) to constituent vector y (independent variables)
        
        order -     order of a smoothing polynomial, which can be included 
                    in the set of independent variables. If order is
                    not specified, no background will be included.
        b -         fit coeffs
        f -         fit result (m x 1 column vector)
        r -         residual   (m x 1 column vector)
        """
        
        if self.order > 0:
            s=np.ones((len(y),1))
            for j in range(self.order):
                s=np.concatenate((s,(np.arange(0,1+(1.0/(len(y)-1)),1.0/(len(y)-1))**j)[:,nA]),1)
            X=np.concatenate((x, s),1)
        else:
            X = x
        
        #calc fit b=fit coefficients
        b = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)),np.transpose(X)),y)
        f = np.dot(X,b)
        r = y - f

        return b,f,r

    
    def inverse_transform(self, X, y=None):
        print("Warning: inverse transform not possible with Emsc")
        return X
    
    def fit(self, X, y=None):
        """fit to X (get average spectrum), y is a passthrough for pipeline compatibility"""
        self._mx = np.mean(X,axis=0)[:,nA]    
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
#     def transform(self, X, y=None, copy=None):
    def transform(self, X, y=None):
        if type(self._mx) == type(None):
            print("EMSC not fit yet. run .fit method on reference spectra")
        else:
            #do fitting
            corr = np.zeros(X.shape)
            for i in range(len(X)):
                b,f,r = self.mlr(self._mx, X[i,:][:,nA])
                corr[i,:] = np.reshape((r/b[0,0]) + self._mx, (corr.shape[1],))
            return corr
    
    
class GlobalStandardScaler(object):
    """Scales to unit standard deviation and mean centering using global mean and std of X, skleran like API"""
    def __init__(self,with_mean=True, with_std=True, normfact=1.0):
        self._with_mean = with_mean
        self._with_std = with_std
        self.std = None
        self.normfact=normfact
        self.mean = None
        self._fitted = False
        
    def fit(self,X, y = None):
        X = np.array(X)
        self.mean = X.mean()
        self.std = X.std()
        self._fitted = True
        
    def transform(self,X, y=None):
        if self._fitted:
            X = np.array(X)
            if self._with_mean:
                X=X-self.mean
            if self._with_std:
                X=X/(self.std*self.normfact)
            return X
        else:
            print("Scaler is not fitted")
            return
            
    def inverse_transform(self,X, y=None):
        if self._fitted:
            X = np.array(X)
            if self._with_std:
                X=X*self.std*self.normfact
            if self._with_mean:
                X=X+self.mean
            return X
        else:
            print("Scaler is not fitted")
            return
            
    def fit_transform(self,X, y=None):
        self.fit(X)
        return self.transform(X)
    
    
## For use with the CEOT orange dataset
def split_data_internal_cv(x_df1, y_df1, x_df2, y_df2, variedade):
    """
    Given two data sets of two different seasons and orchards, split them
    into 4 different datasets for internal cross-validation purposes.
    """
    L1 = 'Paderne'
    L2 = 'Quarteira'
    S1 = 1
    S2 = 2
    
    
    x_A = x_df1[(y_df1['Season'] == S1 ) & (y_df1['Local'] == L2) & (y_df1['Var'] == variedade)].copy()
    y_A = y_df1[(y_df1['Season'] == S1 ) & (y_df1['Local'] == L2) & (y_df1['Var'] == variedade)].copy()
    
    x_B = x_df1[(y_df1['Season'] == S1 ) & (y_df1['Local'] == L1) & (y_df1['Var'] == variedade)].copy()
    y_B = y_df1[(y_df1['Season'] == S1 ) & (y_df1['Local'] == L1) & (y_df1['Var'] == variedade)].copy()
    
    x_C = x_df2[(y_df2['Season'] == S2 ) & (y_df2['Local'] == L2) & (y_df2['Var'] == variedade)].copy()
    y_C = y_df2[(y_df2['Season'] == S2 ) & (y_df2['Local'] == L2) & (y_df2['Var'] == variedade)].copy()
    
    x_D = x_df2[(y_df2['Season'] == S2 ) & (y_df2['Local'] == L1) & (y_df2['Var'] == variedade)].copy()
    y_D = y_df2[(y_df2['Season'] == S2 ) & (y_df2['Local'] == L1) & (y_df2['Var'] == variedade)].copy()
    
    return x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D


## For use with the CEOT orange dataset
def split_global_data(x, y, variedade):
    """
    Given a global data sets of two different seasons and orchards, split them
    into 4 different datasets for internal cross-validation purposes.
    """
    L1 = 'Paderne'
    L2 = 'Quarteira'
    S1 = 1
    S2 = 2
    
    x_A = x[(y['Season'] == S1 ) & (y['Local'] == L2) & (y['Var'] == variedade)].copy()
    y_A = y[(y['Season'] == S1 ) & (y['Local'] == L2) & (y['Var'] == variedade)].copy()
    
    x_B = x[(y['Season'] == S1 ) & (y['Local'] == L1) & (y['Var'] == variedade)].copy()
    y_B = y[(y['Season'] == S1 ) & (y['Local'] == L1) & (y['Var'] == variedade)].copy()
    
    x_C = x[(y['Season'] == S2 ) & (y['Local'] == L2) & (y['Var'] == variedade)].copy()
    y_C = y[(y['Season'] == S2 ) & (y['Local'] == L2) & (y['Var'] == variedade)].copy()
    
    x_D = x[(y['Season'] == S2 ) & (y['Local'] == L1) & (y['Var'] == variedade)].copy()
    y_D = y[(y['Season'] == S2 ) & (y['Local'] == L1) & (y['Var'] == variedade)].copy()
    
    return x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D


## Function to help find the best number of components of the PLS based on MSE Loss of the train in CV set
## and returns the error metrics of the optimal model on the val set.
def pls_optimization_extval(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D, target_y, verbose=True, kfolds=5, plot_opt=False):
    """
    Given 4 different datasets, compute the best PLS model based in on an external validation strategy.
    target_y: the y quantity to predict
    kfolds: number of folds for cross validation
    plot_opt: True or False to plot the results of the optimization in terms of RMSE vs LVs
    ---------- Algorithm --------
    1) Agregate ABC
    2) Compute the best number of PLS Latent Vars based on ABC using kfolds (e.g. 5) cross-validation
    3) Calibrate best PLS on train data (ABC) and...
    3) ...use best PLS to predict D and compute metrics.
    4) Do permutation of the groups and restart the loop (e.g. train on BCD and predict A)
    5) Return the mean RMSE CV and RMSE Val (or EV) for the 4 permutations
    """
    ## Run PLS for a variable number of components (LVs), up to nmax, and compute mean of 'Nfolds' CV error metrics
    nmax=15
    
    #3 Initialization of the lists with the subsets (they permutate at the end of the loop)
    ## Groups names
    splits_names = ['A','B','C','D']
    ## list the 4 datasets
    X_list = np.array([x_A, x_B, x_C, x_D], dtype=object)
    Y_list = np.array([y_A[target_y], y_B[target_y], y_C[target_y], y_D[target_y]], dtype=object)
    
    ## Empty lists to store the error metrics
    best_LVs = []
    val_RMSEs = []
    val_RMSEs_aux = []
    cv_RMSEs = []
    val_R2 = []
    val_SDR = []
    
    ## For each of the combinations compute the best LV, train PLS and compute metrics on the val set
    for i in range(4):
        ## Concatenate the 3 fist datasets for train ...
        x_train = np.concatenate(X_list[:3])
        y_train = np.concatenate(Y_list[:3])
        train_splits = splits_names[:3]
        
        ## ... and use the last dataset for validation 
        x_val = X_list[-1]
        y_val = Y_list[-1]
        val_splits = splits_names[-1]
        
        ## Initialize empty lists to store auxiliary rmse metrics and print what is being used for train and validation      
        val_rmse=[]
        train_rmse=[]
        cv_rmse=[]
    
        # print(str(train_splits)+" -> "+ str(val_splits))
        
        ## Compute PLS for several number of LV using the train set in kfolds CV and store the CV RMSE for each model
        component = np.arange(1, nmax)
        for i in component:
            pls = PLSRegression(n_components=i, scale=True, max_iter=500)
            cv_score = cross_val_score(pls, x_train, y_train, cv=KFold(kfolds, shuffle = False),\
                        n_jobs=-1, scoring='neg_mean_squared_error', error_score=0)
            ## cv_score is a vector with the scores of the individual kfolds so we need the average it to find the mean CV score            
            ## This scorer in sklearn.cross_val_score 'neg_mean_squared_error' returns a negative MSE so we multiply it by -1 when saving its mean value and compute the sqrt() 
            cv_rmse.append(np.sqrt(-np.mean(cv_score)))
            ## Trick to create a small progress counter...
            # counter = 100*(i+1)/nmax
            # stdout.write(" \r%d%% completed" % counter)
            # stdout.flush()
        # stdout.write("\n")
    
        ## Get the mininum value of the computed CV RMSEs for the several number of LVs
        cv_rmsemin = np.min(cv_rmse)
        ## Calculate and print the position of CV mse minimum (indices start on 0, LV starts on 1)
        cv_rmsemin_index = np.argmin(cv_rmse)
        bestLV=cv_rmsemin_index+1
        best_LVs.append(bestLV) ## append best LV of this split to a list
        
        ## Fit the PLS with the best LV found in CV
        pls  = PLSRegression(n_components=bestLV, scale=True, max_iter=500)
        pls.fit(x_train, y_train)
        ## predict val
        y_val_pred = pls.predict(x_val.values)
        ## compute val prediction metrics
        R2_val, RMSE_val, PG_val, CVAR_val, SDR_val = error_metrics(y_val,y_val_pred)

        ## append valus to list
        val_RMSEs.append(RMSE_val) ## append the val RMSE (external validation)
        cv_RMSEs.append(cv_rmsemin) ## append the cv RMSE
        val_R2.append(R2_val)
        val_SDR.append(SDR_val)
        
        
        # Plot position of bestLV in MSE score
        if plot_opt is True:
            plt.figure(figsize=(6,2))
            ax1=plt.subplot()
            ax1.plot(component, np.array(cv_rmse), '-v', color = 'blue', mfc='blue')
            ax1.plot(component[cv_rmsemin_index], np.array(cv_rmse)[cv_rmsemin_index], 'P', ms=10, mfc='red',label='LV with lowest CV MSE')
            plt.xlabel('Number of PLS components')
            plt.ylabel('Mean CV RMSE'+str(train_splits))
            plt.title('# PLS components')
            plt.legend()
            #  plt.axvline(expertLV)
            plt.xlim(left=-1)
            plt.show()
        
        ## shift the lists one element to repeat the validation procedure on another subset
        splits_names = np.roll(splits_names, 1)
        X_list = np.roll(X_list, 1)
        Y_list = np.roll(Y_list, 1)
    
    mean_val_RMSEs = np.round(np.mean(val_RMSEs),3)
    std_val_RMSEs = np.round(np.std(val_RMSEs),3)
    mean_val_R2 = np.round(np.mean(val_R2),3)
    mean_val_SDR = np.round(np.mean(val_SDR),3)
    mean_cv_RMSEs = np.round(np.mean(cv_RMSEs),3)
    std_cv_RMSEs = np.round(np.std(cv_RMSEs),3)
    
    
    if verbose==True:
        print(r' Mean CV RMSE = {} +- {}'.format(mean_cv_RMSEs, std_cv_RMSEs))
        print(r' Mean EV RMSE = {} +- {}, R2 EV = {}, SDR EV = {}'.format(mean_val_RMSEs, std_val_RMSEs, mean_val_R2, mean_val_SDR))
    
    return best_LVs, mean_cv_RMSEs, std_cv_RMSEs, mean_val_RMSEs, std_val_RMSEs, mean_val_R2, mean_val_SDR





## TO BE USED WITH CEOT's ORANGES DATASETS
## Function to help find the best number of components of the PLS based on RMSE Loss using a cross validations strategy
## where each subset is used once as validation set and the rest as training set
def pls_optimization_extval2(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D, target_y, verbose=True,  plot_opt=False):
    """
    Given 4 different datasets, compute the best PLS model based in on an "external validation" strategy.
    target_y: the y quantity to predict
    kfolds: number of folds for cross validation
    plot_opt: True or False to plot the results of the optimization in terms of RMSE vs LVs
    ---------- Algorithm --------
    1) Given input vectors A, B, C and D, agregate ABC
    2) Calibrate best PLS on train data (ABC) and...
    3) ...use best PLS to predict D and compute metrics (RMSECV)
    4) Do permutation of the groups and restart the loop (e.g. train on BCD and predict A)
    5) Compute PLS prediction for each train and tuning sets
    6) Aggregate all preds and true y and compute error metrics
    5) Return the aggregate error metrics, RMSECV and the best LV for each permutation (ABC->D, BCD->A, etc...)
    """
    ## Run PLS for a variable number of components (LVs), up to nmax, and compute mean of 'Nfolds' CV error metrics
    nmax=15
    
    ## Initialization of the lists with the subsets (they permutate at the end of the loop)
    ## Groups names
    splits_names = ['A','B','C','D']
    ## list of colors for the final plots
    cores = ['k', 'r','g', 'b']
    ## list the 4 datasets
    X_list = np.array([x_A, x_B, x_C, x_D], dtype=object)
    Y_list = np.array([y_A[target_y], y_B[target_y], y_C[target_y], y_D[target_y]], dtype=object)
    
    ## Empty lists to store the error metrics
    best_LVs = []
    tuning_RMSEs = []
    ## Store the LENght of the tuning set here:
    tuning_LEN = []
    ## Store the prediction for each train/tuning subset here:
    train_PREDS = []
    tuning_PREDS = []
    ## Store each train/tuning set invidually here:
    train_TRUE_Y = []
    tuning_TRUE_Y = []

    tuning_NAMES = []

    ######### MAIN LOOP: For each of the combinations compute the best LV, train PLS and compute metrics on the tuning set
    for i in range(4):
        ## Concatenate the 3 fist datasets for train ...
        x_train = np.concatenate(X_list[:3])  # e.g. ABC
        y_train = np.concatenate(Y_list[:3])
        train_splits = splits_names[:3]        
        ## ... and use the last dataset for tuning
        x_tuning = X_list[-1]                  # e.g. D
        y_tuning = Y_list[-1]
        tuning_splits = splits_names[-1]
        
        ## Store the tuning RMSEs generated in the #LV scan here:       
        tuning_lvscan_RMSEs = [] 
        
        ## Adapt the maximum number of LS of the search space based on the lenght of the X vector
        if np.shape(x_train)[1] <= 15:
            nmax = np.shape(x_train)[1]
        else:
            nmax = 15
            
        ##### Fit PLS on train data for nmax number of LV and compute RMSE of the tuning set
        component = np.arange(1, nmax+1)
        for i in component:
            ## Define the PLS model
            pls = PLSRegression(n_components=i, scale=True)
            ## Train the PLS model on the training set
            pls.fit(x_train, y_train)
            ## Use the PLS model to predict the tuning set
            y_tuning_pred_i = pls.predict(x_tuning.values)
            ## Compute RMSE of the tuning set
            RMSE_lvscan_tuning = np.sqrt(mean_squared_error(y_tuning, y_tuning_pred_i))      
            ## Append tuning RMSE to a list
            tuning_lvscan_RMSEs.append(RMSE_lvscan_tuning)
            ## Trick to update computation status on the same line (like a progress bar / counter)
            # counter = 100*(i+1)/nmax
            # stdout.write("\r%d%% completed" % counter)
            # stdout.flush()
        # stdout.write("\n")
    
        ## Get the mininum value of the computed RMSEs for the several number of LVs
        tuning_RMSEmin = np.min(tuning_lvscan_RMSEs)
        ## Get the position of the tuning RMSE minimum (indices start on 0, LV starts on 1)
        tuning_RMSEmin_index = np.argmin(tuning_lvscan_RMSEs)
        ## Define the best number of LVs based on the index of the minimum RMSE + 1 because indices start on 0
        bestLV = tuning_RMSEmin_index + 1
        ## Append best LV of this split to a list
        best_LVs.append(bestLV)
        ## Append the tuning RMSE min to a list
        # tuning_RMSEs.append(tuning_RMSEmin)
        ## Store the number of samples in this tuning set split
        # tuning_LEN.append(len(y_tuning))
        # tuning_NAMES.append(tuning_splits)
        
        ## For each subset and best lv we compute a PLS model and store the predictions for latter
        ## Define a PLS with the bestLV and predict the tuning subset
        pls2 = PLSRegression(n_components = bestLV)
        pls2.fit(x_train, y_train)
        y_train_pred = pls2.predict(x_train)
        y_tuning_pred = pls2.predict(x_tuning)
        ## Store predictions in lists
        train_PREDS.append(y_train_pred)
        tuning_PREDS.append(y_tuning_pred)
        ## Store the true values 
        train_TRUE_Y.append(y_train)
        tuning_TRUE_Y.append(y_tuning)
                
        # Plot position of bestLV in RMSE score
        if plot_opt is True:
            plt.figure(figsize=(6,2))
            ax1=plt.subplot()
            ax1.plot(component, np.array(tuning_lvscan_RMSEs), '-v', color = 'blue', mfc='blue')
            ax1.plot(component[tuning_RMSEmin_index], np.array(tuning_lvscan_RMSEs)[tuning_RMSEmin_index], 'P', ms=10, mfc='red',label='LV with lowest CV MSE')
            plt.xlabel('Number of PLS components')
            plt.ylabel('Mean CV RMSE'+str(train_splits))
            plt.title('# PLS components')
            plt.legend()
            #  plt.axvline(expertLV)
            plt.xlim(left=-1)
            plt.show()
        
        ## shift the X and Y lists one element 
        splits_names = np.roll(splits_names, 1)
        X_list = np.roll(X_list, 1)
        Y_list = np.roll(Y_list, 1)
        ## This way, in the next iteration, train = [BCD] and tuning = [A] 
    ####### END OF THE LOOP OVER THE 4 SPLITS #######
    
    ## If needed we can retrieve the mean RMSE from the saved tuning_RMSEs wich gives basically the same as the RMSE_tuning computed below
    # RMSE_MEAN = np.round(np.sqrt(np.sum(tuning_LEN * np.array(tuning_RMSEs)**2) / np.sum(tuning_LEN)),3)

    ## Aggregate all predictions and true_y data and compute error metrics
    y_train_preds = np.concatenate(train_PREDS, dtype=float)
    y_train_true = np.concatenate(train_TRUE_Y, dtype=float)
    y_tuning_preds = np.concatenate(tuning_PREDS, dtype=float)
    y_tuning_true = np.concatenate(tuning_TRUE_Y, dtype=float)
    
    ## Compute error metrics
    R2_train, RMSE_train, PG_train, CVAR_train, SDR_train = error_metrics(y_train_true, y_train_preds)
    R2_tuning, RMSE_tuning, PG_tuning, CVAR_tuning, SDR_tuning = error_metrics(y_tuning_true, y_tuning_preds) 
 
    if verbose==True:
        print(r'#LVs = {}, tuning splits = {}, splits sizes = {}'.format(best_LVs, tuning_NAMES, tuning_LEN ))
        print('R2_train: {},\t RMSE_train: {},\t PG_train: {},\t CVAR_train: {},\t SDR_train: {}'.
              format(R2_train, RMSE_train, PG_train, CVAR_train, SDR_train))
        print('R2_tuning : {},\t RMSE_tuning: {},\t PG_tuning: {},\t CVAR_tuning: {},\t SDR_tuning: {}'.
              format(R2_tuning, RMSE_tuning, PG_tuning, CVAR_tuning, SDR_tuning))
    
    ## Plot the results of all predictions.
    # if plot_opt is True:
    #     labs = ['D','C','B','A']
    #     plt.figure(figsize=(4,4))
    #     for i in np.arange(4):
    #         plt.scatter(y_tuning_TRUE[i], tuning_PREDS[i], c=cores[i], label=labs[i])
    #     plt.xlabel('True Y')
    #     plt.ylabel('Predicted Y')
    #     plt.legend(fontsize=10)    
    #     plt.show() 

    return best_LVs, R2_train, RMSE_train, PG_train, CVAR_train, SDR_train, R2_tuning, RMSE_tuning, PG_tuning, CVAR_tuning, SDR_tuning








def base_pls(X,y,n_components, return_model=False):
 
    pls_simple = PLSRegression(n_components=n_components)
    pls_simple.fit(X, y)
    y_cv = cross_val_predict(pls_simple, X, y, cv=5)
 
    # Calculate scores
    score = r2_score(y, y_cv)
    rmsecv = np.sqrt(mean_squared_error(y, y_cv))
 
    if return_model == False:
        return(y_cv, score, rmsecv)
    else:
        return(y_cv, score, rmsecv, pls_simple)


def pls_optimise_components_parallel(X, y, npc, cv=10):
 
    out = Parallel(n_jobs=-1, verbose=1)\
        (delayed(base_pls)(X, y, n_components=i) for i in range(1,npc+1,1))
 
    rmsecv = np.zeros(npc)
    score = np.zeros(npc)
    for i,j in enumerate(out):
        y_cv, score[i], rmsecv[i] = j[0],j[1],j[2]
 
    opt_comp, score_max, rmsecv_min = np.argmin(rmsecv),\
    score[np.argmin(rmsecv)],rmsecv[np.argmin(rmsecv)]
 
    return (opt_comp+1, score_max, rmsecv_min)





## Function to help find the best number of components of the PLS based on MSE Loss of the train in CV set
## and returns the error metrics of the optimal model on the val set.
def pls_optimization_extval_parallel(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D, target_y, verbose=True, kfolds=5, plot_opt=False):
    """
    Given 4 different datasets, compute the best PLS model based in on an external validation strategy.
    target_y: the y quantity to predict
    kfolds: number of folds for cross validation
    plot_opt: True or False to plot the results of the optimization in terms of RMSE vs LVs
    ---------- Algorithm --------
    1) Agregate ABC
    2) Compute the best number of PLS Latent Vars based on ABC using kfolds (e.g. 5) cross-validation
    3) Calibrate best PLS on train data (ABC) and...
    3) ...use best PLS to predict D and compute metrics.
    4) Do permutation of the groups and restart the loop (e.g. train on BCD and predict A)
    5) Return the mean RMSE CV and RMSE Val (or EV) for the 4 permutations
    """
    ## Run PLS for a variable number of components (LVs), up to nmax, and compute mean of 'Nfolds' CV error metrics
    nmax=15
    
    #3 Initialization of the lists with the subsets (they permutate at the end of the loop)
    ## Groups names
    splits_names = ['A','B','C','D']
    cores = ['k', 'r','g', 'b']
    ## list the 4 datasets
    X_list = np.array([x_A, x_B, x_C, x_D], dtype=object)
    Y_list = np.array([y_A[target_y], y_B[target_y], y_C[target_y], y_D[target_y]], dtype=object)
    
    ## Empty lists to store the error metrics
    best_LVs = []
    val_RMSEs = []
    val_Preds = []
    y_val_True = []
    cv_RMSEs = []
    val_R2 = []
    len_RMSEs = []
    
    ## For each of the combinations compute the best LV, train PLS and compute metrics on the val set
    for i in range(4):
        ## Concatenate the 3 fist datasets for train ...
        x_train = np.concatenate(X_list[:3])
        y_train = np.concatenate(Y_list[:3])
        train_splits = splits_names[:3]
        
        ## ... and use the last dataset for validation 
        x_val = X_list[-1]
        y_val = Y_list[-1]
        val_splits = splits_names[-1]
        
        ## Initialize empty lists to store auxiliary rmse metrics and print what is being used for train and validation      
        val_rmse=[]
        train_rmse=[]
        cv_rmse=[]
        # print(str(train_splits)+" -> "+ str(val_splits))
        
        ## Compute PLS for several number of LV using the train set in kfolds CV and store the CV RMSE for each model
        components = np.arange(1, nmax+1,1)
        parameters = {'n_components':components}
        pls = PLSRegression()
        mod = GridSearchCV(pls, parameters, cv=KFold(kfolds, shuffle = False),\
                        n_jobs=-1, scoring='neg_mean_squared_error', error_score=0,\
                            return_train_score=True)
        mod.fit(x_train, y_train)
        print(mod.best_params_, np.sqrt(-1* mod.cv_results_['mean_train_score'][mod.best_index_]))
    
            ## cv_score is a vector with the scores of the individual kfolds so we need the average it to find the mean CV score            
            ## This scorer in sklearn.cross_val_score 'neg_mean_squared_error' returns a negative MSE so we multiply it by -1 when saving its mean value and compute the sqrt() 
            # cv_rmse.append(np.sqrt(-np.mean(cv_score)))
            ## Trick to create a small progress counter...
            # counter = 100*(i+1)/nmax
            # stdout.write(" \r%d%% completed" % counter)
            # stdout.flush()
        # stdout.write("\n")
    
    #     ## Get the mininum value of the computed CV RMSEs for the several number of LVs
    #     cv_rmsemin = np.min(cv_rmse)
    #     ## Calculate and print the position of CV mse minimum (indices start on 0, LV starts on 1)
    #     cv_rmsemin_index = np.argmin(cv_rmse)
    #     bestLV=cv_rmsemin_index+1
    #     best_LVs.append(bestLV) ## append best LV of this split to a list
        
    #     ## Fit the PLS with the best LV found in CV
    #     pls  = PLSRegression(n_components=bestLV, scale=True, max_iter=1000)
    #     pls.fit(x_train, y_train)
    #     ## predict val
    #     y_val_pred = pls.predict(x_val.values)
    #     ## compute val prediction metrics
    #     R2_val, RMSE_val, PG_val, CVAR_val = error_metrics(y_val,y_val_pred)

    #     ## append valus to list
    #     val_RMSEs.append(RMSE_val) ## append the val RMSE (external validation)
    #     cv_RMSEs.append(cv_rmsemin) ## append the cv RMSE
    #     val_R2.append(R2_val)
    #     val_Preds.append(y_val_pred)
    #     y_val_True.append(y_val.values)
    #     len_RMSEs.append(len(val_RMSEs))
    #     # print(f'y -> {len(y_val)}, y^ -> {len(y_val_pred)}')
        
        
    #     # Plot position of bestLV in MSE score
    #     if plot_opt is True:
    #         plt.figure(figsize=(6,2))
    #         ax1=plt.subplot()
    #         ax1.plot(component, np.array(cv_rmse), '-v', color = 'blue', mfc='blue')
    #         ax1.plot(component[cv_rmsemin_index], np.array(cv_rmse)[cv_rmsemin_index], 'P', ms=10, mfc='red',label='LV with lowest CV MSE')
    #         plt.xlabel('Number of PLS components')
    #         plt.ylabel('Mean CV RMSE'+str(train_splits))
    #         plt.title('# PLS components')
    #         plt.legend()
    #         #  plt.axvline(expertLV)
    #         plt.xlim(left=-1)
    #         plt.show()
        
    #     ## shift the lists one element to repeat the validation procedure on another subset
        splits_names = np.roll(splits_names, 1)
        X_list = np.roll(X_list, 1)
        Y_list = np.roll(Y_list, 1)

    return mod

    # ## Plot the results of all predictions.
    # if plot_opt is True:
    #     labs = ['D','C','B','A']
    #     plt.figure(figsize=(4,4))
    #     for i in np.arange(4):
    #         plt.scatter(y_val_True[i], val_Preds[i], c=cores[i], label=labs[i])
    #     plt.xlabel('True Y')
    #     plt.ylabel('Predicted Y')
    #     plt.legend(fontsize=10)    
    #     plt.show()    
    
    # ## Compute the final RMSE of the 4 individual validation sets aggregated
    # ## Since we defined val_RMSEs as a list of the root of the mean of certain values, then we need to 
    # ## square each rmse, multiply it by its associated count, sum that stuff up, divide by the 
    # ## total count, and take the square root. 
    
    # Total_val_RMSE = np.round(np.sqrt(np.sum(np.array(val_RMSEs)**2 * len_RMSEs) / np.sum(len_RMSEs)),3)
    
    # ## Alternatively we could concatenate the true and predicted y for each permitation and use 
    # ## the mean_squared_error to compute the final RMSE:
    # # y_val_True = np.concatenate(y_val_True)
    # # val_Preds = np.concatenate(val_Preds)
    # # Total_val_RMSE = np.sqrt(mean_squared_error(y_val_True, val_Preds))
    
    # mean_val_RMSEs = np.round(np.mean(val_RMSEs),3)
    # std_val_RMSEs = np.round(np.std(val_RMSEs),3)
    # mean_val_R2 = np.round(np.mean(val_R2),3)
    # mean_cv_RMSEs = np.round(np.mean(cv_RMSEs),3)
    # std_cv_RMSEs = np.round(np.std(cv_RMSEs),3)
    
    # if verbose==True:
    #     print(r' Mean CV RMSE = {} +- {}'.format(mean_cv_RMSEs, std_cv_RMSEs))
    #     print(r' Mean EV RMSE = {} +- {}, R2 EV = {}'.format(mean_val_RMSEs, std_val_RMSEs, mean_val_R2))
    #     print(r' Fianl EV RMSE = {}, from {}'.format(Total_val_RMSE, val_RMSEs))
    
    # return y_val_True, val_Preds, val_RMSEs, len_RMSEs
    #return best_LVs, mean_cv_RMSEs, std_cv_RMSEs, Total_val_RMSE , mean_val_RMSEs, std_val_RMSEs, mean_val_R2







def NorrisDeriv_single(x, s, g, d):
    """
    Norris Derivative implemented according to Pan, Zhang, Shi, NIR news, Vol. 31(1-2),24-27, 2020
    x: single spectrum (1d array)
    s: number of wavelengths in the smoothing-window 
    g: derivative gap
    n0: total number of wavelengths
    d: derivative order (d=0 -> just smoothing, d=1 -> 1st deriv, d=2 -> 2nd deriv)
    """
    # number of wavelenghts in the spectrum
    n0 = len(x)
    ######### Warnings section #########################
    # check if "segment/smoothing" s in odd
    if (s % 2) == 0: 
        print('Error: segment value (s) must be odd (e.g. 1, 3, 5, ...)')
        return
    # create an upper limit for s (can be anything below n0-1)
    elif s > n0/2:
        print('Smoothing upper limit is set to N0/2 -> {}, where N0=len(x)'.format(int(n0/2)))
        return
    # create an upper limit for the derivative gap (can be anything below n0-1)
    if g > n0/4:
        print('Derivative gap upper limit is set to N0/4 -> {}, where N0=len(x)'.format(int(n0/4)))
        return
    # create warning for derivative (allowed 0,1 and 2)
    if (d>2):
        print('Derivative order should be an integer <= 2.')
        return
    if not isinstance(d, int):
        print('d must be integer')
        return
    ######################################################        
    
    # leftmost and rightmost wavlenghts of the smoothing window
    k_i = int((s-1)/2)
    k_f = n0 - int((s-1)/2)
    
    # initialize smoothed x as zeros
    x_sm = np.zeros(n0)
    
    if d in [0,1,2]:
        ## 1) perform smoothing on x 
        for k in np.arange(1,n0):
            if  k_i < k < k_f:
                x_sm[k] = np.sum(x[k-k_i : k+k_i+1])/s #the +1 at the end of the x range has to do with python indexing
            elif k <= k_i:
                 x_sm[k] = np.sum(x[: k+k_i+1]) / (k+k_i) #the +1 at the end of the x range has to do with python indexing
            else:
                 x_sm[k] = np.sum(x[k-k_i : n0+1]) / (n0 - (k-k_i) + 1)
        
    if d in [1,2]:
        ## 2) difference derivation of x_sm, a.k.a 1st deriv
        x_1d = np.zeros(n0)
        for k in np.arange(1,n0):
            if g < k < n0-g:
                x_1d[k] = (x_sm[k+g] - x_sm[k-g]) / (2*g)
            elif k <= g:
                x_1d[k] = (x_sm[k+g] - x_sm[k]) / g
            else:
                x_1d[k] = (x_sm[k] - x_sm[k-g]) / g
            
    if d in [2]:
        ## 2) difference derivation of x_1d, a.k.a 2nd deriv
        x_2d = np.zeros(n0)
        for k in np.arange(1,n0):
            if g < k < n0-g:
                x_2d[k] = (x_1d[k+g] - x_1d[k-g]) / (2*g)
            elif k <= g:
                x_2d[k] = (x_1d[k+g] - x_1d[k]) / g
            else:
                x_2d[k] = (x_1d[k] - x_1d[k-g]) / g
    
    if d == 0:       
        return x_sm
    if d == 1:
        return x_1d
    if d == 2:
        return x_2d
    
    
    
def NorrisDeriv(X, s, g, d):
    """
    Optimized version of the NorrisDeriv_v0. It implements matrix operations instead of a loop for each row.
    Norris Derivative implemented according to Pan, Zhang, Shi, NIR news, Vol. 31(1-2),24-27, 2020
    X: spectra matrix (m spectra (rows) , n0 wavelengths (columns))
    s: number of wavelengths in the smoothing-window 
    g: derivative gap
    n0: total number of wavelengths
    d: derivative order (d=0 -> just smoothing, d=1 -> 1st deriv, d=2 -> 2nd deriv)
    """
    # number of wavelenghts in the spectrum
    n0 = np.shape(X)[1] 
    # number os samples in spectral matrix
    m = np.shape(X)[0]
    
    ######### Warnings section #########################
    # check if "segment/smoothing" s in odd
    if (s % 2) == 0: 
        print('Error: segment value (s) must be odd (e.g. 1, 3, 5, ...)')
        return
    # create an upper limit for s (can be anything below n0-1)
    elif s > n0/2:
        print('Smoothing upper limit is set to N0/2 -> {}, where N0=len(x)'.format(int(n0/2)))
        return
    # create an upper limit for the derivative gap (can be anything below n0-1)
    if g > n0/4:
        print('Derivative gap upper limit is set to N0/4 -> {}, where N0=len(x)'.format(int(n0/4)))
        return
    # create warning for derivative (allowed 0,1 and 2)
    if (d>2):
        print('Derivative order should be an integer <= 2.')
        return
    if not isinstance(d, int):
        print('d must be integer')
        return
    ######################################################        
    
    # leftmost and rightmost wavlenghts of the smoothing window
    k_i = int((s-1)/2)
    k_f = n0 - int((s-1)/2)
    
    if d in [0,1,2]:
        # initialize smoothed values matrix
        X_sm = np.zeros((m, n0))
        ## 1) perform smoothing on x
        for k in np.arange(1,n0):  
            if  k_i < k < k_f:
                X_sm[:,k] = np.sum(X[ : , k-k_i : k+k_i+1], axis=1)/s #the +1 at the end of the x range has to do with python indexing
            elif k <= k_i:
                X_sm[:,k] = np.sum(X[ : , :k+k_i+1], axis=1) / (k+k_i) #the +1 at the end of the x range has to do with python indexing
            else:
                X_sm[:,k] = np.sum(X[: , k-k_i : n0+1], axis=1) / (n0 - (k-k_i) + 1)
        
    if d in [1,2]:
        # initialize the 1d matrix
        X_1d = np.zeros((m, n0))
        ## 2) difference derivation of x_sm, a.k.a 1st deriv
        for k in np.arange(1,n0):
            if g < k < n0-g:
                X_1d[:,k] = (X_sm[: , k+g] - X_sm[: , k-g]) / (2*g)
            elif k <= g:
                X_1d[:,k] = (X_sm[: , k+g] - X_sm[: , k]) / g
            else:
                X_1d[:,k] = (X_sm[: , k] - X_sm[: , k-g]) / g
            
    if d in [2]:
        # initialize the 2d matrix
        X_2d = np.zeros((m, n0))
        ## 2) difference derivation of x_1d, a.k.a 2nd deriv
        for k in np.arange(1,n0):
            if g < k < n0-g:
                X_2d[:,k] = (X_1d[: , k+g] - X_1d[: , k-g]) / (2*g)
            elif k <= g:
                X_2d[:,k] = (X_1d[: , k+g] - X_1d[: , k]) / g
            else:
                X_2d[:,k] = (X_1d[: , k] - X_1d[: , k-g]) / g
        
    if d == 0:       
        return X_sm
    if d == 1:
        return X_1d
    if d == 2:
        return X_2d



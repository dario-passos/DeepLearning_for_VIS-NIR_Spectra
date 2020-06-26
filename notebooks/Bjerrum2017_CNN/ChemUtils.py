#Spectral Utilities
from __future__ import print_function
import numpy as np
import scipy #TODO reimplement as Numpy only
from scipy import newaxis as nA


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

class SavGolFilt(object):
    """Applies a Savitsky-Golay filter of order k and frame width F.
    The order must be odd and the frame width (F) a positive integer of
    a value greater than k
    """
    #TODO use the scipy implementation
    def __init__(self, order=1, width=11):
        self.k = order
        self.frame = width

    def transform(self,myarray,y=None):
        """Applies a Savitsky-Golay filter of order k and frame width F.
        The order must be odd and the frame width (F) a positive integer of
        a value greater than k
        """
        frange = scipy.arange(-(self.frame-1)/2,((self.frame-1)/2)+1)
        f, vande = 0, scipy.zeros((self.frame,self.frame))
        while f < self.frame:    # compute Vandemonde matrix
            vande[f,:] = frange**f
            f = f+1
        vande = scipy.transpose(vande,(1,0))
        vande = vande[:,0:self.k+1]
        Q,R = scipy.linalg.qr(vande,vande.shape[1]) # Do QR decomposition
    
    #    print vande.shape
    #    print Q.shape
    #    print R[0:vande.shape[1]]
        G = scipy.dot(vande,scipy.dot(scipy.linalg.inv(R[0:vande.shape[1]]), 
        scipy.transpose(scipy.linalg.inv(R[0:vande.shape[1]])))) # Find the matrix of differentiators
    
        B = scipy.dot(G,scipy.transpose(vande)) # Projection matrix
    
        myarray = scipy.transpose(myarray)
        extract_array, extract_B = myarray[0:self.frame,:], B[(((self.frame-1)/2)+1):self.frame,:]
        start_array = scipy.dot(extract_B[::-1],extract_array[::-1]) # first bins

        array_size = myarray.shape
        last, mid_array = (self.frame-1)/2, scipy.zeros((array_size[0],array_size[1]),'d')
        extract_B = scipy.reshape(B[((self.frame-1)/2),:],(self.frame,1))
        while last < array_size[0]-((self.frame-1)/2):
            mid_array[last,:] = sum((extract_B*myarray[last-((self.frame-1)/2):last+((self.frame-1)/2)+1,:]),0) #middle bit
            last = last+1
        
        extract_array, extract_B = myarray[array_size[0]-self.frame:array_size[0],:], B[0:(self.frame-1)/2,:]
        end_array = scipy.dot(extract_B[::-1],extract_array[::-1]) # last bins

        mid_array[0:(self.frame-1)/2,:], mid_array[array_size[0]-((self.frame-1)/2):array_size[0],:] = start_array, end_array
        return scipy.transpose(mid_array)

    def fit(self, X,y=None):
        print("Fit not needed for filter")
        pass
        
    def fit_transform(self, X,y=None):
        return self.transform(X)

class EmscScaler(object):
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
            s=scipy.ones((len(y),1))
            for j in range(self.order):
                s=scipy.concatenate((s,(scipy.arange(0,1+(1.0/(len(y)-1)),1.0/(len(y)-1))**j)[:,nA]),1)
            X=scipy.concatenate((x, s),1)
        else:
            X = x
        
        #calc fit b=fit coefficients
        b = scipy.dot(scipy.dot(scipy.linalg.pinv(scipy.dot(scipy.transpose(X),X)),scipy.transpose(X)),y)
        f = scipy.dot(X,b)
        r = y - f

        return b,f,r

    
    def inverse_transform(self, X, y=None):
        print("Warning: inverse transform not possible with Emsc")
        return X
    
    def fit(self, X, y=None):
        """fit to X (get average spectrum), y is a passthrough for pipeline compatibility"""
        self._mx = scipy.mean(X,axis=0)[:,nA]
        
    def transform(self, X, y=None, copy=None):
        if type(self._mx) == type(None):
            print("EMSC not fit yet. run .fit method on reference spectra")
        else:
            #do fitting
            corr = scipy.zeros(X.shape)
            for i in range(len(X)):
                b,f,r = self.mlr(self._mx, X[i,:][:,nA])
                corr[i,:] = scipy.reshape((r/b[0,0]) + self._mx, (corr.shape[1],))
            return corr
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
        
        

def dataaugment(x, betashift = 0.05, slopeshift = 0.05,multishift = 0.05):
    #Shift of baseline
    #calculate arrays
    beta = np.random.random(size=(x.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(x.shape[0],1))*2*slopeshift-slopeshift + 1
    #Calculate relative position
    axis = np.array(range(x.shape[1]))/float(x.shape[1])
    #Calculate offset to be added
    offset = slope*(axis) + beta - axis - slope/2. + 0.5

    #Multiplicative
    multi = np.random.random(size=(x.shape[0],1))*2*multishift-multishift + 1

    x = multi*x + offset

    return x        
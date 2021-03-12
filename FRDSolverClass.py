import numpy as np
import pandas as pd #Is this truly required?
from scipy.ndimage.measurements import center_of_mass #This function is imported in order to measure changes about the center of the PSF.
from scipy import interpolate #Is this required?
import warnings

class FRDsolver(object):
    """FRD solver function for PSF inputs"""
    
    def __init__(self, FRDlist, knownimagelist, imagetosolve):
        """Generates a FRDsolver object"""
        
        self._checkFRDlist(FRDlist)
        self._checkimagelists(knownimagelist,imagetosolve)
        self._checklengths(FRDlist,knownimagelist)
        
        self.FRDlist = FRDlist
        self.knownimagelist = knownimagelist
        self.imagetosolve = imagetosolve
        
        self.residuallist = np.nan

        
    def _checkFRDlist(self,FRDlist):
        """Validates the input FRD list"""
        
        if not isinstance(FRDlist,np.ndarray):
            raise Exception('FRDlist should be a np.ndarray')
            
        if not np.array_equal(FRDlist,np.sort(FRDlist)):
            warnings.warn('FRDlist is not sorted!')  

    def _checkimagelists(self,knownimagelist,imagetosolve):
        """Validates the input images"""
        
        if not isinstance(knownimagelist,np.ndarray):
            raise Exception('knownimagelist should be a np.ndarray')
        
        if not isinstance(imagetosolve,np.ndarray):
            raise Exception('imagetosolve should be a np.ndarray')
            
        for image in knownimagelist:
            if np.shape(image) != np.shape(imagetosolve):
                raise Exception('The input images of known FRD should have the same dimensions as the image to solve.')  
        
    def _checklengths(self, FRDlist, knownimagelist):
        "Verifies that the FRD inputs match the inputted images"
        
        if not len(FRDlist) == len(knownimagelist):
            raise Exception('The length of FRDlist is not equal to the length of knownimagelist. Make sure each image has a corresponding FRD.')
            
        if len(FRDlist) == 0:
            raise Exception('At least one FRD must be input!')
        
        if len(FRDlist) == 1:
            warnings.warn('No meaningful result will occur when only one comparison FRD value is given.')

    def residual_calculate(self,imagetosolve,guessimage):
    
        residualval = 0
        currentimage = imagetosolve 
        modeltocompare = guessimage 
            
        centery, centerx = center_of_mass(currentimage) #Determine the center of the PSF. Ordered y, x as it would appear in imshow() but most important thing is to be consistent with ordering
        centery = int(np.round(centery)) #Rounded to permit easier pixel selection. But as is obvious, "easier" does not necesssarily translate to "better". #Debugging, to be removed
        centerx = int(np.round(centerx))
    
        residualval += np.sum(np.divide(np.square(currentimage[(centery-3):(centery+3),(centerx-3):(centerx+3)]-modeltocompare[(centery-3):(centery+3),(centerx-3):(centerx+3)]),(modeltocompare[(centery-3):(centery+3),(centerx-3):(centerx+3)])))*np.sqrt(2)
        residualval -= np.sum(np.divide(np.square(currentimage[(centery-1):(centery+1),(centerx-1):(centerx+1)]-modeltocompare[(centery-1):(centery+1),(centerx-1):(centerx+1)]),(modeltocompare[(centery-1):(centery+1),(centerx-1):(centerx+1)])))*np.sqrt(2)

        residualval = residualval/40 #Number of pixels in the calculation
        
        return residualval            
            
    def find_FRD_compare_positions(self, FRDlist, knownimagelist, imagetosolve):
        """MinFRD = find_FRD_compare_positions(self, FRDlist, knownimagelist, imagetosolve)
        Solves for minimal FRD and returns it
        
        Parameters
        -----------
    
        imagetosolve: array
            PSF image with unknown FRD.
        
    
        knownimagelist: list
            A list of PSF image arrays of known FRDs
    
        
        FRDlist: list
            A list of FRDs of the corresponding PSF image arrays in imagelist

        """
        
        
        #Begin with validation of the inputs
        self._checkFRDlist(FRDlist)
        self._checkimagelists(knownimagelist,imagetosolve)
        self._checklengths(FRDlist, knownimagelist)
            
        residuallist = []
        minFRD = np.nan
    
        for FRDindex in range(len(FRDlist)):
            residual_current = self.residual_calculate(imagetosolve,knownimagelist[FRDindex])
            residuallist.append(self.residual_calculate(imagetosolve,knownimagelist[FRDindex]))
            if residual_current == np.min(residuallist):
                minFRD = FRDlist[FRDindex] #Still need to return the metric
        
        if np.isnan(minFRD):
            raise Error('No FRD residuals were calculated.')
            
        self.residuallist = residuallist
        self.minFRD = minFRD
        
        return (residuallist,minFRD)
    
    def returnFRDrange(self):
        if np.isnan(self.residuallist):
            raise Error('Must run find_FRD_compare_positions first.')
        else:
            return (np.min(residuallist[residuallist<1]),np.max(residuallist[residuallist<1]))
    
    
    """ def _find_FRD_compare_positions(imagetosolve, imagelist,position,FRDarray):
        MinFRD = find_FRD_compare_positions(imagetosolve, imagelist)
        Runs the shgo minimization algorithm to test for the minimum FRD fitting
    
        Parameters
        -----------
    
        imagetosolve: array
            PSF image with unknown FRD.
        
    
        imagelist: list
            A list of PSF image arrays of known FRDs
        
    
        position: list
            A list of positions from the same fiber to be used to find FRD. 
    
        
        FRDarray: list
            A list of FRDs of the corresponding PSF image arrays in imagelist

        bounds = [(0.015, 0.030)] #could make the FRD bounds an independent input, especially if interpolation is not viable external to generation. Instead could use bounds of input FRDarray
        res = shgo(residual_compare,bounds,args=([position,FRDarray,imagelist,imagetosolve]),n=10,options={'ftol':1e-5, 'maxev':10}) #Minimization algorithm.
        #print(res)
        return res.x   """ 


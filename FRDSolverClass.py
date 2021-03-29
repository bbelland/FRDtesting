import numpy as np
import pandas as pd #Is this truly required?
from scipy.ndimage.measurements import center_of_mass #This function is imported in order to measure changes about the center of the PSF.
from scipy import interpolate #Is this required?
from scipy.ndimage.interpolation import shift
#import matplotlib.pyplot as plt #Debug only
import warnings

class FRDsolver(object):
    """FRD solver function for PSF inputs"""
    
    def __init__(self, FRDarray, knownimagearray, imagetosolve):
        """Generates a FRDsolver object"""

        self.FRDarray = FRDarray
        self.knownimagearray = knownimagearray
        self.imagetosolve = imagetosolve
        
        self._checkFRDarray(FRDarray)
        self._checkimagearrays(knownimagearray,imagetosolve)
        self._checklengths(FRDarray,knownimagearray)
        
        self.residuallist = np.array([np.nan])

        
    def _checkFRDarray(self,FRDarray):
        """Validates the input FRD list"""
        
        if not isinstance(FRDarray,np.ndarray):
            raise Exception('FRDarray should be a np.ndarray')
            
        if not np.array_equal(FRDarray,np.sort(FRDarray)):
            warnings.warn('FRDarray is not sorted!')  

    def _checkimagearrays(self,knownimagearray,imagetosolve):
        """Validates the input images"""
        
        if not isinstance(knownimagearray,np.ndarray):
            raise Exception('knownimagearray should be a np.ndarray')
        
        if not isinstance(imagetosolve,np.ndarray):
            raise Exception('imagetosolve should be a np.ndarray')
            
        for image in knownimagearray:
            if np.shape(image) != np.shape(imagetosolve):
                raise Exception('The input images of known FRD should have the same dimensions as the image to solve.') 

        #self._recenter_imagelist()

    def _recenter_image(self,image,centertoshiftto):
        """Shift an image to a given center, centertoshiftto"""
        return shift(image,np.array(centertoshiftto)-np.array(center_of_mass(image)))

    def _recenter_imagelist(self):
        """Shifts all images to the same center as the first image in the knownimagearray"""  

        center_to_shift = center_of_mass(self.knownimagearray[0])

        temporaryarray = self.knownimagearray

        for element in range(len(temporaryarray)):
             temporaryarray[element] = self._recenter_image(temporaryarray[element],center_to_shift)

        self.knownimagearray = temporaryarray

        
    def _checklengths(self, FRDarray, knownimagearray):
        "Verifies that the FRD inputs match the inputted images"
        
        if not len(FRDarray) == len(knownimagearray):
            raise Exception('The length of FRDarray is not equal to the length of knownimagearray. Make sure each image has a corresponding FRD.')
            
        if len(FRDarray) == 0:
            raise Exception('At least one FRD must be input!')
        
        if len(FRDarray) == 1:
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
            
    def find_FRD_compare_positions(self, FRDarray, knownimagearray, imagetosolve):
        """MinFRD = find_FRD_compare_positions(self, FRDarray, knownimagearray, imagetosolve)
        Solves for minimal FRD and returns it
        
        Parameters
        -----------
    
        imagetosolve: array
            PSF image with unknown FRD.
        
    
        knownimagearray: array
            An array of PSF image arrays of known FRDs
    
        
        FRDarray: array
            An array of FRDs of the corresponding PSF image arrays in knownimagearray

        """
        
        
        #Begin with validation of the inputs
        self._checkFRDarray(FRDarray)
        self._checkimagearrays(knownimagearray,imagetosolve)
        self._checklengths(FRDarray, knownimagearray)
            
        residuallist = []
        minFRD = np.nan
    
        for FRDindex in range(len(FRDarray)):
            residual_current = self.residual_calculate(imagetosolve,knownimagearray[FRDindex])
            residuallist.append(self.residual_calculate(imagetosolve,knownimagearray[FRDindex]))
            if residual_current == np.min(residuallist):
                minFRD = FRDarray[FRDindex] #Still need to return the metric
        
        if np.isnan(minFRD):
            raise Exception('No FRD residuals were calculated.')
            
        self.residuallist = residuallist
        self.minFRD = minFRD
        #print(np.array(residuallist)<2)

        if np.sum(np.array(residuallist)<2) >= 2:
            uncertaintylower = np.min(FRDarray[np.array(residuallist)<2]) 
            uncertaintyupper = np.max(FRDarray[np.array(residuallist)<2])
        
            print("Minimum FRD is {} and within range {} to {}".format(minFRD,uncertaintylower,uncertaintyupper))
            
            if uncertaintylower == np.min(FRDarray):
                warnings.warn('Extracted FRD range extends beyond the minimum of the inputted FRD range! The uncertainty may be larger than the quoted value')

            if uncertaintyupper == np.max(FRDarray):
                warnings.warn('Extracted FRD range extends beyond the maximum of the inputted FRD range! The uncertainty may be larger than the quoted value')

            if uncertaintylower == minFRD:
                warnings.warn('Minimum of the extracted FRD range matches minimum FRD extracted! The uncertainty may extend lower than the quoted value')
                
                #Add extrapolation

            if uncertaintyupper == minFRD:
                warnings.warn('Maximum of the extracted FRD range matches minimum FRD extracted! The uncertainty may extend higher than the quoted value')
                
                #Add extrapolation

            uncertaintyrange = uncertaintyupper - uncertaintylower 

        if np.sum(np.array(residuallist)<2) == 1:
            #print('Only the minimum FRD has less than 2 chi squared value. You may want to increase the number of FRD values tested near minFRD to get an accurate uncertainty')
            uncertaintyrange = 0

        if np.sum(np.array(residuallist)<2) == 0:
            #print('None of the pixels have chi squared value less than 2. Returning the minimum residual's uncertainty. Check the residual of the output to verify it is')
            uncertaintyrange = 0
        
        return (residuallist,minFRD,uncertaintyrange)

    def residual_calculate_smarter(self,imagetosolve,FRDarray,knownimagearray,minFRD,noiselevel):
    
        residualvallist = []
        currentimage = imagetosolve 

        mask = currentimage>noiselevel #Only select pixels with counts significantly above noise level, i.e. 100

        if minFRD is not None: 
            guessFRD = minFRD
        else:
            guessFRD = FRDarray[int(len(FRDarray)/2)] #Arbitrary guess

        modelimage = knownimagearray[FRDarray == guessFRD]

        
        #modeltocompare = guessimage 

        for imageindex in range(len(knownimagearray)):
            #First calculate what is the mask
            modeltocompare = knownimagearray[imageindex]
            currentFRD = FRDarray[imageindex]
            if currentFRD == guessFRD:
                continue
            else:
                FRDresidual = np.abs(modeltocompare*mask - modelimage*mask)
                normalizedweighting = np.nan_to_num(FRDresidual/np.sqrt(currentimage)) #This compares the magnitude of the FRD change to the noise of the currentimage 
                normalizedweighting = normalizedweighting/np.max(normalizedweighting)
                residualval = normalizedweighting*np.divide(np.square(modeltocompare-imagetosolve),imagetosolve)
                residualval = residualval/np.sum(normalizedweighting) #Number of pixels in the calculation
                residualvallist.append(np.sum(residualval))
                #plt.imshow(residualval[0])
                #plt.colorbar()
                #plt.show()
        
        testFRDarray = FRDarray[FRDarray != guessFRD]
        #plt.plot(testFRDarray,residualvallist)
        #plt.show()

        return True 





    
    def returnFRDrange(self):
        if any(np.isnan(self.residuallist)):
            raise Exception('Must run find_FRD_compare_positions first.')
        else:
            if np.sum([residuallist<1]) == 0:
                warnings.warn('No FRD value gives a chi squared under 1.')
            #Test to see if terms under 1 stay under 1?
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



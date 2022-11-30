# ------------------------------------------------------------------- # 
# Developed by: Rota 2030 - Centro Unversitário FACENS,               #
#                           Instituto Técnologico de Aeronautica,     #
#                           Robert Bosch,                             #
#                           Stellantis NV and                         #
#                           Technische Hochschule Ingolstadt.         # 
#                                                                     #
# @author: BRAIN - Brazilian Artificial Intelligence Nucleus          #
# Centro Universitário FACENS, Sorocaba, Brazil, 2021.                #
#                                                                     #
# Developers: Herick Y. S. Ribeiro, Luiz H. Aguiar.                   # 
#                                                                     #
# e-mails: herick.ribeiro@facens.br, luizh5391@gmail.com.             #
#                                                                     #
# ------------------------------------------------------------------- #

# ------------------------------------------------------------------- #
# Description: The cfar_lib implements CA, GOCA, SOCA and OS CFAR al  #
# gorithms, also implement computer vision algorithms that are able   # 
# to generate threshold (Gaussian, mean).                             #
# ------------------------------------------------------------------- #

# ------------------------------------------------------------------- #
# --------------------------- Libraries ----------------------------- #
# ------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
import cv2

# ------------------------------------------------------------------- #
# -------------------------- Functions ------------------------------ #
# ------------------------------------------------------------------- #

# Convert already processed radar data (by FFTs) into images (2D) to use with OpenCV functions.
def cvtData2img(heatmaps, width, height):
    """[summary]

    Args:
        heatmaps ([type]): [description]
        width ([type]): [description]
        height ([type]): [description]

    Returns:
        [type]: [description]
    """
    data = np.expand_dims(np.array(list(heatmaps)),axis=2)                # Expand a axis to create a image
    blank_image = np.zeros((width,height,3), np.uint8)                    # Create a empty image 
    data = blank_image+data                                               # Add the data to a empty image
    image = cv2.cvtColor(data.astype('uint8'),cv2.COLOR_BGR2GRAY)         # Convert data in openCV images

    return image

#--------------------------------------------------------------------#
#--------------------------- Classes --------------------------------#
#--------------------------------------------------------------------#

# A "CFAR" algorithm based on computer vision thresholds.
class CFARCV():

    # Data (2D), type("mean" or "gaussian"), blockSize (window size), c (constant to decrease the value)
    def __init__(self, typecv, blockSize, c, limit = 120):
        self.typeCv = typecv
        self.blocksize = blockSize
        self.C = c
        self.limitCv = limit
        self.thrLowerValue = 0
        self.result = None

    # Generate a filtered image using computer vision algorithms.
    def thresholdImage(self, datacv):
        
        dataCV = np.array(datacv)
        shapeData = np.array(dataCV).shape
        width = shapeData[0]
        height = shapeData[1]
        dataResult = cvtData2img(dataCV, width, height) 

        if(self.typeCv == 'mean'):
            img_thr = cv2.adaptiveThreshold(dataResult,self.limitCv,
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY,self.blocksize,self.C)    
        elif(self.typeCv == 'gaussian'):
            img_thr = cv2.adaptiveThreshold(dataResult,self.limitCv,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,self.blocksize,self.C)
        else:    
            raise TypeError("Invalid Type, try 'mean' or 'gaussian'")
        
       
        dataResult[img_thr == 0] = self.thrLowerValue
    
        self.result = dataResult
        
        return self.result
    
    def update_thr(self, typecv, blockSize, c, limit = 120):

        self.typeCv = typecv
        self.blocksize = blockSize
        self.C = c
        self.limitCv = limit
    
    def get_image(self):
        return self.result

# Cell-averaging CFAR 1D
class CA_CFAR_1D():
    
    def __init__(self, freq):
        self.freq = np.array(freq)
        self.ca_thr = []
        self.goca_thr = []
        self.soca_thr = []
    
    # Return the threshold based on CA_CFAR
    def threshold_CA(self, N, Nc, alpha):
        """[summary]

        Args:
            N ([type]): [description]
            Nc ([type]): [description]
            alpha ([type]): [description]
        """
        for i in range(int(Nc/2) + 1, N - int(Nc/2) - 1):
            Leading_w = self.freq[:i - int(Nc/2)]                             # Creating the Leading Window
            Lagging_w = self.freq[i + int(Nc/2) + 1:N]                        # Creating the Lagging Window

            # Calculating the average of each window (Z)
            Avg_Leadw = np.sum(Leading_w) / len(Leading_w)
            Avg_Laggw = np.sum(Lagging_w) / len(Lagging_w)

            Z = (Avg_Leadw + Avg_Laggw) / 2                                   # Overall average of the windows
            Thr = Z * alpha
            self.ca_thr.append(Thr)

    # Return the threshold based on GOCA_CFAR
    def threshold_GOCA(self, N, Nc, alpha):
        for i in range(int(Nc/2) + 1, N - int(Nc/2) - 1):
            Leading_w = self.freq[:i - int(Nc/2)]                             # Creating the Leading Window
            Lagging_w = self.freq[i + int(Nc/2) + 1:N]                        # Creating the Lagging Window

            # Calculating the average of each window (Z)
            Avg_Leadw = np.sum(Leading_w) / len(Leading_w)
            Avg_Laggw = np.sum(Lagging_w) / len(Lagging_w)

            Z = max(Avg_Leadw, Avg_Laggw)                                     # Taking the highest value between the windows average
            Thr = Z * alpha
            self.goca_thr.append(Thr)

    # Return the threshold based on SOCA_CFAR
    def threshold_SOCA(self, N, Nc, alpha):
        for i in range(int(Nc/2) + 1, N - int(Nc/2) - 1):
            Leading_w = self.freq[:i - int(Nc/2)]                             # Creating the Leading Window
            Lagging_w = self.freq[i + int(Nc/2) + 1:N]                        # Creating the Lagging Window

            # Calculating the average of each window (Z)
            Avg_Leadw = np.sum(Leading_w) / len(Leading_w)
            Avg_Laggw = np.sum(Lagging_w) / len(Lagging_w)

            Z = min(Avg_Leadw, Avg_Laggw)                                     # Taking the highest value between the windows average
            Thr = Z * alpha
            self.soca_thr.append(Thr)

    def threshold(self, N, Nc, alpha):
        
        for i in range(int(Nc/2) + 1, N - int(Nc/2) - 1):
            Leading_w = self.freq[:i - int(Nc/2)]                             # Creating the Leading Window
            Lagging_w = self.freq[i + int(Nc/2) + 1:N]                        # Creating the Lagging Window

            # Calculating the average of each window (Z)
            Avg_Leadw = np.sum(Leading_w) / len(Leading_w)
            Avg_Laggw = np.sum(Lagging_w) / len(Lagging_w)
            
            Z_ca = (Avg_Leadw + Avg_Laggw) / 2                                # Overall average of the windows
            Z_goca = max(Avg_Leadw, Avg_Laggw)                                # Taking the highest value between the windows average
            Z_soca = min(Avg_Leadw, Avg_Laggw)                                # Taking the lowest value between the windows average

            Thr_ca = Z_ca * alpha
            Thr_goca = Z_goca * alpha
            Thr_soca = Z_soca * alpha

            self.ca_thr.append(Thr_ca)
            self.goca_thr.append(Thr_goca)
            self.soca_thr.append(Thr_soca)
    
    def update_thr(self, type, N, Nc, alpha, freq = None):

        if freq != None:
            self.freq = freq
        
        if type == 'All':
            self.threshold(N, Nc, alpha)
        
        elif type == 'CA':  
            self.threshold_CA(N, Nc, alpha)

        elif type == 'GOCA':
            self.threshold_GOCA(N, Nc, alpha)

        elif type == 'SOCA':
            self.threshold_SOCA(N, Nc, alpha)
        
        else:
            raise print("Invalid type. Try to use: 'All', 'CA', 'GOCA', 'SOCA'.")
    
    def get_thr(self, type):

        if type == 'All':
            return {'Threshold CA_CFAR': self.ca_thr,
                    'Threshold SOCA_CFAR': self.soca_thr,
                    'Threshold GOCA_CFAR': self.goca_thr}

        elif type == 'CA':
            return self.ca_thr
        
        elif type == 'GOCA':
            return self.goca_thr
        
        elif type == 'SOCA':
            return self.soca_thr
        
        else:
            raise print("Invalid type. Try to use: 'All', 'CA', 'GOCA', 'SOCA'.")

# Order-statistics CFAR 1D
class OS_CFAR_1D():
    # Data, types("CA", "GOCA", "SOCA"), number of cells, number of guard cells, threshold constant (alpha)
    def __init__(self, freq):
        self.freq = np.array(freq)
        self.threshold = []

    # Uses 2 reference window (left - right)
    def threshold_OS(self, N, Nc, K, alpha):
        
        for i in range(int(Nc/2), N - int(Nc/2)):
            Leading_w = self.freq[i - int(Nc/2): i]                           # Creates the "Leading Window"
            Lagging_w = self.freq[i + 1: i + int(Nc/2) + 1]                   # Creates the "LaggingWindow"

            # Creating the general list
            List = Leading_w + Lagging_w
            List.sort()                                                       # Sort the list
            Z = List[K]                                                       # Select the Kth order

            Thr = Z * alpha
            self.threshold.append(Thr)
    
    # Updates class attributes
    def update_thr(self, N, Nc, K, alpha, freq = None):
        if freq != None:
            self.freq = freq
        
        self.threshold_OS(N, Nc, K, alpha)
    
    def get_thr(self):
        return self.threshold


# CFAR 2D (using data from different channels)
class CFAR_2D():
    def __init__ (self,types, freq, alpha, N, Nr, Nt, Ng, K= 0):
        self.type =  types
        self.freq = freq                                                     # Magnitude[db] data of all 16 channels
        self.alpha = alpha
        self.N = N
        self.Nr = Nr
        self.Nt =  Nt
        self.Ng = Ng
        self.K = K
        self.ca_thr = []
        self.soca_thr = []
        self.goca_thr = []
        self.os_thr = []
        self.result = None
    
    # Takes the index of possible CUT's (Cell under test)
    def get_CUT(self):
        N = self.N
        Nr = self.Nr
        Ntc = self.Nt[1]                                                     # Number of training cell columns
        Ntr = self.Nt[0]                                                     # Number of training cell rows
        Ngc = self.Ng[1]                                                     # Number of guard cell columns
        Ngr = self.Ng[0]                                                     # Number of guard cell rows
        # Number of "general" rows and columns
        Gr = Ntr + Ngr
        Gc = Ntc + Ngc
        
        index_CUT = []                                                       # Saves the index of all possible CUT's

        for x in range(Gr, Nr - Gr):
            for y in range(Gc, N - Gc):
                index = [x, y]
                index_CUT.append(index)
        
        return index_CUT
    
    # Takes the index of the guard cells from a CUT 
    def get_GuardCells(self, index, Ngr, Ngc):

        guard_cells = []
        x, y = index[0], index[1]                                            # CUT index

        for i in range(x - Ngr, x + Ngr + 1):
            for j in range(y - Ngc, y + Ngc + 1):
                if [i, j] != [x,y]:
                    index_gc = [i, j]
                    guard_cells.append(index_gc)
                  
        return guard_cells

    def windows(self, ind, Ngr, Ngc, Gr, Gc):
        Leading_w = []
        Lagging_w = []
        x, y = ind[0], ind[1]
        guard_cells = self.get_GuardCells(ind, Ngr, Ngc)

        for i in range(x - Gr, x + Gr + 1):
            for j in range(y - Gc, y + 1):
                if [i, j] != [x, y] and [i, j] not in guard_cells:
                    Leading_w.append(self.freq[i][j])

        for i in range(x - Gr, x + Gr + 1):
            for j in range(y, y + Gc + 1):
                if [i, j] != [x, y] and [i, j] not in guard_cells:
                    Lagging_w.append(self.freq[i][j])
        
        return Leading_w, Lagging_w

    # CUT cell index, Number of values (columns), Number of rows, Number of training cells, Number of guard cells, alpha
    def threshold_CA(self, index):
        self.ca_thr = []

        Ntc = self.Nt[1]                                                     # Number of training cell columns
        Ntr = self.Nt[0]                                                     # Number of training cell rows
        Ngc = self.Ng[1]                                                     # Number of guard cell columns
        Ngr = self.Ng[0]                                                     # Number of guard cell rows
        # Number of "general" rows and columns
        Gr = Ntr + Ngr
        Gc = Ntc + Ngc

        for ind in index:
            Leading_w, Lagging_w = self.windows(ind, Ngr, Ngc, Gr, Gc)

            Avg_Leadw = sum(Leading_w) / len(Leading_w)
            Avg_Laggw = sum(Lagging_w) / len(Lagging_w)

            Z = (Avg_Leadw + Avg_Laggw) / 2                                  # Overall average of the windows

            Thr = Z * self.alpha
            self.ca_thr.append(Thr)
    
    def threshold_SOCA(self, index):
        
        self.soca_thr = []
        Ntc = self.Nt[1]                                                     # Number of training cell columns
        Ntr = self.Nt[0]                                                     # Number of training cell rows
        Ngc = self.Ng[1]                                                     # Number of guard cell columns
        Ngr = self.Ng[0]                                                     # Number of guard cell rows
        # Number of "general" rows and columns
        Gr = Ntr + Ngr
        Gc = Ntc + Ngc

        for ind in index:
            Leading_w, Lagging_w = self.windows(ind, Ngr, Ngc, Gr, Gc)

            Avg_Leadw = sum(Leading_w) / len(Leading_w)
            Avg_Laggw = sum(Lagging_w) / len(Lagging_w)

            Z = min(Avg_Leadw, Avg_Laggw)                                    # Minimum value of windows

            Thr = Z * self.alpha
            self.soca_thr.append(Thr)
    
    def threshold_GOCA(self, index):
        self.goca_thr = []

        Ntc = self.Nt[1]                                                     # Number of training cell columns
        Ntr = self.Nt[0]                                                     # Number of training cell rows
        Ngc = self.Ng[1]                                                     # Number of guard cell columns
        Ngr = self.Ng[0]                                                     # Number of guard cell rows
        # Number of "general" rows and columns
        Gr = Ntr + Ngr
        Gc = Ntc + Ngc

        for ind in index:
            Leading_w, Lagging_w = self.windows(ind, Ngr, Ngc, Gr, Gc)

            Avg_Leadw = sum(Leading_w) / len(Leading_w)
            Avg_Laggw = sum(Lagging_w) / len(Lagging_w)

            Z = max(Avg_Leadw, Avg_Laggw)                                    # Maximum value of windows

            Thr = Z * self.alpha
            self.goca_thr.append(Thr)
    
    # Kth value
    def threshold_OS(self, index):
        self.os_thr = []

        Ntc = self.Nt[1]                                                     # Number of training cell columns
        Ntr = self.Nt[0]                                                     # Number of training cell rows
        Ngc = self.Ng[1]                                                     # Number of guard cell columns
        Ngr = self.Ng[0]                                                     # Number of guard cell rows
        # Number of "general" rows and columns 
        Gr = Ntr + Ngr
        Gc = Ntc + Ngc
        
        for ind in index:
            Leading_w, Lagging_w = self.windows(ind, Ngr, Ngc, Gr, Gc)
            
            General_w = Leading_w + Lagging_w
            General_w.sort()
            Z = General_w[self.K]

            Thr = Z * self.alpha
            self.os_thr.append(Thr)
    
    def get_thr(self, type):

        if type == 'All':
            return {'Threshold CA_CFAR': self.ca_thr,
                    'Threshold SOCA_CFAR': self.soca_thr,
                    'Threshold GOCA_CFAR': self.goca_thr,
                    'Threshold OS_CFAR': self.os_thr}

        elif type == 'CA':
            return self.ca_thr
        
        elif type == 'GOCA':
            return self.goca_thr
        
        elif type == 'SOCA':
            return self.soca_thr
        
        elif type == 'OS':
            return self.os_thr
        
        else:
            raise print("Invalid type. Try to use: 'All', 'CA', 'GOCA', 'SOCA'.")
    
    def updateParameters(self, types, freq, alpha, N, Nr, Nt, Ng, K= 0):
        self.type =  types
        self.freq = freq                                                     # Magnitude[db] data of all 16 channels
        self.alpha = alpha
        self.N = N
        self.Nr = Nr
        self.Nt =  Nt
        self.Ng = Ng

    def thresholdImage(self, freq):
            self.freq = freq.transpose()[1:] 
            thr = []
            if self.type == 'CA':
                ind = self.get_CUT()
                self.threshold_CA(ind)
                thr = self.ca_thr
            elif self.type == 'GOCA':
                ind = self.get_CUT()
                self.threshold_GOCA(ind)
                thr = self.goca_thr
            elif self.type == 'SOCA':
                ind = self.get_CUT()
                self.threshold_SOCA(ind)
                thr = self.soca_thr
            elif self.type == 'OS': 
                ind = self.get_CUT()
                self.threshold_OS(ind)
                thr = self.os_thr
            else:
                print("Invalid type! Try to use: CA, GOCA, SOCA or OS")

            initial = 0                                                      # Dropping the first values
            final = len(thr) + initial                                       # Calculating the final index
            
            firstThr = thr[0]
            lastThr = thr[len(thr)-1]
            lenPad = int((len(freq) - len(thr))/2)
            
            for i in range(lenPad):
                thr.append(lastThr)

            for i in range(lenPad):
                thr.insert(0,firstThr)    

            
            x = len(freq)
            
            freq = freq[:x, initial:final]
            thr_list = []

            data = np.array(freq)                                            # Transform into a np.array
            
            for i in range(len(data[0])):
                thr_list.append(thr)
            thr_list = np.array(thr_list).transpose()
            data = np.where(data >= thr_list, data, 0)                       # Filter the image
            self.result = data

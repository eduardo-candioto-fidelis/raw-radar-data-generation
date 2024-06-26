# -*- coding: utf-8 -*-
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
# Developers: Herick Y. S. Ribeiro, Luiz H. Aguiar, Vinicius A. Leite.# 
#                                                                     # 
# e-mails: herick.ribeiro@facens.br, luizh5391@gmail.com,             #
# vini.fal2002@gmail.com                                              #
#                                                                     #
#                                                                     #
# ------------------------------------------------------------------- #

# ------------------------------------------------------------------- #
# Description:  Gen_Cube implements the processing of radar data using#
# FFTs. The HDF5 file already has all the necessary attributes, and   #
# then the processed values are stored in .npy files                  #
# ------------------------------------------------------------------- #

# ------------------------------------------------------------------- #
# -------------------------- Libraries ------------------------------ #
# ------------------------------------------------------------------- #

from curses import raw
import os
import h5py
from matplotlib import axes
import numpy as np

from numpy.core.fromnumeric import shape
from matplotlib import pyplot as plt
import numpy.matlib
from numpy.fft import fft
from pyqtgraph.metaarray.MetaArray import axis


from src import bsd_lib as bsd
from src import CFAR_Lib as cfar



# ------------------------------------------------------------------- #
# -------------------------- Classes -------------------------------- #
# ------------------------------------------------------------------- #

class Radar_Cube():
    C = 299792458  # Speed of light

    def __init__(self, file, dmax=-1):
        self.db = h5py.File(file, 'r', libver='latest')
        self._attrs = dict()
        self._originalCube = {'data': [], 'frame': 0}
        self._frameSize = 0
        self._RP = []
        self._vRangeExt = None
        self._dmax = dmax
        self._beforeMean = None

        self._getAttributes()

    def _getAttributes(self):

        self._attrs['channels']    = sorted([x for x in self.db.keys() if x[3:].isnumeric()], key=lambda x: int(x[3:]))
        self._attrs['fStart']      = self.db.attrs['fStrt'][0]  # Chirp Frequency Start
        self._attrs['fStop']       = self.db.attrs['fStop'][0]  # Chirp Frequency End
        self._attrs['nLoop']       = self.db.attrs['NLoop'][0]  # Necessary number of frames to calculate Doppler
        self._attrs['n']           = self.db.attrs['N'][0]  # Number of samples in a chirp
        self._attrs['tRampUp']     = self.db.attrs['TRampUp'][0]  # Ramp up time
        self._attrs['tRampDown']   = self.db.attrs['TRampDo'][0]  # Ramp down time
        self._attrs['calIm']       = self.db.attrs['CalIm']  # Calibration values for each antenna (imaginary)
        self._attrs['calRe']       = self.db.attrs['CalRe']  # Calibration values for each antenna (real)
        self._attrs['fusca']       = self.db.attrs['FuSca'][0]  # Fusca value
        self._attrs['fs']          = self.db.attrs['fs'][0]  # Sample rate
        self._tp                   = self._attrs['tRampUp'] + self._attrs['tRampDown']
        self._attrs['NrChn']       = self.db.attrs['NrChn'][0]

        self._attrs['calData']     = []
        self._attrs['S']           = self.__slope()
        self._attrs['vMax']        = self.__vMax()
        self._attrs['vScale']      = []
        self._attrs['deltaR']      = self.__deltaR()
        self._attrs['deltaV']      = self.__deltaVR()
        self._attrs['maxFrm']      = self.__maxFrame()
        self._imageFrame           = None
        self._frame                = None
        
        self.__set_calData()
        self.__distMax()
        self.maxAng = int(85 * 2)
        self.gen_Cube(0)
        self.range_Profile()
        self.range_Azimuth()
        self.range_Doppler()
        self.cam_Image()
        

    def __deltaR(self):
        # Calculates the distance resolution factor
        deltaR = self.C / (2 * (self._attrs['fStop'] - self._attrs['fStart']))

        return deltaR

    def __deltaVR(self):
        # Calculates the velocity resolution factor
        _lambda = (self.C / ((self._attrs['fStart'] + self._attrs['fStop']) / 2))
        vr = _lambda / (2 * self._attrs['nLoop'] * (self._attrs['tRampUp']))

        return vr

    def __maxFrame(self):

        maxFrame = self.db['Chn1'].shape[0]/self.db.attrs['NLoop'][0]

        return maxFrame

    def __set_calData(self):
        # Creates the calData matrix
        for i in range(len(self._attrs['calRe'])):
            self._attrs['calData'].append(complex(self._attrs['calRe'][i], self._attrs['calIm'][i]))

    def __slope(self):
        # Calculates de slope
        slope = (self._attrs['fStop'] - self._attrs['fStart']) / self._attrs['tRampUp']

        return slope

    def __vMax(self):
        # Calculates de maximum velocity
        wellenlaenge = (self.C / ((self._attrs['fStart'] + self._attrs['fStop']) / 2))
        vMax = wellenlaenge / (4 * (self._attrs['tRampUp']))

        return vMax

    def __distMax(self):
        # Calculates and set the maximum distance
        self._attrs['dMax'] = ((self._attrs['fs'] * self.C) / (2 * self._attrs['S'])) / 2

        if self._dmax < 0:
            self.setDist = self._attrs['dMax']
        else:
            self.setDist = self._dmax

    def cam_Image(self):
        dbkeys = list(self.db.keys())

        if dbkeys[0][2] == 'm':
            height = int(self.db[dbkeys[0]].shape[1])
            width  = int(self.db[dbkeys[0]].shape[0]/(len(list(self.db[dbkeys[1]])) * 3))
            im = np.array(self.db[dbkeys[0]][width*3*self._frame: width*3*self._frame +  width*3])
            im = im.flatten()
            self._imageFrame = np.array(np.reshape(im, (height, width,3)))
            return self._imageFrame
        else:
            print('Cam image not found!')
            

    def gen_Cube(self, frame):
        # Creates the cube and add it to the dictionary
        if frame >= self._attrs['maxFrm']-1:
            return False
        _cube = []
        _initChirp = frame * self._attrs['nLoop']
        _finalChirp = frame * self._attrs['nLoop'] + self._attrs['nLoop']
        self._frame = frame

        for index, channel in enumerate(self._attrs['channels']):
            _cube.append([])

            for chirp in self.db[channel][_initChirp:_finalChirp]:
                _res = chirp[1:]  # Drop first value (frame counter)
                _cube[index].append(_res)

        self._originalCube['data'] = _cube
        self._originalCube['frame'] = int(frame)
        
        return True

    def get_Cube(self, cube, frame):

        self._originalCube['data'] = cube
        self._originalCube['frame'] = int(frame)

    def _window(self, type):
        # Creates a 2D window to perfom the FFT
        if type == 0:
            window_smoothing_vel = np.ones((1, self._attrs['nLoop']))
            window_smoothing_chirp = np.hanning(self._attrs['n'] - 1)
            window_smoothing_total_chirp = sum(window_smoothing_chirp)
            window_smoothing_vel = np.transpose(np.hanning(self._attrs['nLoop']) * window_smoothing_vel)
            window_smoothing_total_vel = sum(window_smoothing_vel)
            window_smoothing_chirp = np.expand_dims(window_smoothing_chirp, axis=1)

        elif type == 1:
            window_smoothing_vel = np.ones((1, self._attrs['nLoop']))
            window_smoothing_chirp = np.hamming(self._attrs['n'] - 1)
            window_smoothing_total_chirp = sum(window_smoothing_chirp)
            window_smoothing_vel = np.transpose(np.hamming(self._attrs['nLoop']) * window_smoothing_vel)
            window_smoothing_total_vel = sum(window_smoothing_vel)
            window_smoothing_chirp = np.expand_dims(window_smoothing_chirp, axis=1)
        elif type == 2:
            window_smoothing_vel = np.ones((1, self._attrs['nLoop']))
            window_smoothing_chirp = np.kaiser(self._attrs['n'] - 1, 10)
            window_smoothing_total_chirp = sum(window_smoothing_chirp)
            window_smoothing_vel = np.transpose(np.kaiser(self._attrs['nLoop'], 10) * window_smoothing_vel)
            window_smoothing_total_vel = sum(window_smoothing_vel)
            window_smoothing_chirp = np.expand_dims(window_smoothing_chirp, axis=1)
        elif type == 3:
            window_smoothing_vel = np.ones((1, self._attrs['NrChn']))
            window_smoothing_chirp = np.hamming(self._attrs['n'] - 1)
            window_smoothing_total_chirp = sum(window_smoothing_chirp)
            window_smoothing_vel = np.transpose(np.hamming(self._attrs['NrChn']) * window_smoothing_vel)
            window_smoothing_total_vel = sum(window_smoothing_vel)
            window_smoothing_chirp = np.expand_dims(window_smoothing_chirp, axis=1)
        elif type == 4:
            window_smoothing_vel = np.ones((1, self._attrs['NrChn']))
            window_smoothing_chirp = np.hanning(self._attrs['n'] - 1)
            window_smoothing_total_chirp = sum(window_smoothing_chirp)
            window_smoothing_vel = np.transpose(np.hanning(self._attrs['NrChn']) * window_smoothing_vel)
            window_smoothing_total_vel = sum(window_smoothing_vel)
            window_smoothing_chirp = np.expand_dims(window_smoothing_chirp, axis=1)

        return window_smoothing_chirp, window_smoothing_vel, window_smoothing_total_chirp, window_smoothing_total_vel

    def range_Profile(self, raw_data_frame=None):
        if raw_data_frame is not None:
            raw_data_frame = np.array([[channel[1:]] for channel in raw_data_frame])
            self._originalCube['data'] = raw_data_frame
        
        self._RP = []
        NFFT = 2 ** 11

        kf = (self._attrs['fStop'] - self._attrs['fStart']) / self._attrs['tRampUp']
        vRange = np.arange(NFFT // 2) / NFFT * self._attrs['fs'] * self.C / (2 * kf)

        RMin = 0
        RMax = 100
        RMinIdx = np.argmin(np.abs(vRange - RMin))
        RMaxIdx = np.argmin(np.abs(vRange - RMax))
        self._vRangeExt = vRange[RMinIdx:RMaxIdx]

        cut = 40
        for chn in range(self._attrs['NrChn']):
            data = np.transpose(self._originalCube['data'][chn])
            window_chirp, _, sum_window_chirp, _ = self._window(3)
            
            RP = 2 * np.fft.fft(data[cut:, :] * window_chirp[cut:], n=NFFT, axis=0) / sum_window_chirp * self._attrs['fusca']
            #print(RP[0])
            result = 20 * np.log10(np.abs(RP))
            self._RP.append(RP)

        self._RP = np.array(self._RP)

        return {'Data': result,
                'Scale': self._vRangeExt}

    def range_Doppler(self):

        _, window_2D, _, sum_window_2D = self._window(0)

        NFFTVel = 2 ** 9

        self.range_Profile()

        RDAr = np.zeros((len(self._RP[0]) + 1, NFFTVel))
        for chn in self._RP:
            RD = np.fft.fftshift(np.fft.fft(chn * window_2D.transpose(), n=NFFTVel, axis=1)) / sum_window_2D
            #print(RD)
            RD2 = 20 * np.log10(np.abs(np.concatenate((RD[int(len(RD) / 2) - 1:, :], RD[:int(len(RD) / 2), :]))))
            RDAr += RD2
        RD = RDAr / len(self._RP)
        RD = RD[:int(len(RD) / 2)]

        self._VScale = np.linspace(-self._attrs['vMax'], self._attrs['vMax'], len(RD[0]))
        self._DScale = np.linspace(0, self._attrs['dMax'], len(RD))

        distMax = self._attrs['dMax']
        binLen = len(RD)
        RdMaxDistIndex = int((binLen * self.setDist) / distMax)
        

        return {'Data': RD[:RdMaxDistIndex, :],
                'Scale': self._VScale}

    def range_Azimuth(self):

        NFFTAng = 2 ** 7

        _, window_2D, _, sum_window_2D = self._window(1)

        RA = []

        cut = 0
        for chn in range(len(self._RP)):
            RA.append([])
            for i in range(cut, int(len(self._RP[0]) - cut)):
                RA[chn].append(self._RP[chn][i][int(self._attrs['nLoop'] / 2)])

        CalData = np.concatenate((self._attrs['calData'][0:self._attrs['NrChn']], self._attrs['calData'][self._attrs['NrChn'] + 1:]))
        mCalData = np.matlib.repmat(CalData, len(RA[0]), 1).transpose()

        RA = np.fft.fftshift(np.fft.fft(RA * mCalData[:self._attrs['NrChn']], n=NFFTAng, axis=0))
        RA = (RA * window_2D) / sum_window_2D
        RA = 20 * np.log10(np.abs(RA)).transpose()

        RA = np.concatenate((RA[int(len(RA) / 2) - 1:, :], RA[:int(len(RA) / 2), :]))
        RA = RA[:int(len(RA) / 2), :]

        x = np.zeros((cut, len(RA[0]))) - 80

        RA = np.concatenate((x, RA))

        RAMax = np.max(RA)

        RA = RA - RAMax
        distMax = self._attrs['dMax']
        binLen = len(RA)
        RaMaxDistIndex = int((binLen * self.setDist) / distMax)

        self._AScale = self.get_ascale(len(RA[0]))

        return {'Data': RA[:RaMaxDistIndex, :],
                'Scale': self._AScale}

    def get_ascale(self, len):
        # Calculates the mean frequency and wavelength        
        Fmean = (self._attrs['fStart'] + self._attrs['fStop']) / 2
        _lambda = (self.C / Fmean)

        # Generates the angles based on the results of the FFT
        ascale = np.fft.fftfreq(len, _lambda / 2)
        ascale = np.degrees(np.arcsin(_lambda * ascale))
        ascale = np.fft.fftshift(ascale)

        return ascale
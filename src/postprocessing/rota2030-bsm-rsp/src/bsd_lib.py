'''This lib provides the methods to detect the target.'''
# -------------------------------------------------------------------- #
# Developed by: Rota 2030 - Centro Universitário FACENS,               #
#                           Instituto Técnologico de Aeronautica,      #
#                           Robert Bosch,                              #
#                           Stellantis NV and                          #
#                           Technische Hochschule Ingolstadt.          #
#                                                                      #
# @author: BRAIN - Brazilian Artificial Intelligence Nucleus           #
# Centro Universitário FACENS, Sorocaba, Brazil, 2021.                 #
#                                                                      #
# Developers: Herick Y. S. Ribeiro, Luiz H. Aguiar, Vinicius A. Leite. #
#                                                                      #
# e-mails: herick.ribeiro@facens.br, luizh5391@gmail.com,              #
# vini.fal2002@gmail.com                                               #
#                                                                      #
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
# Description: This lib provide the method to detect the target.       #
# -------------------------------------------------------------------- #

import math
from collections import Counter
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from numpy import array, diag, eye, log
from numpy.linalg import det
from skimage.filters.rank import *
from skimage.morphology import *

from src import CFAR_Lib as cfar

def map_to_positive(x):
    """Maps the values of the array to the positive numbers"""
    if np.min(x) < 0:
        return x + np.abs(np.min(x))
    return x

def _format_measures(measures):
    """[summary]

    Args:
        measures ([type]): [description]

    Returns:
        [type]: [description]
    """
    frame_measures = []

    for measure in measures:
        measure_formated = [
            measure["Azi"],
            measure["Depth"],
            measure["Vel"],
        ]
        frame_measures.append(measure_formated)

    return frame_measures

class BlindspotDetector:
    """[summary]

    Returns:
        [type]: [description]
    """

    def __init__(self, radar):
        self._detect = Detect(radar)

        #classificator = joblib.load("./src/radar-classification-rfc-tabular.joblib")
        #self._tracker = Tracker(classificator=classificator, gate_size=14.75)

    def detect(self, rd_img, ra_img):
        """[summary]

        Args:
            rd_img ([type]): [description]
            ra_img ([type]): [description]

        Returns:
            [type]: [description]
        """
        target = self._detect.search_moving_targets(rd_img, ra_img)
        #target["tracks"] = []

        frame_measures = _format_measures(target["RAmov"])
        #target["tracks"] = self._tracker.track_objects(
        #    frame_measures, target["RAmov"]
        #)

        return target




class Detect:
    """[summary]
    """
    CENTER_RATIO = 0.015 # Percent of full size image that represent the center zone
    BSD_ZONE_RANGE = 90 # Sizeof the BSD zone [meters]

    # CFAR PARAMETERS
    CFAR_STEPS = 6 # Steps of the CFARCV parameters along the image

    # C parameters limits for Side zones and center zone
    RD_C = {
        'Side':
        {'max':19, 'min':8},
        'Center':
        {'max':32, 'min':27}
    }

    # Minimal area to consider reflection as target
    MIN_AREA = {
        'RD':
        {'Side': 70, 'Center': 2},
        'RA':
        {'Moving': 5, 'Static': 25}
    }


    def __init__(self, radar):
        self._radar = radar  # Radar object from sinal processing Lib
        self._max_dist = self._radar.setDist  # Max distance of the images

        self._cfar_rd = cfar.CFARCV(
            "gaussian", 3, -9
        )  # Instance of the CFAR algoritm

        self._RARoIDist = (
            1  # Size of RoI in meters to evaluate a moving target
        )
        self._RAdiv = 4  # Number of divisions to do the threshold on RA image
        self._RAc = -9  # Value of "C" constant of CFARCV threshold (RA image)

        self._RAstaticArea = 3

        # --------------- Images ------------ #
        self.RAThrImgMoving = None
        self.RAThrImgStatic = None
        self.RDThrImg = None
        self.RAThrImg = None
        

    def search_moving_targets(self, rd_img, ra_img):
        """Searchs and finds objects on the RD and RA Images"""

        if np.min(rd_img) < 0:
            self._min_rd_or = np.min(rd_img)

        if np.min(ra_img) < 0:
            self._min_ra_or = np.min(ra_img)

        rd_img = map_to_positive(rd_img)
        ra_img = map_to_positive(ra_img)


        # Sets the size of the center zone by a percent of the full RD image
        dopplerAxisLen = len(rd_img[0])
        center_hys = int(dopplerAxisLen * self.CENTER_RATIO)
        center = int(dopplerAxisLen/2)

        zones = np.split(rd_img, [center - center_hys, center + center_hys], axis=1)

        rd_right = zones[0]
        rd_center = zones[1]
        rd_left = zones[2]

        # Computes the bin index of the bsd zone
        binLen = len(rd_img)
        RightMaxDistIndex = int((binLen * self.BSD_ZONE_RANGE) / self._max_dist)

        # Apply CFAR filter to mitigate the noise in the image and highlight the targets on RD Image
        RDLeftThr = self.noise_filter_RD(
            rd_left, self.RD_C['Side']['max'], self.RD_C['Side']['min'], self.CFAR_STEPS
        )
        RDRightThr = self.noise_filter_RD(
            rd_right, self.RD_C['Side']['max'], self.RD_C['Side']['min'], self.CFAR_STEPS
        )
        RDcenterThr = self.noise_filter_RD(
            rd_center, self.RD_C['Center']['max'], self.RD_C['Center']['min'], self.CFAR_STEPS
        )

        # Locate the targets on each RD zone
        tgtLocInRDLeft = self.locate_targets(
            RDLeftThr, {"Des": 0, "Area": self.MIN_AREA['RD']['Side'], "type": "RD"}
        )

        tgtLocInRDCenter = self.locate_targets(
            RDcenterThr,
            {
                "Des": int(dopplerAxisLen / 2) - center_hys,
                "Area": self.MIN_AREA['RD']['Center'],
                "type": "RD",
            },
        )

        tgtLocInRDRight = self.locate_targets(
            RDRightThr,
            {
                "Des": int(dopplerAxisLen / 2) + 1 + center_hys,
                "Area": self.MIN_AREA['RD']['Side'],
                "type": "RD",
            },
        )

        # Append all targets estimations in RD Image
        targetsLocationsInRD = np.array([])

        if len(tgtLocInRDLeft) != 0:
            if len(targetsLocationsInRD) != 0:
                targetsLocationsInRD = np.concatenate(
                    [targetsLocationsInRD, tgtLocInRDLeft]
                )
            else:
                targetsLocationsInRD = tgtLocInRDLeft
        if len(tgtLocInRDRight) != 0:
            if len(targetsLocationsInRD) != 0:
                targetsLocationsInRD = np.concatenate(
                    [targetsLocationsInRD, tgtLocInRDRight]
                )
            else:
                targetsLocationsInRD = tgtLocInRDRight

        movingTargetsLocationsInRD = targetsLocationsInRD

        if len(tgtLocInRDCenter) != 0:
            if len(targetsLocationsInRD) != 0:
                targetsLocationsInRD = np.concatenate(
                    [targetsLocationsInRD, tgtLocInRDCenter]
                )
            else:
                targetsLocationsInRD = tgtLocInRDCenter

        # Apply CFAR filter to mitigate the noise in the image and highlight the targets on RA Image
        self.noise_filter_RA(ra_img)

        # Locate the targets on RA image
        TargetsLocationsInRA = self.locate_targets(
            self.RAThrImg, {"Des": 0, "Area": self.MIN_AREA['RA']['Moving'], "type": "RAm"}
        )

        self.RAThrImg += self._min_ra_or

        # Relates found targets on each image to get the list of detected objects
        tgtsInMovimentRA, tgtsStaticRA = self.data_association(
            movingTargetsLocationsInRD, TargetsLocationsInRA
        )
        tgtsInMovimentRAConvert = self.convert_cordinates(tgtsInMovimentRA)

        # Filters the reflections found on the same distance by intensity
        if len(tgtsInMovimentRAConvert) > 0:
            tgtsInMovimentRAmaxDb = self.maxDbFilter(tgtsInMovimentRAConvert)
        else:
            tgtsInMovimentRAmaxDb = []

        # Fill the Right size with zeros if necessary
        if len(rd_img) != len(RDRightThr):

            RDRightZero = np.zeros(
                (len(rd_img) - RightMaxDistIndex, len(RDRightThr[0]))
            ).astype(np.uint)
            RDRightThr = np.concatenate((RDRightThr, RDRightZero), axis=0)

        # Concatenate all images and brings back the original values
        self.RDThrImg = (
            np.concatenate((RDLeftThr, RDcenterThr, RDRightThr), axis=1)
            + self._min_rd_or
        )

        self.targets_Dict = {
            "RAmov": tgtsInMovimentRAmaxDb,
            "RAsta": self.convert_cordinates(tgtsStaticRA),
            "RD": targetsLocationsInRD,
        }

        return self.targets_Dict


    def centeroidnp(self, arr):

        """Finds the centroid of multiple points"""
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return int(sum_x / length), int(sum_y / length)


    def locate_targets(self, img, par, img_original=None):
        """Locates the reflecetions by contours on radar images"""

        # Chooses the min value by the image type
        if par["type"] == "RD":
            mins = self._min_rd_or
        elif par["type"] == "RA" or par["type"] == "RAm":
            mins = self._min_ra_or

        imgThr = img.astype(np.uint8)

        # Find the contours in the image
        contours, _ = cv2.findContours(
            imgThr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        imgcut = imgThr
        points = []

        if len(contours) != 0:
            i = 0
            for cont in contours:
                # Gets a bounding box of the reflections
                (x, y, w, h) = cv2.boundingRect(cont)
                if cv2.contourArea(cont) > par["Area"]:
                    # Segments the reflection of image
                    imgcut = imgThr[y : y + h, x : x + w]
                    xi = x + par["Des"]
                    xf = xi + w
                    yi = y
                    yf = y + h
                    roi = [xi, xf, yi, yf]

                    # Gets the max value and it location of the reflections
                    min, maxs, _, locmax = cv2.minMaxLoc(imgcut, mask=imgcut)
                    result = np.where(imgcut == maxs)
                    coordinates = np.array(list(zip(result[0], result[1])))

                    # Gets the mean of values inside the reflections
                    refMean = cv2.mean(imgcut + mins, mask=imgcut)[0]

                    if img_original is not None:
                        imgcut = img_original[y : y + h, x : x + w]

                    if len(coordinates) > 1:
                        yc, xc = self.centeroidnp(coordinates)
                        points.append(
                            [
                                par["Des"] + x + xc,
                                int(y + yc),
                                cv2.contourArea(cont),
                                maxs + mins,
                                min + mins,
                                refMean,
                                i,
                                roi,
                                imgcut,
                            ]
                        )
                    else:
                        points.append(
                            [
                                par["Des"] + x + locmax[0],
                                int(y + locmax[1]),
                                cv2.contourArea(cont),
                                maxs + mins,
                                min + mins,
                                refMean,
                                i,
                                roi,
                                imgcut,
                            ]
                        )
                    i += 1
        return points


    def noise_filter_RD(self, RDImg, maxC, minC, div):
        """Filters all noises on the RD image"""
        # Computes the size of each slice of the image
        RDzone = []
        binLen = len(RDImg)
        distdiv = int(((self._max_dist / div) / self._max_dist) * binLen)

        # Divide the image on 'div' slices
        RDThr = []
        for i in range(1, div + 1):
            RDzone.append(RDImg[int(distdiv * (i - 1)) : int(distdiv * i), :])

        # Apply the CFAR By dist
        for i, im in enumerate(RDzone):
            # Compute the C value for each distance
            C = ((maxC - minC) / (len(RDzone) - 1)) * (
                len(RDzone) - i - 1
            ) + minC

            # Compute the BlockSize value for each distance
            leftBlockSize = RDImg.shape[0]
            if leftBlockSize % 2 == 0:
                leftBlockSize += 1
            self._cfar_rd.blocksize = leftBlockSize
            self._cfar_rd.C = -C
            RDThr.append(self._cfar_rd.thresholdImage(im))

        # If have some diference of the original size and the final value fill with zeros
        RDThr = np.concatenate([x for x in RDThr])
        zer = np.zeros((RDImg.shape[0] - RDThr.shape[0], RDThr.shape[1]))
        RDThr = np.concatenate([RDThr, zer], axis=0)

        return dilation(RDThr)


    def noise_filter_RA(self, RAImg):
        """Filters all noises on the RA image"""
        # Compute the blocksize of CFARCV by a division of RA image
        BlockSize = int(len(RAImg) / self._RAdiv)
        if BlockSize % 2 == 0:
            BlockSize += 1
        self._cfar_rd.blocksize = BlockSize
        self._cfar_rd.C = self._RAc

        # Applies a median filter to mitigate noise in RA image
        medians = median(RAImg.astype(np.uint8).transpose(), disk(2))

        # Applies the CFARCV on RA image
        img = self._cfar_rd.thresholdImage(medians)
        imgThr = img.transpose()
        self.RAThrImg = dilation(imgThr).astype(np.float64)


    def data_association(self, targetsRD, targetsRA):
        """Relates the targets found on RD and RA Images by distance"""

        # Gets the range resolution of the radar
        deltaR = self._radar._attrs["deltaR"]
        targetsInmoviment = []
        targetsStatics = []
        
        x = 0
        RAid = 0
        RDid = 0

        # Gets each targets on RD image
        for i, t in enumerate(targetsRD):
            distRD = self._radar._DScale[int(t[1])]
            # Gets each targets on RA image
            for j, k in enumerate(targetsRA):
                distRA = self._radar._DScale[int(k[1])]
                # Verifies if the target are on the same distance
                if (distRD + self._RARoIDist) > distRA and (
                    distRD - self._RARoIDist
                ) < distRA:
                    objRange = distRA
                    objVelo = self._radar._VScale[int(t[0])] * 3.6
                    objAng = self._radar._AScale[int(k[0])]
                    peak = k[3]
                    targetsInmoviment.append(
                        {
                            "Id": x,
                            "Depth": objRange,
                            "Vel": objVelo,
                            "Azi": objAng,
                            "Db": peak,
                            "idRD": i,
                            "roiRD": t[7],
                            "imgCutRD": t[8],
                            "peakRD": t[3],
                            "minRD": t[4],
                            "areaRD": t[2],
                            "meanRD": t[5],
                            "idRA": RAid,
                            "roiRA": k[7],
                            "imgCutRA": k[8],
                            "peakRA": k[3],
                            "minRA": k[4],
                            "areaRA": k[2],
                            "meanRA": k[5],
                        }
                    )
                    RAid += 1
                    x += 1
                # elif(distRD + (self._RARoIDist+2*deltaR)) < distRA or (self._RARoIDist+2*deltaR) > distRA:
                else:
                    objRange = distRA
                    objVelo = 0
                    objAng = self._radar._AScale[int(k[0])]
                    peak = k[3]
                    if k[2] > self.MIN_AREA['RA']['Static']:
                        targetsStatics.append(
                            {
                                "Id": x,
                                "Depth": objRange,
                                "Vel": objVelo,
                                "Azi": objAng,
                                "Db": peak,
                                "idRD": i,
                                "roiRD": t[7],
                                "imgCutRD": t[8],
                                "peakRD": t[3],
                                "minRD": t[4],
                                "areaRD": t[2],
                                "meanRD": t[5],
                                "idRA": RAid,
                                "roiRA": k[7],
                                "imgCutRA": k[8],
                                "peakRA": k[3],
                                "minRA": k[4],
                                "areaRA": k[2],
                                "meanRA": k[5],
                            }
                        )
                    x += 1

                # print(" ["+str(i) + ","+str(j) +"] : "+ str(distRD)+' , '+str(distRA))
        if len(targetsRD) == 0:
            for j, k in enumerate(targetsRA):
                distRA = self._radar._DScale[int(k[1])]
                objRange = distRA
                objVelo = 0
                objAng = self._radar._AScale[int(k[0])]
                peak = k[3]
                if k[2] > self.MIN_AREA['RA']['Static']:
                    targetsStatics.append(
                        {
                            "Id": x,
                            "Depth": objRange,
                            "Vel": objVelo,
                            "Azi": objAng,
                            "Db": peak,
                            "idRD": 0,
                            "roiRD": 0,
                            "imgCutRD": 0,
                            "peakRD": 0,
                            "minRD": 0,
                            "areaRD": 0,
                            "meanRD": 0,
                            "idRA": RAid,
                            "roiRA": k[7],
                            "imgCutRA": k[8],
                            "peakRA": k[3],
                            "minRA": k[4],
                            "areaRA": k[2],
                            "meanRA": k[5],
                        }
                    )
        return targetsInmoviment, targetsStatics

    def round_number(self, x):
        return round(x * 2) / 2

    def maxDbFilter(self, targets):
        """Filters the reflections found on the same distance by intensity"""
        # Gets all targets
        dataFrame = pd.DataFrame.from_dict(targets)
        # Rounds the range of each target in .5
        # dataFrame['Depth'] = dataFrame['Depth'].apply(self.round_number)
        dataFrame["Depth"] = dataFrame["Depth"].astype(int)

        # Agroups the targets by range and get its indices
        distancia = dataFrame.groupby(by="Depth").indices
        dbList = []
        movingTargets = []

        # Filters the reflections found on the same distance by intensity
        for dist in distancia:
            dbList = []
            distlist = []
            for db in distancia[dist]:
                # print(targets[db])
                distlist.append(targets[db])
                dbList.append(targets[db]["Db"])
            # print(dist)
            idx = dbList.index(max(dbList))
            poped = distlist.pop(idx)
            distlist.insert(0, poped)

            # print(targets[distancia[dist][idx]])
            movingTargets.append(distlist[0])

        return movingTargets

    def convert_cordinates(self, tgts):
        """Converts the attributes of each targets (Range, Vel, Azi) into position (x, y), and velocity vectors(x, y)."""
        cvtTgt = []
        for i, tgt in enumerate(tgts):
            Px = tgt["Depth"] * math.cos(math.radians(tgt["Azi"]))
            Py = tgt["Depth"] * math.sin(math.radians(tgt["Azi"]))
            Vx = tgt["Vel"] * math.cos(math.radians(tgt["Azi"]))
            Vy = tgt["Vel"] * math.sin(math.radians(tgt["Azi"]))
            cvtTgt.append(
                {
                    "Id": i,
                    "Px": abs(Px),
                    "Py": Py,
                    "Vx": Vx,
                    "Vy": Vy,
                    "Db": tgt["Db"],
                    "Depth": tgt["Depth"],
                    "Vel": tgt["Vel"],
                    "Azi": tgt["Azi"],
                    "idRD": tgt["idRD"],
                    "roiRD": tgt["roiRD"],
                    "imgCutRD": tgt["imgCutRD"],
                    "peakRD": tgt["peakRD"],
                    "minRD": tgt["minRD"],
                    "areaRD": tgt["areaRD"],
                    "meanRD": tgt["meanRD"],
                    "idRA": tgt["idRA"],
                    "roiRA": tgt["roiRA"],
                    "imgCutRA": tgt["imgCutRA"],
                    "peakRA": tgt["peakRA"],
                    "minRA": tgt["minRA"],
                    "areaRA": tgt["areaRA"],
                    "meanRA": tgt["meanRA"],
                }
            )

        return cvtTgt

    def plot3D(self, img, coordinates=[]):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        listChn = []
        listChirp = []
        listSample = []
        img = np.array(img)
        c = []
        for i in range(len(img)):
            for j in range(len(img[0])):
                # if img[i][j] >-80:
                listChn.append(i)
                listChirp.append(j)
                c.append(img[i][j])

        ax.scatter(
            np.array(listChn), np.array(listChirp), c, c=c, cmap=cm.coolwarm
        )
        for co in coordinates:
            ax.scatter(co[0], co[1], img[co[0]][co[1]] + 0.5, c="g", s=50)
        # plt.show()
        plt.show()
        # plt.pause(3)
        # plt.close()


TENTATIVE, CONFIRMED, DELETED = 0, 1, 2


class Track:
    """
    This class represents a route, it is only used by the Tracker class.
    """

    def __init__(self, first_estimate, fx, hx):
        self.track_id = None
        self.estimates = [first_estimate]

        self.status = TENTATIVE
        self.count_new_associations = 0
        self.count_frames = 0
        self.count_no_new_sequential_associations = 0

        T = 0.1
        points = MerweScaledSigmaPoints(n=6, alpha=0.75, beta=2, kappa=-3)
        ukf = UnscentedKalmanFilter(
            dim_x=6, dim_z=4, fx=fx, hx=hx, dt=T, points=points
        )

        ukf.Q = self._calc_Q(T, 0.1, 0.1)
        ukf.R = diag([10, 10, 10, 10])
        ukf.SI = eye(4)
        ukf.S = eye(4)
        ukf.P = diag([38, 60, 40, 24, 10, 40])
        ukf.x = first_estimate

        self.ukf = ukf

        self.classes_history = []
        self.actual_class = "Nothing"

    def _calc_Q(self, T, std_x, std_y):
        x, y = std_x, std_y

        Q = array(
            [
                [
                    x * (T ** 4) / 4,
                    x * (T ** 3) / 2,
                    x * (T ** 2) / 2,
                    0,
                    0,
                    0,
                ],
                [x * (T ** 3) / 2, x * (T ** 2), x * T, 0, 0, 0],
                [x * (T ** 2) / 2, x * T, x, 0, 0, 0],
                [
                    0,
                    0,
                    0,
                    y * (T ** 4) / 4,
                    y * (T ** 3) / 2,
                    y * (T ** 2) / 2,
                ],
                [0, 0, 0, y * (T ** 3) / 2, y * (T ** 2), y * T],
                [0, 0, 0, y * (T ** 2) / 2, y * T, y],
            ]
        )

        return Q


class Tracker:
    """
    It implements multi-target tracking with the unscented kalman filter,
    the global nearest neighbor, and M/N. To use it it is necessary to
    instantiate an object and pass the measurements by the method track_objects
    every frame.


    Parameters
    ----------

    gate_size : float
        Circle size for associating measurements.

    associations_to_confime : int
        Number of associations to a track to confirm it.

    frames_to_confirme : int
        Maximum number of frames to reach the number of associations_to_confirme.

    no_associations_to_delete : int
        Number of frames with no association to delete a track.


    Methods
    -------

    track_objects
        Main method of the class used for tracing. Uses measurements in the format:
        [[azi, depth, velocity]]


    Example
    -------

    from tracker import Tracker

    tracker = Tracker()

    for frame_with_measurement in frames_with_measurements:
        tracks = tracker.track_objects(frame_with_measurement)
        ...
    """

    def __init__(
        self,
        classificator,
        gate_size=14.675,
        associations_to_confirme=3,
        frames_to_confirme=5,
        no_associations_to_delete=5,
    ):
        self._classifcator = classificator

        self._tracks = []
        self._next_track_id = 0

        self._gate_size = gate_size

        self._associations_to_confirme = associations_to_confirme
        self._frames_to_confirme = frames_to_confirme
        self._no_associations_to_delete = no_associations_to_delete

    def track_objects(self, measures, classification_datas):
        cartesian_measures = self._convert_to_cartesian(measures)

        self._predict_all()
        tracks_associateds = self._associate_all(cartesian_measures)
        self._manage_tracks(tracks_associateds)
        self._update_all(
            cartesian_measures, tracks_associateds, classification_datas
        )

        self._remove_deleted_tracks()
        self._create_new_tracks(
            cartesian_measures, tracks_associateds, classification_datas
        )

        return self._get_confirmed_tracks()

    def _convert_to_cartesian(self, measures):
        cartesian_measures = []

        for measure in measures:
            azi, rangem, velocity = (
                math.radians(measure[0]),
                measure[1],
                measure[2] / 3.6,
            )

            x = rangem * math.cos(azi)
            vx = velocity * math.cos(azi)
            y = rangem * math.sin(azi)
            vy = velocity * math.sin(azi)

            cartesian_measures.append([x, vx, y, vy])

        return cartesian_measures

    def _predict_all(self):
        for track in self._tracks:
            self._predict(track)

    def _predict(self, track):
        track.ukf.predict()

    def _associate_all(self, measures):
        tracks_associateds = [None] * len(self._tracks)

        if len(self._tracks) <= 0:
            return []

        general_distances = []

        for track in self._tracks:
            general_distance = []

            for measure in measures:
                general_d2 = self._calc_distance(track, measure)
                general_distance.append(general_d2)

            general_distances.append(general_distance)

        general_distances = array(general_distances)

        for measure_id in range(len(measures)):
            track_idl = general_distances[:, measure_id].argmin()
            minimal_distance = general_distances[:, measure_id].min()

            if minimal_distance <= self._gate_size:
                tracks_associateds[track_idl] = measure_id

        return tracks_associateds

    def _calc_distance(self, track, measure):
        x_prior, SI, S = track.ukf.x, track.ukf.SI, track.ukf.S

        y = measure - self._hx_ca(x_prior)

        d2 = y.T @ SI @ y
        general_d2 = d2 + log(det(S))

        return general_d2

    def _manage_tracks(self, tracks_associateds):
        for track_idl, measure_id in enumerate(tracks_associateds):
            track = self._tracks[track_idl]

            if measure_id is None:
                self._manage_without_association(track)
            else:
                self._manage_with_association(track)

    def _manage_without_association(self, track):
        if track.status == TENTATIVE:
            track.count_frames += 1
            if (
                self._associations_to_confirme - track.count_new_associations
                > self._frames_to_confirme - track.count_frames
            ):
                track.status = DELETED

        elif track.status == CONFIRMED:
            track.count_no_new_sequential_associations += 1
            if (
                track.count_no_new_sequential_associations
                >= self._no_associations_to_delete
            ):
                track.status = DELETED

    def _manage_with_association(self, track):
        if track.status == TENTATIVE:
            track.count_frames += 1
            track.count_new_associations += 1
            if track.count_new_associations >= self._associations_to_confirme:
                track.status = CONFIRMED
                track.track_id = self._next_track_id
                self._next_track_id += 1

        elif track.status == CONFIRMED:
            track.count_no_new_sequential_associations = 0

    def _update_all(self, measures, tracks_associateds, classification_datas):
        for track_idl, measure_id in enumerate(tracks_associateds):
            track = self._tracks[track_idl]

            if track.status == DELETED:
                continue

            if measure_id is None:
                self._update(None, track)
            else:
                measure = measures[measure_id]
                self._update(measure, track)

                classification_data = classification_datas[measure_id]
                self._classify_measure(classification_data, track)

    def _update(self, measure, track):
        track.ukf.update(measure)
        track.estimates.append(track.ukf.x)

    def _classify_measure(self, classification_data, track):
        CLASSES = [
            "Car",
            "Motorcycle",
            "Truck",
            "Person",
            "VulnerableUser",
            "Ghost",
            "CornerReflector",
            "Environment",
            "Noise",
        ]
        COMMON_FEATURES = [
            "Depth",
            "Vel",
            "Azi",
            "peakRA",
            "areaRA",
            "minRA",
            "meanRA",
            "peakRD",
            "areaRD",
            "minRD",
            "meanRD",
        ]

        data = []
        for common_feature in COMMON_FEATURES:
            data.append(classification_data[common_feature])
        data.append(classification_data["imgCutRD"].shape[0])
        data.append(classification_data["imgCutRD"].shape[1])
        data.append(classification_data["imgCutRA"].shape[0])
        data.append(classification_data["imgCutRA"].shape[1])

        class_pred = CLASSES[int(self._classifcator.predict([data]))]
        track.classes_history.append(class_pred)

        actual_class = Counter(track.classes_history).most_common()[0][0]
        print(actual_class)
        track.actual_class = actual_class

    def _remove_deleted_tracks(self):
        for track_idl, track in enumerate(self._tracks):
            if track.status == DELETED:
                del self._tracks[track_idl]

    def _create_new_tracks(
        self, measures, tracks_associateds, classification_datas
    ):
        for measure_id in range(len(measures)):
            if measure_id not in tracks_associateds:
                measure = measures[measure_id]
                classification_data = classification_datas[measure_id]
                self._create_new_track(measure, classification_data)

    def _convert_to_state(self, measure):
        return array([measure[0], measure[1], 0, measure[2], measure[3], 0])

    def _create_new_track(self, measure, classification_data):
        new_track = Track(
            self._convert_to_state(measure), self._fx_ca, self._hx_ca
        )

        self._classify_measure(classification_data, new_track)

        self._tracks.append(new_track)

    def _get_confirmed_tracks(self):
        confirmed_tracks = []

        for track in self._tracks:
            if track.status == CONFIRMED:
                track_copy = deepcopy(track)
                confirmed_tracks.append(
                    {
                        "id": track_copy.track_id,
                        "estimates": track_copy.estimates,
                        "actual_class": track_copy.actual_class,
                    }
                )

        return confirmed_tracks

    def _fx_ca(self, X, dt):
        x, vx, ax, y, vy, ay = X[0], X[1], X[2], X[3], X[4], X[5]

        x_prior = x + vx * dt + ax * (dt ** 2) / 2
        vx_prior = vx + ax * dt
        ax_prior = ax

        y_prior = y + vy * dt + ay * (dt ** 2) / 2
        vy_prior = vy + ay * dt
        ay_prior = ay

        return array(
            [x_prior, vx_prior, ax_prior, y_prior, vy_prior, ay_prior]
        )

    def _hx_ca(self, X):
        x, vx, y, vy = X[0], X[1], X[3], X[4]

        return array([x, vx, y, vy])

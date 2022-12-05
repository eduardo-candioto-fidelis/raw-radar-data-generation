# -*- coding: utf-8 -*-
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
# Developers: Eduardo C. Fidelis, Herick Y. S. Ribeiro,                #
# Luiz H. Aguiar, Vinicius A. Leite.                                   #
#                                                                      #
# e-mails: educafi41@gmail.com, herick.ribeiro@facens.br,              #
# luizh5391@gmail.com, vini.fal2002@gmail.com                          #
#                                                                      #
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
# Description: The goal of this code is processing and show radar      #
# images according to choosed parameters.                              #
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
# --------------------------- Libraries ------------------------------ #
# -------------------------------------------------------------------- #


import math

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import numpy as np

from src import bsd_lib as bsd
from src import plot



# Managing the terminal inputs
#argparser = argparse.ArgumentParser(
#        description='Radar Signal Processing')
#argparser.add_argument(
#    '--pltimg',
#    action='store_true',
#    help='Enables plot radar images')
#argparser.add_argument(
#    '--pltdt',
#    action='store_true',
#    help='Enables plot radar detections')
#argparser.add_argument(
#    '--pltrec',
#    action='store_true',
#    help='Enables plot recorded radar detections')
#argparser.add_argument(
#    '--rec',
#    action='store_true',
#    help='Enables record data')
#argparser.add_argument(
#    '--GPU',
#    action='store_true',
#    help='Enables data processing by the GPU') 
#argparser.add_argument(
#    action='store_false',
#    help='Enables use processed radar data')
#argparser.add_argument(
#    '--path',
#    metavar='path',
#    default='.',
#    help='Path to RadarLog folder')          
#
#args = argparser.parse_args()


# Flags to manage the functionalities of the code.
#PLOT_RADAR_IMAGES = args.pltimg
#PLOT_RECORD = args.pltrec 
#PLOT_DETECT = args.pltdt
#USE_GPU = args.GPU 
#RECORD_CSV = args.rec
#RAW_DATA = args.prc 


# Path to data
PATH = "/home/eduardo/workspace/rota2030-bsm-srdg/data/real/"

try:
    from src import Signal_Processing as sp
except:
    raise ValueError("Error to import Sinal Processing CPU")


# ------------------------------------------------------------------- #
# -------------------- Pre-processed Paths -------------------------- #
# ------------------------------------------------------------------- #

fileProcessed = PATH + "RadarLog/Dados Processados/EXP_16_C/"

# ------------------------------------------------------------------- #
# ------------------------- Raw-Data Paths -------------------------- #
# ------------------------------------------------------------------- #

file = PATH + "EXP_17_M.h5"

# ------------------------------------------------------------------- #
# -------------------- File name to record -------------------------- #
# ------------------------------------------------------------------- #

nameFile = "datas\\deletar.csv"

# ------------------------------------------------------------------- #
# ---------------- Setup of Radar function -------------------------- #
# ------------------------------------------------------------------- #

# Instantiates each of the classes
radar = sp.Radar_Cube(file, dmax=30)
detector_blind_spot = bsd.BlindspotDetector(radar)

#if PLOT_RADAR_IMAGES:
#    pltgraph = plot.plotGraphs(radar)

# Creates the dict to save the data
Pointsrecord = {
    "Frame": [],
    "TargetIndex": [],
    "Px": [],
    "Py": [],
    "Vx": [],
    "Vy": [],
    "Depth": [],
    "Vel": [],
    "Azi": [],
    "Db": [],
}

video = []

# Gets the images from our RawData Dataset and pre-process it
radar.gen_Cube(0)

# Processes the raw data by the CPU
RP = radar.range_Profile()["Data"]
RDImg = radar.range_Doppler()["Data"]
RAimg = radar.range_Azimuth()["Data"]

# Gets object on each image
target_dict = detector_blind_spot.detect(RDImg, RAimg)

radar._attrs['nLoop'] = 1
# Loading raw data:
data = sys.argv[1]
raw_data = np.load(f'./data/generated/{data}.npy')

print("Max frames:", int(radar._attrs["maxFrm"]))

ra_imgs = []

i = 0
while i < int(len(raw_data)):
    print(f"Frame: {i} / {len(raw_data)}")
    
    radar.range_Profile(raw_data[i])
    RAimg = radar.range_Azimuth()['Data']
    print(RAimg.shape)
    plt.figure(figsize=(10, 15))
    plt.imshow(RAimg)
    plt.savefig(f'./plots/generation/{i}.png')
    plt.close('all')

    ra_imgs.append(RAimg)

    detector_blind_spot._detect.noise_filter_RA(RAimg)

    #plt.figure(figsize=(10, 15))
    #plt.imshow(detector_blind_spot._detect.RAThrImg)
    #plt.savefig(f'./plots/real/EXP_17_M_rathr_all/{i}.png')
    #plt.close('all')

    #if PLOT_RADAR_IMAGES:
    #    try:
    #        # Plot the range-doppler image
    #        pltgraph.plotRangeDoppler(RDImg, target_dict["RD"])
    #        #        # Plot the range-azimuth image
    #        pltgraph.plotRangeAzimuthPolar(
    #            detector_blind_spot._detect.RAThrImg, target_dict["RAmov"], mins=True
    #        )
    #        pltgraph.update()
    #    except:
    #        print("Plot RD/RA error")
    #    
    detections = detector_blind_spot._detect.locate_targets(
        detector_blind_spot._detect.RAThrImg,
        {
            'Des': 0,
            'Area': detector_blind_spot._detect.MIN_AREA['RA']['Moving'],
            'type': 'RAm'
        }
    )

    #for detection in detections:
    #    print('A')
    #    print(radar._AScale[detection[0]])
    #    print('D')
    #    print(radar._DScale[detection[1]])
    #    print('Db')
    #    print(detection[3])
#
    #print('\n\n')

    i += 1

np.save('data/generation/generation_ra.npy', np.array(ra_imgs))

# ------------------------------------------------------------------- #
# ---------------- Record Data on CSV File -------------------------- #
# ------------------------------------------------------------------- #
if RECORD_CSV:
    detected_objects = pd.DataFrame(
        columns=[
            "Frame",
            "TargetIndex",
            "Px",
            "Py",
            "Vx",
            "Vy",
            "Azi",
            "Db",
        ]
    )

    for key in Pointsrecord:
        detected_objects[key] = Pointsrecord[key]
        detected_objects.to_csv(nameFile)



# ------------------------------------------------------------------- #
# ---------- Plot the path of the moving objects -------------------- #
# ------------------------------------------------------------------- #

if plotRecord:
    ax = plt.axes(projection="3d")
    ax.set_xlabel("Posição x")
    ax.set_ylabel("Posição y")
    ax.set_zlabel("Velocidade")
    ax.set_xlim((0, 25))
    ax.set_ylim((-20, 10))
    ax.set_zlim((-11, 5))

    graph_colors = [
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "gray",
        "olive",
        "cyan",
        "pink",
        "yellow",
        "aqua",
        "peru",
        "lightgreen",
        "darkviolet",
        "indigo",
        "rosybrown",
        "black",
    ]

    for index, track in enumerate(all_tracks):
        x = []
        y = []
        v = []
        vx = []
        vy = []
        for measure in track["estimates"]:
            x.append(measure[0])
            y.append(measure[3])
            vel_scalar_no_signal = math.sqrt(measure[1] ** 2 + measure[3] ** 2)
            vel_scalar = -vel_scalar_no_signal if measure[1] < 0 else vel_scalar_no_signal
            v.append(vel_scalar)
            vx.append(measure[1])
            vy.append(measure[4])

        ax.scatter(x, y, v, color=graph_colors[10 if index > 9 else index])
        # ax.quiver(x, y, v, vx, vy, 0, color='#66f')

    plt.show()
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

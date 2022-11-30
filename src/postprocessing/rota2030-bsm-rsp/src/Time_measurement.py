# -------------------------------------------------------------------#
# Developed by: Rota 2030 - Centro Unversitário FACENS,              #
#                           Instituto Técnologico de Aeronautica,    #
#                           Robert Bosch,                            #
#                           Stellantis NV and                        #
#                           Technische Hochschule Ingolstadt.        # 
#                                                                    #
# @author: BRAIN - Brazilian Artificial Intelligence Nucleus         #
# Centro Universitário FACENS, Sorocaba, Brazil, 2021.               #
#                                                                    #
# Developers: Herick Y. S. Ribeiro, Luiz H. Aguiar.                  # 
#                                                                    #
# e-mails: herick.ribeiro@facens.br, luizh5391@gmail.com.            #
#                                                                    #
#                                                                    #
# -------------------------------------------------------------------#

#--------------------------------------------------------------------#
#--------------------------- Libraries ------------------------------#
#--------------------------------------------------------------------#

import matplotlib.pyplot as plt
import timeit
from statistics import mean, stdev, variance
import random
import matplotlib.pyplot as plt

import cfar_lib as cfar
import gen_cube as cube

file = r"D:\RadarData\Measurement_16.h5"

radar = cube.Radar_Cube(file)

radar.set_Frame_size(128)
d = radar.Distance(0)

def Time_VC(intensity,blockSize):
    #print("entrou")
    inicio = timeit.default_timer()

    cfr = cfar.CFARCV(intensity, 'gaussian', blockSize, -12, limit=150)
    thrimg = cfr.thresholdImage()

    fim = timeit.default_timer()

    return (fim - inicio)

def Time_CFAR(intensity):

    intensity = intensity.transpose()[1:]
    
    Nr = len(intensity) 
    N = len(intensity[1])
    Nt = [29, 29]
    Ng = [1, 1]
    alpha_CA = 1.375
    alpha_OS = 1.35

    inicio = timeit.default_timer()

    cfr = cfar.CFAR_2D(intensity)
    Cut = cfr.get_CUT(N, Nr, Nt, Ng)
    #thr = cfr.Threshold(Cut, "OS", alpha_OS)
    thr = cfr.threshold_CA(Cut, N, Nr, Nt, Ng, alpha_CA)
    

    fim = timeit.default_timer()

    return (fim - inicio) 

def get_time(total_time):
    # Get the general average 
    gen_avgVC = mean(total_time)
    print("General average: {:.12f}ms".format(gen_avgVC))

    # Get the standard deviation
    sdevVC = stdev(total_time, xbar=None)
    print("Standard deviation: {:.12f}ms".format(sdevVC))

    # Get the variance
    varVC = variance(total_time, gen_avgVC)
    print("Variance: {:.12f}ms".format(varVC))

    return gen_avgVC

sizes = [random.randrange(1, 1000, 2) for i in range(0, 5)]
means = []

for i in sizes:

    # Get random intensity values
    ind = random.randint(0,300)
    v  = radar.Velocity(ind)
    intensity = v[1]

    # Get the total processing time of 1000 "Time_VC" uses
    total_timeVC = timeit.Timer("Time_VC(intensity, i)", globals=locals()).repeat(repeat = 10,number=10)
    # Get the average for each repetition
    total_timeVC = list(map(lambda x: x / 1000, total_timeVC))

    print("Blocksize: {} - Intensity: {}".format(i, ind))

    time_mean = get_time(total_timeVC)
    means.append(time_mean)

    print()

""" plt.figure(figsize=(10, 10))
plt.plot(sizes, means)
plt.show() """

time_cfar = Time_CFAR(intensity)

#print("Tempo do algoritmo de VC: {}".format(time_v))
print("Tempo do algoritmo CFAR: {}".format(time_cfar))

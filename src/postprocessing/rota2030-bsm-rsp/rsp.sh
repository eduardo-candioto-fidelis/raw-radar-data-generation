#!/bin/bash
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
# Description: The script goal is to prepare all dependencies for the  #
# correct work of the main code.                                       #
# -------------------------------------------------------------------- #

# Creates the help function that show all option of this script
Help()
{
   # Displays Help
   echo "This script is responsible for manage the options of rsp main code" 
   echo
   echo "Syntax: rsp.sh [-i|r|d|g|c|p|a]"
   echo "Options:"
   echo "i     Enables plot of charts Range-Doppler and Range-Azimuth. Disabled by default." 
   echo "r     Enables plot of recorded data. Disabled by default."
   echo "g     Enables data processing by the GPU. Disabled by default."
   echo "d     Disables detection plot. Enabled by default."
   echo "c     Enables record data in a CSV file. Disabled by default."
   echo "p     Enables use of processed radar data. Disabled by default."
   echo "a     Enables all visualization. Disabled by default."
   echo
}

# Initializes variables
plotImages=0
plotRecorded=0
processed=0
gpu=0
detection=0
record=0
# Manages options of this script
while getopts "irdgcpa" option
do
   case $option in
      h) # display Help
         Help
         exit;;
      s) scenario=${OPTARG}
        ;;
      p) plot=1
        ;;
      r) route=1
        ;;
      b) bb=1
        ;;
      i) ip=${OPTARG}
        ;;
      g) hud=1
	      ;;
      d) dimension=${OPTARG}
	      ;;
      c) rss=1
	      ;;
      m) m=${OPTARG}
        ;;
      a) plotImages=1
        plotRecorded=1
        processed=1
        gpu=1
        detection=1
        record=1   
         ;;
     \?) # incorrect option
         echo "Error: Invalid option"
         exit;;
   esac
done
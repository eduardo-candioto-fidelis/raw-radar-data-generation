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
# Description: The goal of this lib is provide the resources to plot   #
# images and videos of the radar data.                                 #
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
# --------------------------- Libraries ------------------------------ #
# -------------------------------------------------------------------- #


import numpy as np
import polarTransform
import matplotlib.pyplot as plt
import cv2
import math
from matplotlib.animation import FuncAnimation
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg

from src import CFAR_Lib as cfar


# -------------------------------------------------------------------- #
# ---------------------------- Function ------------------------------ #
# -------------------------------------------------------------------- #

def to_polar(cartesian_estimate):

    x, y = cartesian_estimate[0], cartesian_estimate[3]
    vx, vy = cartesian_estimate[1], cartesian_estimate[4]

    depth = (x**2 + y**2) ** 0.5
    azi = -math.atan2(y, x)

    vel = (vx**2 + vy**2) ** 0.5
    vel = -vel if vx < 0 else vel

    return [azi, depth, vel]

# -------------------------------------------------------------------- #
# ---------------------------- Classes ------------------------------- #
# -------------------------------------------------------------------- #


class plotGraphs():
    
    def __init__(self, radar):
        self._radar         = radar
        self._app           = QtGui.QApplication([])
        self._ViewRD        = None
        self._ImgRD         = None
        self._scatterRD     = None
        self._ViewRA        = None
        self._ImgRA         = None
        self._scatterRA     = None
        self._color         = ['r','b','g']
        self.rdTexts        = []
        self.ratext         = []
        self.InitplotRD()
        self.InitplotRA()

    
    def InitplotRD(self):

        self._ViewRD        = pg.PlotItem(title='Range-Doppler Image')
        self._scatterRD     = pg.ScatterPlotItem()
        self._textRD        = pg.TextItem(text="Ponto",fill='b')
        self._ViewRD.setLabel('left', 'R', units='m')
        self._ViewRD.setLabel('bottom', 'V', units='km/h')
        
        self._ImgRD         = pg.ImageView(view=self._ViewRD)
        self._ImgRD.show()
        self._ImgRD.ui.roiBtn.hide()
        self._ImgRD.ui.menuBtn.hide()
        #Img.ui.histogram.hide()

        self._ImgRD.getHistogramWidget().gradient.loadPreset('flame')
        self._ImgRD.getView().showGrid(True, True,3)
        self._ImgRD.getView().invertY(False)
        self._scatterRD.addPoints([{'pos': [0,0],'data':1}])

        self._ImgRD.addItem(self._scatterRD)
        self._ImgRD.addItem(self._textRD)
        self._posV = self._radar._VScale[0]*3.6
        self._posD = self._radar._DScale[0]
        self._resV = ((self._radar._VScale[-1]-self._radar._VScale[0])/len(self._radar._VScale))*3.6
        self._resD = self._radar._DScale[-1]/len(self._radar._DScale)

    def InitplotRA(self):
        self._ViewRA        = pg.PlotItem(title='Range-Azimuth Image')
        self._scatterRA     = pg.ScatterPlotItem()
        self._textRA        = pg.TextItem(text="Ponto",fill='b')
        self._ViewRA.setLabel('left', 'R', units='m')
        self._ViewRA.setLabel('bottom', 'R', units='m')
        self._ImgRA         = pg.ImageView(view=self._ViewRA)
        self._ImgRA.show()
        
        #self._ImgRA.ui.roiBtn.hide()
        self._ImgRA.ui.menuBtn.hide()
        #Img.ui.histogram.hide()

        self._ImgRA.getHistogramWidget().gradient.loadPreset('flame')
        self._ImgRA.getView().showGrid(True, True,3)
        self._ImgRA.getView().invertY(False)
        self._ImgRA.getView().invertX(True)
        
        self._scatterRA.addPoints([{'pos': [0,0],'data':1}])
        self._ImgRA.addItem(self._scatterRA)
        self._ImgRA.addItem(self._textRA)
        self._resD = self._radar._DScale[-1]/len(self._radar._DScale)
        self._posDn = -self._radar.setDist * np.cos(np.radians((180/2)-(self._radar.maxAng/2)))
        
    
    def plotRangeAzimuthPolar(self, RA, Detections = [], Statics = [], mins= True):
        
        minRA = np.min(RA)
        ajustAng = np.radians((180-self._radar.maxAng)/2)
        
        RA, _ = polarTransform.convertToCartesianImage(RA.transpose(),center="middle-middle" ,initialRadius=0, finalRadius=len(RA), initialAngle=np.pi+ajustAng,
                                                                    finalAngle= 2*np.pi - ajustAng)
        if mins == True:
            RA[RA >= 0] = minRA
        self._ImgRA.setImage(cv2.rotate(np.flipud(RA),cv2.ROTATE_90_COUNTERCLOCKWISE),pos=[self._posDn,self._posD],scale=[self._resD,self._resD])
        self.plot_grid(self._ViewRA)
        self._scatterRA.clear()
        self._textRA.hide()
        listPoit = []
        if len(Detections) > 0:
            listPoit = []
            txtList = []
            tgtList = []
           
            for detect in Detections:
                tgtList.append(detect)
                listPoit.append({'pos': [round(-1*detect['Py'],2),round(detect['Px'],2),-1],'data':1,'pen': {'color':self._color[1],'width':4}})
 

            for tgt in tgtList:
                text2 = pg.TextItem(text="Moving Target",fill=self._color[1])
                text2.setPos(round(-1*tgt['Py'],2),round(tgt['Px'],2))
                text2.setText('Moving Target - V:' +str(round(tgt['Vel'],2))+" [km/h]")
                self._ViewRA.addItem(text2)
                txtList.append(text2)
                text2.show()
        
        if len(Statics) > 0:
            
            for stat in Statics:
                listPoit.append({'pos': [round(-1*stat['Py'],2),round(stat['Px'],2),-1],'data':1,'pen': {'color':self._color[2],'width':4},'size':int(10)})
        self._scatterRA.setData(listPoit)

        self._ViewRA.setAspectLocked(False)
        pg.QtGui.QApplication.processEvents()
        if len(Detections) > 0:
            self.ratext = txtList
       
        
    def plotRangeDoppler(self, RDimg, Detections = []):
        
        self._ImgRD.setImage(np.flipud(np.rot90(RDimg)),pos=[self._posV,self._posD],scale=[self._resV,self._resD])
        self.plot_grid(self._ViewRD)
        self._scatterRD.clear()
        self._textRD.hide()
        if len(Detections) > 0:
            listPoit = []
            d=0
            c=0
            txtList = []
            for j in range(len(Detections)):
                if self._radar._VScale[int(Detections[j][0])]*3.6 > 1 or -1 > self._radar._VScale[int(Detections[j][0])]*3.6:
                    text2 = pg.TextItem(text="Teste",fill=self._color[0])
                else:
                    text2 = pg.TextItem(text="Teste",fill=self._color[1])
                text2.hide()
                self._ViewRD.addItem(text2)
                txtList.append(text2)

            for j,x in enumerate(Detections):

                if self._radar._VScale[int(x[0])]*3.6 > 1 or -1 > self._radar._VScale[int(x[0])]*3.6:
                    txtList[j].setPos(self._radar._VScale[int(x[0]-30)]*3.6,self._radar._DScale[int(x[1]-15)])
                    txtList[j].setText('D:' +str(round(self._radar._DScale[int(x[1])],2))+"[m],  V: "+ str(round(self._radar._VScale[int(x[0])]*3.6,2))+'[km/h]')
                    listPoit.append({'pos': [round(self._radar._VScale[int(x[0])]*3.6,2),round(self._radar._DScale[int(x[1])],2),-1],'data':1,'pen': {'color':self._color[0],'width':4}})
                    
                else:
                    txtList[j].setPos(self._radar._VScale[int(x[0])-10]*3.6,self._radar._DScale[int(x[1])]+1)
                    txtList[j].setText('D:' +str(round(self._radar._DScale[int(x[1])],2))+"[m]")
                    listPoit.append({'pos': [round(self._radar._VScale[int(x[0])]*3.6,2),round(self._radar._DScale[int(x[1])],2),-1],'data':1,'pen': {'color':self._color[1],'width':4}})
                
                txtList[j].show()
                d+=1
            self._scatterRD.setData(listPoit)
        
        self._ViewRD.setAspectLocked(False)
        pg.QtGui.QApplication.processEvents()
        if len(Detections)>0:
            self.rdTexts = txtList
        

    def plot_grid(self,view):

        top = view.getAxis("top")
        bottom = view.getAxis("bottom")
        left = view.getAxis("left")
        right = view.getAxis("right")
        top.setZValue(1)
        bottom.setZValue(1)
        left.setZValue(1)
        right.setZValue(1)
    
    def update(self):

        if len(self.rdTexts) > 0:
            for txt in self.rdTexts:
                txt.hide()
                self._ViewRD.removeItem(txt)
        if len(self.ratext) > 0:
            for txt in self.ratext:
                txt.hide()
                self._ViewRA.removeItem(txt)


class plotObjects():
    def __init__(self) -> None:
        plt.style.use('dark_background')
        self._fig = plt.figure(figsize=(10, 7))
    

    def show_in_live_plot(self, tracks, RAStat=[]):
        """[summary]

        Args:
            tracks ([type]): [description]
            RAStat (list, optional): [description]. Defaults to [].
        """
        azi_estimates = []
        depth_estimates = []
        vels_estimates = []
        actual_classes = []
        colors = []
        tracks_ids = []
        graph_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'gray',
                        'olive', 'cyan', 'pink', 'yellow', 'aqua', 'peru',
                        'lightgreen', 'darkviolet', 'indigo', 'rosybrown',
                        'black']
        for track in tracks:
            polar_estimate = to_polar(track['estimates'][-1])
            azi_estimates.append(polar_estimate[0])
            depth_estimates.append(polar_estimate[1])
            vels_estimates.append(polar_estimate[2])
            actual_classes.append(track['actual_class'])
            colors.append(graph_colors[track['id'] % 15])
            tracks_ids.append(track['id'])
        
        ax = self._fig.add_subplot(projection='polar')
        ax.set_rmax(30)
        ax.set_thetamax(90)
        ax.set_thetamin(-90)
        ax.set_ylim(0, 30)
        ax.set_theta_zero_location("N")
        if len(RAStat) > 0:
            statDepth = []
            statAzi = []
            
            for stat in RAStat:
                statDepth.append(stat['Depth'])
                statAzi.append(stat['Azi'])
            ax.scatter(statAzi, statDepth, c='green')
        ax.scatter(azi_estimates, depth_estimates, c=colors)
        for track_id, azi_estimate, depth_estimate, vel_estimate, actual_class in zip(tracks_ids, azi_estimates, depth_estimates, vels_estimates, actual_classes):
            ax.annotate(f'id: {track_id}, vel: {round(vel_estimate*3.6, 1)} km/h, {actual_class}', (azi_estimate, depth_estimate))

        ax.grid(b=True)
        ax.set_title("Moving targets Tracking ")
        plt.draw()
        plt.pause(1)
        plt.show(block=False)
        plt.clf()

    def show_in_live_plot_cartesian(self, tracks, RAStat=[]):
        x_estimates = []
        y_estimates = []
        vels_estimates = []
        actual_classes = []
        colors = []
        tracks_ids = []
        graph_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'gray', 
                        'olive', 'cyan', 'pink', 'yellow', 'aqua', 'peru', 
                        'lightgreen', 'darkviolet', 'indigo', 
                        'rosybrown', 'black'] 
                        

        for track in tracks:
            #polar_estimate = to_polar(track['estimates'][-1])
            x, y = track['estimates'][-1][0], track['estimates'][-1][3]
            vx, vy = track['estimates'][-1][1], track['estimates'][-1][4]
            x_estimates.append(x)
            y_estimates.append(y)
            vels_estimates.append(abs((vx**2 + vy**2) ** 0.5))
            actual_classes.append(track['actual_class'])
            colors.append(graph_colors[track['id'] % 15])
            tracks_ids.append(track['id'])
        
        ax = self._fig.add_subplot()
    
        ax.set_ylim(0, 30)
        ax.set_xlim(-30, 30)

        if len(RAStat) > 0:
            statPx = []
            statPy = []
            for stat in RAStat:
                statPx.append(stat['Px'])
                statPy.append(stat['Py'])
            ax.scatter(statPy, statPx, c='green')
        ax.scatter(y_estimates, x_estimates, c=colors)
        for track_id, x_estimate, y_estimate, vel_estimate, actual_class in zip(tracks_ids, x_estimates, y_estimates, vels_estimates, actual_classes):
            ax.annotate(f'id: {track_id}, vel: {round(vel_estimate*3.6, 1)} km/h, {actual_class}', (y_estimate, x_estimate))

        ax.grid(b=True, linewidth=0.1)
        ax.set_title("Moving targets Tracking")
        plt.draw()
        plt.pause(1)
        plt.show(block=False)
        plt.clf()


class plotVideos():
    ''' The class's goal is plot until six videos in the same figure by a list of parameters '''

    def __init__(self, radar,bsd):
        # List of plots/videos 
        self._plotList = []

        # Radar processing object
        self._radar = radar

        # Radar detect object
        self._bsd = bsd

        # Plot lib object
        self._plotLib = plotImages()

        # Figure where will plot all videos
        self._fig = None

        # Axes of the figure, in other words, object of each plot
        self._axesList = []
        
        # Lenght of the list of plots/videos
        self._lenList = 0

        # Initial and final frames
        self.finalFrame = 330
        self.initialFrame = 250

        # Scales of plot's axis
        self._scaleVelocity = self._radar.Velocity(1)['Scale']
        self._scaleDistance = self._radar.Distance(1, True)['Scale']
        self._scaleAzimuth = {'min':0,'max':180} 

        # Dictionary of scales 
        self._dictScales = {'v': self._label_create(self._scaleVelocity,5),
                            'd': self._label_create(self._scaleDistance,10),
                            'a': self._scaleAzimuth
                            }

        # List of CFAR for each plot
        self._cfarList = []
    
    def _init_CFAR(self):
        ''' The goal of this method is create a list of cfar object to process the desire plot, 
        in accordance with the type choosed  '''

        # Pass through each plot and creates the corresponding cfar object if choosed.
        for i, plot in enumerate(self._plotList):
            if plot['CFAR'] == 'CV':
                parameters = plot['CFARParameters']
                self._cfarList.append(cfar.CFARCV(parameters['Type'], 
                                                  parameters['BlockSize'], 
                                                  parameters['c'], 
                                            limit=parameters['Limit']))
            elif plot['CFAR'] == 'CFAR2D':
                parameters = plot['CFARParameters']
                self._cfarList.append(cfar.CFAR_2D(parameters['Type'],
                                                   parameters['Freq'],
                                                   parameters['Alpha'],
                                                   parameters['N'],
                                                   parameters['Nr'],
                                                   parameters['Nt'],
                                                   parameters['Ng'],
                                                   K = parameters['K']))
            elif plot['CFAR'] == None:
                self._cfarList.append(None)

    def _plot_setup(self):
        '''The goal of this method is creates and setup each plot by the parameters'''
        line = 1
        colun = 1

        # Computes the distribution of each plot
        if self._lenList == 2: 
            colun = 2
        elif self._lenList  > 2 and self._lenList <= 6:
            colun =  2
            line = math.ceil(self._lenList/2)
        
        # Creates the figure to plot
        self._fig = plt.figure(figsize=(line, colun))
      

        for i, plots in enumerate(self._plotList):
            
            # Verifies which type of plot is the current plot and set its parameters
            # STD is a standard plot with 2D heatmap.
            if plots['TypePlot'] == 'STD':

                # Creates each plot
                self._axesList.append(self._fig.add_subplot(line,colun,i+1))
                
                # Sets the title of each plot
                self._axesList[i].set_title(plots['Title'])
                
                # Sets labels of axis x of each graph
                self._axesList[i].set_xlabel(plots['LabelX'])
                labelx,tickx = self._dictScales[plots['ScaleX']]                
                self._axesList[i].set_xticks(tickx)
                self._axesList[i].set_xticklabels(np.flipud(labelx))

                # Sets labels of axis y of each graph
                self._axesList[i].set_ylabel(plots['LabelY'])
                labely,ticky = self._dictScales[plots['ScaleY']]  
                self._axesList[i].set_yticks(ticky)
                self._axesList[i].set_yticklabels(labely)

            # POLAR is a polar plot to range-azimuth images
            elif plots['TypePlot'] == 'POLAR' or plots['TypePlot'] == 'POLARDETECT':

                # Creates each plot in projection polar
                self._axesList.append(self._fig.add_subplot(line,colun,i+1, projection="polar"))

                # Sets the title of each plot
                self._axesList[i].set_title(plots['Title'])

                # Sets the theta limit of the plot [Field of View]
                self._axesList[i].set_thetamin(self._scaleAzimuth['min'])
                self._axesList[i].set_thetamax(self._scaleAzimuth['max'])
                self._axesList[i].set_xlabel(plots['LabelY'])
                self._axesList[i].set_xticks(np.linspace(0,math.pi,15))
                self._axesList[i].set_xticklabels(np.linspace(-90,90,15).astype(int))
            
            elif plots['TypePlot'] == 'VIDEO':
                pass            

    def _label_create(self,reference, size):
        '''The goal of this method is creates list of labels and tick by the reference list and size of result list''' 

        # Fills the array of labels
        label = np.flipud(np.around(np.linspace(np.amin(reference),np.amax(reference),size),2))
        # Fills the array of ticks per label
        tick = np.linspace(0, len(reference),size)
        
        return [label,tick]

    def _update_Figure(self,i):
        '''This method update the frame of each plot'''

        for j,plots in enumerate(self._plotList):
            
            # Verifies which type of plot is the current plot
            # STD is a standard plot with 2D heatmap.
            if plots['TypePlot'] == 'STD':
                if plots['Type'] == 'RD':
                    # Verifies if is needed to do cfar
                    if plots['CFAR'] != None:
                        self._cfarList[j].thresholdImage(self._radar.Velocity(i+self.initialFrame)['Data'])
                        self._axesList[j].imshow(np.flipud(self._cfarList[j].result), 
                                                            cmap='CMRmap',
                                                            interpolation='gaussian',
                                                            aspect='auto',
                                                            animated=True)
                    else:        
                        self._axesList[j].imshow(np.flipud(self._radar.Velocity(i+self.initialFrame)['Data']), 
                                                            cmap='CMRmap',
                                                            interpolation='gaussian',
                                                            aspect='auto',
                                                            animated=True)
            
            # POLAR is a polar plot to range-azimuth images
            elif plots['TypePlot'] == 'POLAR':
                self._axesList[j].cla()
                # Verifies if is needed to do cfar
                if plots['CFAR'] != None: 
                    self._cfarList[j].thresholdImage(self._radar.Azimuth(i+self.initialFrame)['Data'])
                    re = self._plotLib.plotPolar(np.fliplr(self._cfarList[j].result), self._scaleDistance)
                else:
                    re = self._plotLib.plotPolar(np.fliplr(self._radar.Azimuth(i+self.initialFrame)['Data']), self._scaleDistance)
                pc = self._axesList[j].pcolormesh(re['th'], re['r'], np.array(re['z']).transpose(),shading='auto')
                self._axesList[j].plot(re['azm'], re['r'], color='hot',ls='none')
                self._axesList[j].set_title(plots['Title'])
                self._axesList[j].set_thetamin(self._scaleAzimuth['min'])
                self._axesList[j].set_thetamax(self._scaleAzimuth['max'])
                self._axesList[j].set_rmax(max(self._scaleDistance))
                self._axesList[j].set_xlabel(plots['LabelY'])
                self._axesList[j].set_xticks(np.linspace(0,math.pi,15))
                self._axesList[j].set_xticklabels(np.linspace(-90,90,15).astype(int))
                self._axesList[j].grid()
                #self._fig.colorbar(pc,ax=self._axesList[j])
               
            
            elif plots['TypePlot'] == 'POLARDETECT':
                theta = []
                r = []
                self._axesList[j].cla()
                
                target = self._bsd.search_targets(i+self.initialFrame)
                
                if target != None:
                    k = 0
                    for tgt in target:
                        theta.append(target[tgt]['Azimuth'])
                        r.append(target[tgt]['Distance'])
                        self._axesList[j].text(0.40066,50-int(k*3),
                        "Reflections in moviment["+str(k)+"] - Distance: "+str(target[tgt]['Distance'])+"[m], Velocity: "+str(target[tgt]['Velocity'])+"[km/h], Azimuth: "+str(-1*target[tgt]['Azimuth'])+ '[º]')
                        k+=1
                    
                    theta= -1*np.radians(theta)
                    
                    self._axesList[j].scatter(theta,r)
                    self._axesList[j].grid()
                self._axesList[j].set_title('Range-Azimuth Detections')
                self._axesList[j].set_thetamin(-90)
                self._axesList[j].set_thetamax(90)
                self._axesList[j].set_rmax(max(self._scaleDistance))
                self._axesList[j].grid()
                self._axesList[j].set_theta_zero_location('N')
                self._axesList[j].set_xticks(np.linspace(-math.pi/2,math.pi/2,15))
                self._axesList[j].set_xticklabels(np.linspace(-90,90,15).astype(int))
                self._axesList[j].set_xlabel('Distance [m]')
                
            
        return self._axesList
        
                        
    def plotListVideo(self,List,frames):
        '''This method verify the size of list and call all of init methods'''
        
        # Sets the initial and final frame
        self.initialFrame = frames[0]
        self.finalFrame = frames[1]

        # Gets the list of plots
        self._plotList = List
        self._lenList = len(self._plotList)

        # Inits all of CFARs per plot (If have)
        self._init_CFAR()

        # Verifies the maximum size of plots
        if self._lenList > 0 and self._lenList <=6:
            self._plot_setup()
        else:
            print("ERROR: Wrong size of list ! Size must be between 1 and 6.")

    def show(self):
        '''This method creates the animate and show it'''
        # Creates the animatation
        ani = FuncAnimation(self._fig, self._update_Figure, frames=self.finalFrame-self.initialFrame,repeat=False, interval=50)
        
        # Inits the animation of all the plots.
              
        plt.show()


class plotImages():
    ''' The goal of this class is plot different type of graphs. '''

    def plot3d(self,img):
        ''' Method to plot 3d graphs'''

        x = []
        y = []
        z = []
        c = []

        # Creates the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Reorganises the image to plot 3d 
        for i in range(len(img)):
            for j in range(len(img[0])):
                    x.append(i)
                    y.append(j)
                    z.append(img[i][j])
                    c.append(img[i][j])
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        c = np.array(c)

        # Plots the 3D graphs
        ax.scatter(x, y, z, c=c, cmap=plt.hot())
        plt.show()

    def plotDistance(self, img, scale, frame, chirp):
        img = img[frame][chirp]
        fig = plt.figure()
        
        plt.plot(scale, img)
        plt.xlabel("Distance")
        plt.ylabel("X(dbV)")
        plt.title("Distance(m)")
        plt.show()

    def plotPolar(self,img,dist):
        ''' Get a radar 2D image and convert to polar dimentions'''
        azm = []
        z = []
        
        # Computes the azimuth resolution
        azmRel = list(map(lambda x: x*(180/len(img)), range(len(img))))
        radIndex = []
    
        
        for i in range(len(img[0])):
            for j in range(1+int(len(img)/len(img[0]))):
                radIndex.append(i)
        
        if(len(radIndex) > len(img)):
            sup  = len(radIndex) - len(img)
            radIndex= radIndex[sup:]

        # Organizes the value of each bin to plot polar.
        for i in range(len(radIndex)):
            z.append([])
            for j in range(len(radIndex)):
                z[i].append(img[i][radIndex[j]])

        
        azm = np.radians(azmRel) 

        # Convert dist scale into radial scale and azimuth to theta scale
        r, th = np.meshgrid(dist, azm)

        returns = {'azm': azm,
                    'r':  r,
                    'th': th,
                    'z':  z}
        
        return returns 

    def dataToHeatmap(self, data):

        # Convert data into RGB images
        img = plt.imshow(data,cmap='CMRmap',interpolation='gaussian')    
        img = np.array(img.make_image((1080,1920)))[0]
        
        return img

    
    def get_polar_Img(self,img,limit):
        '''Convert data to RGB polar image.'''
        verticalLinesImage = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Convert the data into polar data
        polarImage, _ = polarTransform.convertToCartesianImage(verticalLinesImage,center="middle-middle" ,initialRadius=limit[0], finalRadius=limit[1], initialAngle=np.pi,
                                                                    finalAngle= 2*np.pi)
        
        # Convert polar data into RGB image
        imgPlot = plt.imshow(polarImage,cmap='CMRmap',interpolation='gaussian',aspect='auto')
        imgPlotReturn = np.array(imgPlot.make_image((1920,1080)))

        return np.fliplr(imgPlotReturn[0])


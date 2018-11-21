Created on Mon Sep 18 15:37:06 2017

@author: alinsi
"""



################################################
import tifffile 
import numpy as np
import math
import cv2
from skimage.filters import threshold_sauvola
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening
#from scipy.ndimage import gaussian_filter
from skimage.util import invert
import imutils
import pandas as pd
#from timeit import default_timer as timer
import os 
#import sys
#if len(sys.argv) !=2: 
#placeread=str(sys.argv[2])
import argparse

 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,help="path to input image")
ap.add_argument("-o", "--output", required=True,help="path to output image")
ap.add_argument("-ci", "--capturedframei", required=True,help="starting frame number for reconstruction")
ap.add_argument("-cf", "--capturedframef", required=True,help="end frame number fore reconstruction")
args = vars(ap.parse_args())
# 
## load the input image from disk
folderread = args["input"]
foldersave = args["output"]

###################################################################

#
#import sys
#
#if len(sys.argv) <2:
#    print"at least 2 arguments needed"
#    sys.exit(1)
#else:
#    folderread=sys.argv[0]
#    foldersave=sys.argv[1]

#folderread="/sperm/spermnov19/Pos0"
#foldersave="/sperm/spermnov19/Pos0"
#    for x in sys.argv:
#        print "Argument:{}".format(x)
###################################################################
#ROIs=pd.read_csv('/home/alinsi/Desktop/10x_4/Pos0/ROIs.csv')
######################################################################
averagedframei=int(args["capturedframei"])##this is only for normallization--whats appropriate for normalization?
averagedframef=int(args["capturedframef"])
capturedframei=int(args["capturedframei"])#you cant choose 3 or 4 as it will1hink its 3 channel of colorshh
capturedframef=int(args["capturedframef"])
placeread=folderread+"/img_00000{}_Default_000.tif"#"/img_00000{}_Default_000.tif"#"/img_000000{}_Default_000.tif"
save_path=foldersave+"/MIPelapsed{}.tif"
placesave=foldersave+"/binary{}.tif"


rangenum=range(capturedframei,capturedframef+1)





minN = 1024##change


##enter some parameters for recontruction##
lambda0 = 0.000650
delx=5.32/1024
dely=6.66/1280
i=complex(0,1)
pi=math.pi

#2maxd=150#loop82ing of distance in mm from object to CCD , maximum is 22cm, 220mm,minmum is 60mm6=
#mind=30
#steps=5#5
maxd=285#loop82ing of distance in mm from object to CCD , maximum is 22cm, 220mm/20,minmum is 60mm6
mind=270
steps=1#5
###0=yes, 1=No normalizaiton step###
normalization=0
##########initialize empty arrays and indexing####################

##index the number of reconstructing distances
imageslices=int((maxd-mind)/steps)#when steps are not in intergers you must convert the index to int
slicerange = np.arange(0, imageslices, 1)    
##initializaitng empty arrays
threeD=np.empty((imageslices,minN,minN))##this is the stack of reconstructed images from a single frame
##captured frames? or imageslices
# this is reference point for finding dp, its a copy of threeD

interval=5
intermediate=int(len(range(averagedframei,averagedframef+1))/interval)
n=0##CHANGE THIS DOUBLE CHECK BEFORE WRITTING SCRIPT


minprojstack=np.empty((interval,minN,minN))
#minprojstack=np.empty((len(rangenum),minN,minN))
threeDPositions=pd.DataFrame()


##############calculate Transfer Function only once #############
   ###smart adaptives, this need to be manually entered or using computer vision to detect##########


dp = np.arange(mind, maxd, steps)

####################################################################


#distanced looping step

xmax=minN*delx/2
ymax=minN*dely/2

nx =np.arange (-minN/2,minN/2,1)
ny =np.arange (-minN/2,minN/2,1)sudo docker run -it -v /home:/app/data holographictest /bin/bash

X=nx*delx
Y=ny*dely

[XX,YY]=np.meshgrid(X,Y)



#########transfer function only needs to be calculated once for everything ##################
GG2=np.zeros((imageslices,minN,minN),dtype=np.complex64)



for d in dp:
    ind=int(round((d-mind)/steps))
    den = np.sqrt(d**2+XX**2+YY**2)
    num = np.exp(-i*2*pi/lambda0*den)
    g=i/lambda0*num/den#g is the impulse response of propagation
    GG2[ind,:,:]=np.fft.fft2(g)



#end1=timer()
#print("time for transfer is : ",(end1-start))
#######################normalize image###########################3

if normalization==0:
    
    stackss=np.float32(np.zeros((1024,1280)))
##remember to use the number of frames for all the captured images for good average affect
##this may differ from the number of frames you actually want to reconstruct
    for f in range(averagedframei,averagedframef+1):
        
        f2="{:04d}".format(f)
#        f2=f
        if os.path.exists(placeread.format((f2)))==False:
#            print "pass"            
            continue##if file dosnt exit 
        else:
            read_path=placeread.format((f2))
            h = (tifffile.imread(read_path))
            h1 = np.array(h).astype(np.float32)
    #        h1=gaussian_filter(h1,sigma=5.0)
#            (Nx,Ny)= h1.shape[:2]
#            minN = min(Nx,Ny)int(args["capturedframei"])
#            h1 = h1[:,1280-minN:]
#            
            stackss +=h1
    
    averagestack=stackss/(len(range(averagedframei,averagedframef+1)))
    
else:
    averagestack=1
    #averagestack=tifffile.imread('/home/alinsi/Desktop/biofilm/Pos0/reconstruct007.tif')
###############################Reconstruction################################################################
####################every 20 frames at a time######################################################################
#end2=timer()
#print("time for normalization is : ",(end2-start2))    
#start3=timer()

porosities=pd.DataFrame()
number=1
#background=tifffile.imread('/home/alinsi/Desktop/biofilm/Pos0/reconstruct007.tif')
for f in rangenum:
#    f2=f
    f2="{:04d}".format(f)
    if os.path.exists(placeread.format((f2)))==False:      
        continue##if file dosnt exit 
    else:    
        frameindex=(f-rangenum[0])%interval##################INTERVAL10
        read_path=placeread.format((f2))
        save_path2=placesave.format((f2))
        q = (tifffile.imread(read_path))            
        q1 = np.array(q).astype(np.float32)
    #    q1=gaussian_filter(q1,sigma=5.0)  
        q2= q1/averagestack  
        q3 = q2[:minN,:minN]  
        hh = np.array(q3).astype(np.float32)
        H=np.fft.fft2(hh)
        ###recontruction~~~~~~~~####
        for d in dp:
            ind=int(round((d-mind)/steps))
            Rec_image=np.fft.fftshift(np.fft.ifft2(GG2[ind]*H))
            amp_rec_image = np.abs(Rec_image)
            threeD[ind,:,:]=amp_rec_image.astype(np.float32)
            
        threeD=threeD/threeD.max()*255.0

#        tifffile.imsave(save_path2,threeD.astype(np.float32))  ##convert to 8bit 
#    end3=timer()
#    print("time for stack is : ",(end3-start3))    
#    start4=timer()
    
    ########maxproj##########################
        maxproj=np.ndarray.min(threeD,axis=0)
        tifffile.imsave(save_path2,maxproj.astype(np.float32))
    #
    #  
    
        #threshold to find 
        maxproj*=255.0/maxproj.max()
#        tifffile.imsave(save_path2,maxproj.astype(np.float32))
#    #
#    
#        start_time=timeit.default_timer()
        ##try otsu
#        thresh=threshold_otsu(maxproj)
#        binary=maxproj>thresh
        thresh_sauvola = threshold_sauvola((maxproj), window_size=51, k=0.4, r=35)
    #    thresh_sauvola = threshold_sauvola((maxproj), window_size=35, k=0.05, r=40)
        binary_sauvola = maxproj > thresh_sauvola
        
#    
        opening=binary_opening((binary_sauvola),np.ones((1,1),np.uint8))
    #    opening=binary_opening((binary_sauvola),np.ones((1,1),np.uint8))
    #    opening=opening.astype(np.uint8)
        opening=invert(binary_sauvola)
        opening=opening.astype(np.uint8)
            ########CALCULATE POROSITY###################################################################################
        porosity=np.float(np.count_nonzero(opening)/np.size(opening)*100.0)
        
        a=np.float(np.count_nonzero(opening))
        b=np.float(np.size(opening))
        porosity=a/b*100.0
    #    recordporosity=pd.DataFrame( {'frame':f,'porosity':porosity},index=[str(f)])
    #    porosities=porosities.append(recordporosity,ignore_index=True)
        ###############################################
        minprojstack[frameindex,:,:]=opening###IF U WANT TO SAVE MINIMUM PROJECTION
    
        maxproj=maxproj.astype(np.uint8)
#        tifffile.imsave()
        ##save binarized MIP###################
    
#        ###
#        color=cv2.cvtColor(opening,cv2.COLOR_GRAY2RGB)
#    
        cnts=cv2.findContours((opening.copy()),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts=cnts[0] if imutils.is_cv2() else cnts[1]
        
    #    end4=timer()
    #    print("time for binarization and contours is : ", (end4-start4))    
    #    start5=timer() 
    
        cXs=[]#a list of center x position of particles
        cYs=[]#a list of center y position of particles
        metricarray=[]
       
     #i need to create a 3d array for recording positions   
        threeDPosition=np.empty((len(rangenum),len(cnts),3))
        threeDPosition=np.empty((interval,len(cnts),3))
    #    #index for particles identified
        index=range(len(cnts))
        ###loop over the contours
        for k,c in enumerate (cnts):
             M = cv2.moments(c)
             (x,y,w,h2) = cv2.boundingRect(c)
             (x3,y3), radius= cv2.minEnclosingCircle(c)
    
             if (M["m00"] != 0):
    
                 cX = int(M["m10"] / M["m00"])
                 cY = int(M["m01"] / M["m00"])
#                 cv2.drawContours(color, [c], -1, (0, 255, 0), 3)
#                 cv2.putText(color, "particle {}".format(k), (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

#    
                 metricarray=[]  
                 for d2 in dp:##becareful d was used before, so use d2, ind was also used, use ind2
                     ind2=int(round((d2-mind)/steps))
                     particle = threeD[ind2,y:y+h2,x:x+w]
                     Metric=np.linalg.norm(particle)#CALCULATE AREA INTENSITY OF THE PARTICULAR PARTICLES IN EACH RECONSTRUCTION STEP
                     metricarray=np.append(metricarray,[Metric])# A LIST OF INTENSITY OF THE OBJECT OF INTERST AT EACH RECONSTRUCTION STEP
                 minimumvalue=min(metricarray)
                 minimumindex=np.argmin(metricarray)
                 minimumdp=dp[minimumindex]
                 #print ('the minimum value of {} is {} at index {} and distance {}'.format(i,minimumvalue,minimumindex,minimumdp))
                 
                 threeDPosition[frameindex,k,:]=cX,cY,minimumdp
                 
                 threeD2=maxproj[cY-100:cY+100,cX-100:cX+100]
                 if threeD2.size==0:
                     break
                # tifffile.imsave((foldersave+"/number{}.tif").format(number),minproj.astype(np.float32))                 
                 threeDposition=pd.DataFrame( {'frame':f,'particle': k, 'cX': cX, 'cY': cY, 'mindp': minimumdp,'radius':radius,'porosity':porosity,'number':number},index=[str(f)])
                 threeDPositions=threeDPositions.append(threeDposition,ignore_index=True)
                 number=number+1
         
             else:
                 pass
#######         
###########
    if ((f+1)%interval==0):

        threeDPositions.to_csv(args["output"]+'/threeDPositions{}.csv'.format(n))

        f3="{:04d}".format(n)
        savepath=save_path.format((f3))
#        tifffile.imsave(savepath,minprojstack.astype(np.float32))
#        tifffile.imsave(save_path2,threeD.astype(np.float32))
#        minprojstack=np.empty((interval,minN,minN))#reset minprojstack
        threeDPositions=pd.DataFrame()#reset 3d posiitions frame
        threeDPosition=np.empty((interval,len(cnts),3))#resetmin
        n=n+1
    else:
        pass
####now check if reconstruction was within focus distance by saving only one sampel##
#tifffile.imsave(save_path2,threeD.astype(np.float32))
window_size=51
kk=0.4
rr=35
imageinfo=pd.DataFrame( {'mind':mind,'maxd': maxd, 'step_size': steps, 'normalized(0yes1no)':normalization , 'laser(nm)':lambda0,'rr':rr,'kk':kk,'window':window_size},index=[0])
imageinfo.to_csv(args["output"]+'/imageinfoall.csv')
#tifffile.imsave(save_path.format(),minprojstack.astype(np.float32))

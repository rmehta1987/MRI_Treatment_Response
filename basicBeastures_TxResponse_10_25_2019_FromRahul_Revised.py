'''This script extracts intensity and histogram features of the ROI:
    these are the specifc features extracted:
    10th percentile
    25th percentile
    50th percentile
    75th percentile
    mean, variance, kurtosis, and the skewness calculated from the lower 25% of the histograms'''

import glob
import numpy as np
import matplotlib.pyplot as plt
#from visdom import Visdom
from scipy import stats
import math
import functools
import warnings


np.seterr(all='warn')  #for warnings 

def getFiles(file_path,name):

    afiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[0])))
    bfiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[1])))
    dfiles = sorted(glob.glob('%s/*%s.npy'%(file_path,name[2])))
    diff_files = sorted(glob.glob('%s/*%s.npy'%(file_path,name[3])))
    perf_files = sorted(glob.glob('%s/*%s.npy'%(file_path,name[4])))
    f_files = sorted(glob.glob('%s/*%s.npy'%(file_path,name[5])))
    lafiles = []
    lbfiles = []
    ldfiles = []
    ldiff_files = []
    lperf_files = []
    lf_files = []
    print ("obtaining files")
    for i, (a,b,d,e,f,g) in enumerate(zip(afiles, bfiles, dfiles, diff_files, perf_files, f_files)):
        lafiles.append(np.load(a, allow_pickle = True))
        lbfiles.append(np.load(b, allow_pickle = True))
        ldfiles.append(np.load(d, allow_pickle = True))
        ldiff_files.append(np.load(e, allow_pickle = True))
        lperf_files.append(np.load(f, allow_pickle = True))
        lf_files.append(np.load(g, allow_pickle = True))

    
    return lafiles, lbfiles, ldfiles, ldiff_files, lperf_files, lf_files

def getFiles2(file_path,name):

    afiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[0])))
    bfiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[1])))
    dfiles = sorted(glob.glob('%s/%s*.npy'%(file_path,name[2])))
    diff_files = sorted(glob.glob('%s/%s*.npy'%(file_path,name[3])))
    perf_files = sorted(glob.glob('%s/%s*.npy'%(file_path,name[4])))
    f_files = sorted(glob.glob('%s/%s*.npy'%(file_path,name[5])))
    lafiles = []
    lbfiles = []
    ldfiles = []
    ldiff_files = []
    lperf_files = []
    lf_files = []
    print ("obtaining files")
    for i, (a,b,d,e,f,g) in enumerate(zip(afiles, bfiles, dfiles, diff_files, perf_files, f_files)):
        lafiles.append(np.load(a, allow_pickle = True))
        lbfiles.append(np.load(b, allow_pickle = True))
        ldfiles.append(np.load(d, allow_pickle = True))
        ldiff_files.append(np.load(e, allow_pickle = True))
        lperf_files.append(np.load(f, allow_pickle = True))
        lf_files.append(np.load(g, allow_pickle = True))

    
    return lafiles, lbfiles, ldfiles, ldiff_files, lperf_files, lf_files


#Histogram (Weighted measurements) Statistics
# @freq is the probabilty at each bin
# @val is the bin center
def mean_(val, freq):
    return np.average(val, weights = freq)

def median_(val, freq):
    ord = np.argsort(val)
    cdf = np.cumsum(freq[ord])
    return val[ord][np.searchsorted(cdf[-1] // 2, cdf)]

def mode_(val, freq): #in the strictest sense, assuming unique mode
    return val[np.argmax(freq)]

def var_(val, freq):
    avg = mean_(val, freq)
    dev = freq * (val - avg) ** 2
    return dev.sum() / (freq.sum() - 1)

def kurt_(name,val,freq):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            numer = (val - mean_(val,freq))
            knumer = numer/np.sqrt(var_(val,freq))
            knumer = freq*(knumer**4)
            return (knumer.sum()/(freq.sum()))-3
        except Warning as e:
            print ("Kurt warning: ", e)
            undef = -39.1754
            print ("name: ", name)
            return undef
    

def skew_(name, val,freq):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            numer = (val - mean_(val,freq))
            snumer = numer/np.sqrt(var_(val,freq))
            snumer = freq*(snumer**3)
            return (snumer.sum()/(freq.sum()))
        except Warning as e:
            print ("Skew warning: ", e)
            print ("name: ", name)
            undef = -39.1754
            return undef

def percentile(thehist, N, percent):
    '''
    Find the percentile of a list of values.
    @parameter thehist - is the histgroam percentile is calculated on
    @parameter N - is the CDF of histogram
    @parameter percent - a float value from 0.0 to 1.0.
    @return - the bin-edge corresponding to the percentile value'''


    #theval = np.floor(N[-1]*percent)
    theval = N[-1]*percent
    theind = np.searchsorted(N,theval)
  
    if theind == 0:
        bc1 = thehist[1][theind+1]
        theperc = theval / thehist[0][theind]
        bc = bc1*theperc
    else:
        thediff = np.floor(theval - N[theind-1])
        theperc = thediff / thehist[0][theind]
        bc2 = thehist[1][theind+1]
        bc1 = thehist[1][theind]
        bc = ((bc2-bc1)*theperc)+bc1
    
    return bc,theind

def getHistFeatures(name, amaps, bmaps, dmaps, diffmaps, perfmaps, fmaps, bin_width):
    '''Get 10th, 25th, 75th percentile histogram features
    quartile [75th - 25th], mean, median, variance, kurtosis, and the skewness
    @name - name of different maps
    '''
    
    numfeats = 9
    #histograms of shape (amaps.shape and 9 for each histogram feature)
    ahist = np.zeros((amaps.shape[0],numfeats))
    bhist = np.zeros((amaps.shape[0],numfeats))
    dhist = np.zeros((amaps.shape[0],numfeats))
    diffhist = np.zeros((amaps.shape[0],numfeats))
    perfhist = np.zeros((amaps.shape[0],numfeats))
    fhist = np.zeros((amaps.shape[0],numfeats))
    lahist = []
    lbhist = []
    ldhist = []
    ldiffhist = []
    lperfhist = []
    lfhist = []

    plt.figure()
    amap = fmaps[5].reshape(96,96)
    imgplot = plt.imshow(amap)
    imgplot.set_cmap('nipy_spectral')

    #skew_kurt = viz.line(
    #Y=np.column_stack((np.zeros((1)),np.zeros((1)))),
    #X=np.column_stack((np.zeros((1)),np.zeros((1)))),
    #opts=dict(xlabel='sample',ylabel='Value',title='Skew and Kurtosis values {}'.format(name[-1])))

    """
    percentiles = viz.line(
    Y=np.column_stack((np.zeros((1)),np.zeros((1)),np.zeros((1)),np.zeros((1)))),
    X=np.column_stack((np.zeros((1)),np.zeros((1)),np.zeros((1)),np.zeros((1)))),
    opts=dict(xlabel='sample',ylabel='Value',title='Percentiles'))

    mean_mid = viz.line(
    Y=np.column_stack((np.zeros((1)),np.zeros((1)))),
    X=np.column_stack((np.zeros((1)),np.zeros((1)))),
    opts=dict(xlabel='sample',ylabel='Value',title='Mean and Median')) """

    amin = (amaps.ravel()[np.flatnonzero(amaps)]).min()
    amax = (amaps.ravel()[np.flatnonzero(amaps)]).max()
    bmin = (bmaps.ravel()[np.flatnonzero(bmaps)]).min()
    bmax = (bmaps.ravel()[np.flatnonzero(bmaps)]).max()
    dmin = (dmaps.ravel()[np.flatnonzero(dmaps)]).min()
    dmax = (dmaps.ravel()[np.flatnonzero(dmaps)]).max()
    diffmin = (diffmaps.ravel()[np.flatnonzero(diffmaps)]).min()
    diffmax = (diffmaps.ravel()[np.flatnonzero(diffmaps)]).max()
    perfmin = (perfmaps.ravel()[np.flatnonzero(perfmaps)]).min()
    perfmax = (perfmaps.ravel()[np.flatnonzero(perfmaps)]).max()
    fmin = (fmaps.ravel()[np.flatnonzero(fmaps)]).min()
    fmax = (fmaps.ravel()[np.flatnonzero(fmaps)]).max()
    
    #print ("amin {:.2f} amax {:.2f} bmin {:.2f} bmax {:.2f} dmin {:.2f} dmax {:.2f} diffmin {:.2f} diffmax {:.2f} perfmin {:.2f} perfmax {:.2f} fmin {:.2f} fmin {:.2f}"
    #       .format(amin, amax, bmin, bmax, dmin, dmax, diffmin, diffmax, perfmin, perfmax, fmin, fmax))

    for i, (a, b, d, e, f, g) in enumerate(zip(amaps, bmaps, dmaps, diffmaps, perfmaps, fmaps)):

        #First generate histogram for amap:
        thist = np.histogram(a.ravel()[np.flatnonzero(a)],bins=bin_width[0],range=(amin,amax),density=True)
        cs  = np.cumsum(thist[0])
        temp = "name {} and map number {}".format(name[0],i)
        #calculate histogram features
        lahist.append(thist)
        ahist[i][0], _ = percentile(thist, cs.tolist(), 0.10)
        ahist[i][1], _ = percentile(thist, cs.tolist(), 0.25)
        ahist[i][2], _ = percentile(thist, cs.tolist(), 0.50) #median
        ahist[i][3], idx = percentile(thist, cs.tolist(), 0.75)

        #calculate bin centers:
        bc = thist[1] + (thist[1][1] - thist[1][0])/2

        
        #calculate median
        #amid = ahist[i][2]
        
        #calculate mean
        amean = mean_(bc[:-1],thist[0])

        #calculate variance
        avar = var_(bc[:-1],thist[0])

        #calculate kurtosis
        akurt = kurt_(temp,bc[:-1],thist[0])
        
        #calculate skew
        askew = skew_(temp,bc[:-1],thist[0])

        #calculate quartile 
        aquartile = ahist[i][3]-ahist[i][1]

        ahist[i][4] = aquartile
        ahist[i][5] = amean
        ahist[i][6] = avar
        ahist[i][7] = akurt
        ahist[i][8] = askew

        #First generate histogram for bmap:
        thist = np.histogram(b.ravel()[np.flatnonzero(b)],bins=bin_width[1],range=(bmin,bmax),density=True)
        temp = "name {} and map number {}".format(name[1],i)
        cs  = np.cumsum(thist[0])
        lbhist.append(thist)
        #calculate histogram features
        bhist[i][0], _ = percentile(thist, cs.tolist(), 0.10)
        bhist[i][1], _ = percentile(thist, cs.tolist(), 0.25)
        bhist[i][2], _ = percentile(thist, cs.tolist(), 0.50)
        bhist[i][3], idx = percentile(thist, cs.tolist(), 0.75)

        #calculate bin centers:
        bc = thist[1] + (thist[1][1] - thist[1][0])/2
        
        #calculate median
       # bmid = bhist[i][2]
        
        #calculate mean
        bmean = mean_(bc[:-1],thist[0])

        #calculate variance
        bvar = var_(bc[:-1],thist[0])

        #calculate kurtosis
        bkurt = kurt_(temp,bc[:-1],thist[0])
        
        #calculate skew
        bskew = skew_(temp, bc[:-1],thist[0])

        #calculate quartile
        bquartile = bhist[i][3]-bhist[i][1]

        bhist[i][4] = bquartile
        bhist[i][5] = bmean
        bhist[i][6] = bvar
        bhist[i][7] = bkurt
        bhist[i][8] = bskew


        #First generate histogram for dmap:
        thist = np.histogram(d.ravel()[np.flatnonzero(d)],bins=bin_width[2],range=(dmin,dmax),density=True)
        cs  = np.cumsum(thist[0])
        ldhist.append(thist)
        temp = "name {} and map number {}".format(name[2],i)
        #calculate histogram features
        dhist[i][0], _ = percentile(thist, cs.tolist(), 0.10)
        dhist[i][1], _ = percentile(thist, cs.tolist(), 0.25)
        dhist[i][2], _ = percentile(thist, cs.tolist(), 0.5)
        dhist[i][3], idx = percentile(thist, cs.tolist(), 0.75)

     
        #calculate bin centers:
        bc = thist[1] + (thist[1][1] - thist[1][0])/2
               
        #calculate median
        #dmid = dhist[i][2]
        
        #calculate mean
        dmean = mean_(bc[:-1],thist[0])

        #calculate variance
        dvar = var_(bc[:-1],thist[0])

        #calculate kurtosis
        dkurt = kurt_(temp, bc[:-1],thist[0])
        
        #calculate skew
        dskew = skew_(temp, bc[:-1],thist[0])

        #calculate quartile
        dquartile = dhist[i][3]-dhist[i][1]

        dhist[i][4] = dquartile
        dhist[i][5] = dmean
        dhist[i][6] = dvar
        dhist[i][7] = dkurt
        dhist[i][8] = dskew

        #####Ivim histograms***************************************************************************************#

        #First generate histogram for diff_map:
        thist = np.histogram(e.ravel()[np.flatnonzero(e)],bins=bin_width[3],range=(diffmin,diffmax),density=True)
        cs  = np.cumsum(thist[0])
        ldiffhist.append(thist)
        temp = "name {} and map number {}".format(name[3],i)
        #calculate histogram features
        diffhist[i][0], _ = percentile(thist, cs.tolist(), 0.10)
        diffhist[i][1], _ = percentile(thist, cs.tolist(), 0.25)
        diffhist[i][2], _ = percentile(thist, cs.tolist(), 0.50)
        diffhist[i][3], idx = percentile(thist, cs.tolist(), 0.75)

     

        #calculate bin centers:
        bc = thist[1] + (thist[1][1] - thist[1][0])/2

       
        #calculate median
        #diffmid = diffhist[i][2]
        
        #calculate mean
        diffmean = mean_(bc[:-1],thist[0])

        #calculate variance
        diffvar = var_(bc[:-1],thist[0])

        #calculate kurtosis
        diffkurt = kurt_(temp, bc[:-1],thist[0])
        
        #calculate skew
        diffskew = skew_(temp, bc[:-1],thist[0])

        #calculate quartile
        diffquartile = diffhist[i][3]-diffhist[i][1]

        diffhist[i][4] = diffquartile
        diffhist[i][5] = diffmean
        diffhist[i][6] = diffvar
        diffhist[i][8] = diffkurt
        diffhist[i][8] = diffskew


        #First generate histogram for perf_map:
        thist = np.histogram(f.ravel()[np.flatnonzero(f)],bins=bin_width[4],range=(perfmin,perfmax),density=True)
        cs  = np.cumsum(thist[0])
        lperfhist.append(thist)
        temp = "name {} and map number {}".format(name[4],i)
        #calculate histogram features
        perfhist[i][0], _ = percentile(thist, cs.tolist(), 0.10)
        perfhist[i][1], _ = percentile(thist, cs.tolist(), 0.25)
        perfhist[i][2], _ = percentile(thist, cs.tolist(), 0.50)
        perfhist[i][3], idx = percentile(thist, cs.tolist(), 0.75)

   

        #calculate bin centers:
        bc = thist[1] + (thist[1][1] - thist[1][0])/2

  
        
        #calculate median
        #perfmid = perfhist[i][2]
        
        #calculate mean
        perfmean = mean_(bc[:-1],thist[0])

        #calculate variance
        perfvar = var_(bc[:-1],thist[0])

        #calculate kurtosis
        perfkurt = kurt_(temp, bc[:-1],thist[0])
        
        #calculate skew
        perfskew = skew_(temp, bc[:-1],thist[0])

        #calculate quartile
        perfquartile = perfhist[i][3]-perfhist[i][1]

        perfhist[i][4] = perfquartile
        perfhist[i][5] = perfmean
        perfhist[i][6] = perfvar
        perfhist[i][7] = perfkurt
        perfhist[i][8] = perfskew



        #First generate histogram for f_map:
        thist = np.histogram(g.ravel()[np.flatnonzero(g)],bins=bin_width[5],range=(fmin,fmax),density=True)
        cs  = np.cumsum(thist[0])
        lfhist.append(thist)
        temp = "name {} and map number {}".format(name[5],i)
        #calculate histogram features
        fhist[i][0], _ = percentile(thist, cs.tolist(), 0.10)
        fhist[i][1], _ = percentile(thist, cs.tolist(), 0.25)
        fhist[i][2], _ = percentile(thist, cs.tolist(), 0.50)
        fhist[i][3], idx = percentile(thist, cs.tolist(), 0.75)

 

        #calculate bin centers:
        bc = thist[1] + (thist[1][1] - thist[1][0])/2

        #calculate median
        #fmid = fhist[i][2]
        
        #calculate mean
        fmean = mean_(bc[:-1],thist[0])

        #calculate variance
        fvar = var_(bc[:-1],thist[0])

        #calculate kurtosis
        fkurt = kurt_(temp, bc[:-1],thist[0])
        
        #calculate skew
        fskew = skew_(temp, bc[:-1],thist[0])

        #calculate quartile
        fquartile = fhist[i][3]-fhist[i][1]

        fhist[i][4] = fquartile
        fhist[i][5] = fmean
        fhist[i][6] = fvar
        fhist[i][7] = fkurt
        fhist[i][8] = fskew
        temp = 'Name: {} - Index: {}'.format(name[-1], i)
        #if i==5:
        #    viz.bar(X=thist[0],Y=bc[:-1],opts=dict(title=temp))

        ##Visualization updates:

        #viz.line(X=np.ones((1))*i,Y=[perfskew],win=skew_kurt,name='s',update='append')
        #viz.line(X=np.ones((1))*i,Y=[perfkurt],win=skew_kurt,name='k',update='append')


        #percentile updates
        #viz.line(X=np.ones((1))*i,Y=[ahist[i][0]],win=percentiles,name='o',update='append')
        #viz.line(X=np.ones((1))*i,Y=[ahist[i][1]],win=percentiles,name='t',update='append')
        #viz.line(X=np.ones((1))*i,Y=[ahist[i][2]],win=percentiles,name='f',update='append')
        #viz.line(X=np.ones((1))*i,Y=[ahist[i][3]],win=percentiles,name='s',update='append')

        #mean_mid updates
        #viz.line(X=np.ones((1))*i,Y=[amid],win=mean_mid,name='med',update='append')
        #viz.line(X=np.ones((1))*i,Y=[amean],win=mean_mid,name='mean',update='append')

        
        
    return ahist, bhist, dhist, lahist, lbhist, ldhist, diffhist, perfhist, fhist, ldiffhist, lperfhist, lfhist






def binwidth(maps):

    #use Scott's bin width algorithm: h = 3.49 * std * n^(-1/3), already in numpy
    tmaps = maps.reshape(-1)
    fmap = tmaps[tmaps!=0]
    #width = np.histogram_bin_edges(fmap, bins='fd', range=(fmap.min(), fmap.max()))

    width = np.arange(fmap.min(),fmap.max(),0.00005)

    return width

def saveFeatures(file_path, name, afiles, bfiles, dfiles, ahist, bhist, dhist, lahist, lbhist, ldhist):
    #order of features: padded map, label, patientname, histogram features, original histogram
    for  i, _ in enumerate(afiles):
        
        patientname = afiles[i][2]
        label = afiles[i][1]
        af = file_path + '/%s_%s'%(name[0],patientname)
        np.save(af,[afiles[i][0], label, patientname, ahist[i],lahist[i]])
        af = file_path + '/%s_%s'%(name[1],patientname)
        np.save(af,[bfiles[i][0], label, patientname, bhist[i],lbhist[i]])
        af = file_path + '/%s_%s'%(name[2],patientname)
        np.save(af,[dfiles[i][0], label, patientname, dhist[i],ldhist[i]])


def saveFeaturesIVIM(file_path, name, afiles, bfiles, dfiles, ahist, bhist, dhist, lahist, lbhist, ldhist):
    #order of features: padded map, label, patientname, histogram features, original histogram
    for  i, _ in enumerate(afiles):
        
        patientname = afiles[i][2]
        label = afiles[i][1]
        af = file_path + '/%s_%s'%(name[0],patientname)
        np.save(af,[afiles[i][0], label, patientname, ahist[i],lahist[i]])
        af = file_path + '/%s_%s'%(name[1],patientname)
        np.save(af,[bfiles[i][0], label, patientname, bhist[i],lbhist[i]])
        af = file_path + '/%s_%s'%(name[2],patientname)
        np.save(af,[dfiles[i][0], label, patientname, dhist[i],ldhist[i]])

#viz = Visdom()

def getMaps(allfiles):

    allmaps = []
    for afile in allfiles:
        allmaps.append(np.array([temp.reshape(-1) for temp in np.array(afile)[:,0]]))
    
    return allmaps


def getFeatures(getv2, binwidth, file_path, name, ofile_path, onames):

    #Get all the maps
    print ("Extracting features of %s"%(name[0]))
    if getv2:
        afiles, bfiles, dfiles, diff_files, perf_files, f_files = getFiles2(file_path,name)
    else:
        afiles, bfiles, dfiles, diff_files, perf_files, f_files = getFiles(file_path,name)
    amaps, bmaps, dmaps, diffmaps, perfmaps, fmaps = getMaps([afiles, bfiles, dfiles, diff_files, perf_files, f_files])
    print ("get histograms")

    ahist, bhist, dhist, lahist, lbhist, ldhist, diff_hist, perf_hist, f_hist, ldiff_hist, lperf_hist, lfhist = getHistFeatures(name,amaps, bmaps, dmaps, diffmaps, perfmaps, fmaps, binwidth)

    print ("obtained histograms")
    print ("saving A,B,DDC Features")
    saveFeatures(ofile_path, onames[0:3], afiles, bfiles, dfiles, ahist, bhist, dhist, lahist, lbhist, ldhist)
    print ("saving diff,perf,F Features")
    saveFeaturesIVIM(ofile_path, onames[3:], diff_files, perf_files, f_files, diff_hist, perf_hist, f_hist, ldiff_hist, lperf_hist, lfhist)
    #plt.show()

def runScript():

    #histogram bin width
    awidth = 80
    bwidth = 80
    dwidth = 80
    diff_width = 80
    perf_width = 80
    f_width = 80

    binwidth = [awidth,bwidth,dwidth, diff_width, perf_width, f_width]

    #Generate and Save Features of Orignal Maps *********************************************** #

    file_path = 'mmasks'
    #ofile_path = ['maxminFeat', 'maxminFeatIvim']
    ofile_path = 'maxminFeat'
    name = ['apad','bpad','dpad','diffpad', 'perfpad', 'fpad']
    oname = ['apad_feat','bpad_feat','dpad_feat', 'diffpad_feat', 'perfpad_feat', 'fpad_feat']

    getFeatures(False, binwidth,file_path, name, ofile_path, oname)
    print ("Saved Features of Original Parameter Maps")

    #Generate and Save Features of Augmented Orignal Maps (non-cropped) *********************************************** #

    file_path = 'maxminAug'
    name = ['ogaug_alpha_','ogaug_beta_','ogaug_ddc_', 'ogaug_diff', 'ogaug_perf', 'ogaug_f']
    ofile_path = 'maxminFeat'
    oname = ['ogaug_alpha_feat','ogaug_beta_feat','ogaug_ddc_feat', 'ogaug_diff_feat', 'ogaug_perf_feat', 'ogaug_f_feat']

    getFeatures(True, binwidth,file_path, name, ofile_path, oname)
    print ("Saved Features of Original Augmented Parameter Maps")

    #Generate and Save Features of Augmented Crop Images 1 ********************************************** #

    file_path = 'maxminAug'
    name = ['cropaug1_alpha_','cropaug1_beta_','cropaug1_ddc_', 'cropaug1_diff', 'cropaug1_perf', 'cropaug1_f']
    ofile_path = 'maxminFeat'
    #ofile_path = ['maxminFeat', 'maxminFeatIvim']
    oname = ['cropaug1_alpha_feat','cropaug1_beta_feat','cropaug1_ddc_feat', 'cropaug1_diff_feat', 'cropaug1_perf_feat', 'cropaug1_f_feat']
    getFeatures(True, binwidth,file_path, name, ofile_path, oname)
    print ("Saved Features of Cropped Set 1 Augmented Parameter Maps")



    #Generate and Save Features of Augmented Crop Images 2
    file_path = 'maxminAug'
    name = ['cropaug2_alpha_','cropaug2_beta_','cropaug2_ddc_', 'cropaug2_diff', 'cropaug2_perf', 'cropaug2_f']
    ofile_path = 'maxminFeat'
    #ofile_path = ['maxminFeat', 'maxminFeatIvim']
    oname = ['cropaug2_alpha_feat','cropaug2_beta_feat','cropaug2_ddc_feat', 'cropaug2_diff_feat', 'cropaug2_perf_feat', 'cropaug2_f_feat']
    getFeatures(True, binwidth,file_path, name, ofile_path, oname)
    print ("Saved Features of Cropped Set 2 Augmented Parameter Maps")

    #Generate and Save Features of Augmented Crop Images 3

    file_path = 'maxminAug'
    name = ['cropaug3_alpha_','cropaug3_beta_','cropaug3_ddc_', 'cropaug3_diff', 'cropaug3_perf', 'cropaug3_f']
    #ofile_path = ['maxminFeat', 'maxminFeatIvim']
    ofile_path = 'maxminFeat'
    oname = ['cropaug3_alpha_feat','cropaug3_beta_feat','cropaug3_ddc_feat', 'cropaug3_diff_feat', 'cropaug3_perf_feat', 'cropaug3_f_feat']
    getFeatures(True, binwidth,file_path, name, ofile_path, oname)
    print ("Saved Features of Cropped Set 3 Augmented Parameter Maps")

###Notes:

runScript()

print("Finished Features")

'''The nth percentile was the point at which n% of the voxel 
values that form the histogram were found to the left (13). 
The quartile means the difference between ADC25 and ADC75.'''

    
 #________________________K1 Band flux Measure______________________________

#----------------------------------------------------------------------------
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as col
import time
import datetime as dt 
import math as m
import astropy.io.fits as pf
import glob
import os
from scipy import ndimage
from scipy import optimize
import imageio # used to create gif.
from itertools import zip_longest
import multiprocessing as mp
import time 
import numpy.fft as fft

#----------------------------------------------------------------------------
#________________________Path and directories and global parameters_____________

#dir_path1 = 'C:/Users/varga_000/Documents/GPI/gpi_pipeline_1.3.0_source/gpi_pipeline_1.3.0_r3839M_source/data/Reduced/130212/S20130212S0'
#dir_path1 = 'C:/Users/varga_000/Documents/GPI/gpi_pipeline_1.3.0_source/gpi_pipeline_1.3.0_r3839M_source/data/Reduced/k1/S20130212S0'
dir_path1 = "saved/K1band/S20130212S0"
#Hband_path = "C:/Users/varga_000/Documents/GPI/gpi_pipeline_1.3.0_source/gpi_pipeline_1.3.0_r3839M_source/data/Reduced/130211/S20130211S0"
hband_path = "saved/Hband/H_reduced_data/Hband/S20130211S0"


def multiple_files(path):
    """ creates a list of reduce data path.
    Args:
        path - is the path where the reduce files are located + their common name
            ex) "C:/Python34/GPIcode/saved/Hband/H_reduced_data/Hband/S20130211S0"
            loaction = C:/Python34/GPIcode/saved/Hband/H_reduced_data/Hband/
            common name = S20130211S0
    Return: 
        list of path where reduced are located.
    """

    fyles = sorted(glob.glob(path + '*.fits'))
    if not fyles:
        print('LIST OF PATH IS EMPTY: ', fyles)
        print('please specify full path + common file name')
        print('ex) "C:/Python34/GPIcode/saved/Hband/H_reduced_data/Hband/S20130211S0" ')
        print(fyles)

    return fyles


    #====================== EXTRACT Header and Image ===========================

""" purpose: the functions in this section are used to read in the files and
           Extract the important iformation"""

#===========================================================================

def open_txt(path):
    # i will use to this to open the approximate center files for dm/sat/star
    # skips forst row.
    return np.loadtxt(path, skiprows=1, dtype=float, delimiter=',')


def file_number(fyle): # enumerates the files for any band SUB FUCNTION
    #fyle = multiple_files(path)[index]
    trash1 = os.path.split(fyle)
    #trash1 = fyle.split("/",4)

    #print(trash1)
    fyle_number = trash1[1].replace(".fits",'')
    #print(fyle_number)

    return fyle_number
   

def open_img(file_location): #opens the matrix containing image
    hdu = pf.open(file_location)
    hdr = hdu[0].header
    #print(hdr)
    img = hdu[0].data
    return img


def open_hdr(file_location):
    hdu = pf.open(file_location)
    hdr = hdu[0].header
    return hdr


#--------------------------------------------------------------------------

#====================== replacing long function hear will organize later ===========================

""" NEW SECTION NEED TITLE """

#===========================================================================


def get_info1(path,index): # path is the raw data location for a particular fits file w/o coronograph
    fyle = multiple_files(path)[index]
    print("file name " , fyle)
    num_file = file_number(fyle)

    hdulist = pf.open(fyle)
    img = hdulist[1].data
    hdr = hdulist[1].header
    img = np.ma.array(img,mask = np.isnan(img))
    return hdr,img,num_file


def image(path, index, slyce): # just displays image 

    data = get_info1(path, index)
    print(np.shape(data))
    img_slice = data[1][slyce]
    #masked = np.ma.array(img_slice,mask =np.isnan(img_slice))
    title = data[2]

    plt.figure(1)
    img = plt.imshow(img_slice, origin='lower', interpolation='nearest', cmap='PuRd', vmin=0, vmax=500)
    plt.colorbar()
    plt.title('Raw file number: ' + str(title) + ', slice : ' + str(slyce))
    plt.xlabel('X [pixel]')
    plt.ylabel('Y [pixel]')
    return plt.show()






# ======================================= required flux fucntions =====================

def radiu(centrod_x,centrod_y):
    # the 281X281 is is the shape of the image
    y,x = np.indices((281,281))
    r = np.sqrt((x-centrod_x)**2 + (y-centrod_y)**2)
    return [r,x,y]


def anu(radii,img,object_radi=10,anu_radi=25):
    # "radii" comes from get_info1
    mask = img
    mask = np.ma.masked_where(radii>object_radi,img)
        
    sky = img
    sky = np.ma.masked_where(radii>anu_radi,sky)
    #sky = np.ma.masked_where(radii<=object_radi,sky)
    sky = np.ma.masked_where(radii<=10,sky)
    
    skyval = np.nanmean(sky)
    #print(skyval,'mean')
    
    return [skyval,sky,mask]


def centrod(band_path,approx_center_path,index,slyce,xy_index,center_x=False, center_y=False, object_radi=10,anu_radi=25):
    if center_x == False:
        center_x, center_y = open_txt(approx_center_path)[xy_index,1:]
        #print(center_x,center_y)
    #print(center_x, 'this is x')
    #print( center_y, ' this is y')
    
    data = radiu(center_x,center_y)
    img = get_info1(band_path,index)[1][slyce]
         
    radii, xx, yy = data[0], data[1], data[2]
    
    anu_data = anu(radii,img,object_radi,anu_radi)
    skyval, mask = anu_data[0], anu_data[2]
    #print(skyval,'skyval')
    xbar = np.nansum(mask*xx*(img-skyval))/np.sum(mask*(img-skyval))
    ybar = np.nansum(mask*yy*(img-skyval))/np.sum(mask*(img-skyval))

    return [xbar,ybar,img,skyval]

def fluxi(band_path,approx_center_path,index,slyce,xy_index,center_x=False, center_y=False, object_radi=10,anu_radi=35):
    
    centro = centrod(band_path,approx_center_path,index,slyce,xy_index,center_x,center_y,object_radi,anu_radi)
    radii = radiu(centro[0],centro[1])[0]
    #print(radii)
    img = centro[2]
    skyval = centro[3]
    #print(skyval)
    
    aperture = np.where(radii <= object_radi)
    #skysub = img[aperture] - skyval
    skysub = img[aperture]
    flux = np.sum(skysub)

    return flux

def flux_range(band_path,approx_center_path,index,slyce,xy_index,object_radi,anu_radi=15,center_x=False, center_y=False):
    radi_range = np.arange(.05,object_radi,.5)
    #print(radi_range)
    flux_range = np.array([])
    for i in radi_range:
        #print(i)
        flux = fluxi(band_path,approx_center_path,index,slyce,xy_index,center_x,center_y,i,anu_radi)
        flux_range = np.append(flux_range,flux)
        
    return radi_range,flux_range


def display_flux_range(band_path,approx_center_path,index,slyce,xy_index,object_radi,anu_radi=15,center_x=False, center_y=False):
    name = str(input('input name of object: '))
    print(name)
    #print(foo)
    data = flux_range(band_path,approx_center_path,index,slyce,xy_index,object_radi,anu_radi,center_x, center_y)
    radius_array = data[0]
    flux_array = data[1]
    plt. plot(radius_array,flux_array,'o-')
    plt.title(name,fontsize=20)
    plt.xlabel('radius',fontsize=18)
    plt.ylabel('flux',fontsize=18)
    plt.grid(True)
    plt.show()



# ======================================================================================



def fluxi_range(band_path,approx_center_path,index,slyce,xy_index,r_stop,anu_radi = 35):
    radi_range = np.arange(.05,r_stop,.5)
    flux_range = np.array([])
    center_x, center_y = open_txt(approx_center_path)[xy_index,1:]
    #print(center_x, 'this is x')
    #print( center_y, ' this is y')
    
    for i in radi_range:
        flux = fluxi(band_path,approx_center_path,index,slyce,xy_index,i,anu_radi)
        flux_range = np.append(flux_range,flux)
    return radi_range,flux_range

"""def flux_range(filenumber,slice_number,r_end):
    data = radius(filenumber,slice_number)
    img = data[0]
    radii = data[1]
    skyval = anulus(filenumber,slice_number)[0]

    r =.00001
    fluxi = np.array([])
    radiius =np.array([])
    skysub = img - skyval
    while r <= r_end:
        aperture = np.where(radii <= r)
        #print(aperture)
        flux = np.sum(skysub[aperture])
        #print(flux)
        fluxi = np.append(fluxi,flux)
        #print(fluxi)
        radiius = np.append(radiius,r)
        r +=1
    
    return [radiius,fluxi]"""

#radiu(centrod_x,centrod_y)
#radiu returns return [r,x,y]
#get_info1(path,index)
#return hdr,img,num_file

def masket(band_path,approx_center_path,index,slyce,xy_index,object_radi=10,anu_radi=20):

    center_x, center_y = open_txt(approx_center_path)[xy_index,1:]
    radii = radiu(center_x,center_y)[0]
    sky_radi = radii
    o_radi = radii

    img_raw = get_info1(band_path,index)[1][slyce]
    
    mask = img_raw
    mask = np.ma.masked_where(o_radi>object_radi,mask)
        
    sky = img_raw
    sky = np.ma.masked_where(sky_radi>anu_radi,sky)
    sky = np.ma.masked_where(sky_radi<=object_radi,sky)
    return [sky,mask]

def display_anul(band_path,approx_center_path,index,slyce,xy_index,object_radi=10,anu_radi=20):
    data = masket(band_path,approx_center_path,index,slyce,xy_index,object_radi,anu_radi)
    sky = data[0]

    plt.figure(1)
    plt.imshow(sky,origin='lower',interpolation='nearest',cmap='gnuplot',vmin=0,vmax=500)
    plt.colorbar()
    plt.title('Anulus')
    plt.xlabel('x [pixel]')
    plt.ylabel('y [pixel]')
    return plt.show()
 
def display_masket(band_path,approx_center_path,index,slyce,xy_index,object_radi=10,anu_radi=20):
    data = masket(band_path,approx_center_path,index,slyce,xy_index,object_radi,anu_radi)
    sky = data[1]
    
    plt.figure(1)
    plt.imshow(sky,origin='lower',interpolation='nearest',cmap='gnuplot',vmin=0,vmax=500)
    plt.colorbar()
    plt.title('Anulus')
    plt.xlabel('x [pixel]')
    plt.ylabel('y [pixel]')
    return plt.show()
# ========================================================================================
#def fluxi(band_path,approx_center_path,index,slyce,xy_index,object_radi=10,anu_radi=35):
#return flux


def auto_flux(band_path,approx_center_path,index,slyce,xy_index,end_file,center_x=False, center_y=False, object_radi=10,anu_radi=35):

    """ this only finds the flux of one slice at a time. and appends it to to an empty array 
    therefore creating an array of flux for different slices """

    start_file = index #0 for coronagraph 15 for no coronagraph this number is the itteration of file in the FLUXI FUNCTION!!.
    # NOTE THAT SLICE AND RADI ARE CONSTANT FOR FILE ITERATION
    flux_arr = np.array([])
    while start_file <end_file: # 34 for no coronagraph 15 for coronagraph itteration of slice (wavelength)
        #flux = fluxi(start_file,slice_number,radi)
        flux = fluxi(band_path,approx_center_path,start_file,slyce,xy_index,center_x,center_y,object_radi,anu_radi)
        flux_arr = np.append(flux,flux_arr)
        #print('next file...' + str(start_file))
        start_file +=1
    return flux_arr

def auto_slice1(band_path,approx_center_path,start_file,end_file,xy_index,center_x=False, center_y=False, object_radi=10,anu_radi=20):
    slyce = 3 # starting slice, ie well difined sat spot!
    flux = []

    while slyce < 37:
       
        data = auto_flux(band_path,approx_center_path,start_file,slyce,xy_index,end_file,center_x,center_y,object_radi,anu_radi)
        flux.append(data)
        print('Next slice... '+ str(slyce))
        centro = centrod(band_path,approx_center_path, start_file + 5,slyce,xy_index,center_x, center_y)
        print(start_file + 5,'file number')
        center_x = centro[0]
        center_y = centro[1]
        print(center_x,center_y,'center')
        #if abs(xi-center_x)> 6:
        #    print("warning check x value")
        #if abs(yi-center_y)> 6:
        #    print("warning check y value")

        #print(slyce,'slice')
        #display_testmask(band_path,center_x,center_y,5,slyce,object_radi,anu_radi)

        slyce+=1
    print(np.shape(flux))
    
    flux = np.vstack((flux))
    print(flux)
    return flux



def auto_save(band_path,approx_center_path,start_file,end_file,cent_start,cent_end,object_name,band,object_radi=10,anu_radi=35,save=True):
    while cent_start <= cent_end:
        name = object_name + str(cent_start)
        center_x=False
        center_y=False
        #print(cent_start,'tadaa')
        print('saving '+ name + '...')
        flux = auto_slice1(band_path,approx_center_path,start_file,end_file,cent_start,center_x, center_y,object_radi,anu_radi)
        #print(flux)
        save_txt(name, band, flux, save)
        cent_start +=1

#def centrod(band_path,approx_center_path,index,slyce,xy_index,object_radi=10,anu_radi=25):
 #   return [xbar,ybar,img,skyval]


def save_txt(object_name,band,flux,save=False):
    if save == True:
        new_file = str(object_name) +'_'+str(band)+'_flux.txt'
        if not os.path.exists(new_file):
            np.savetxt(new_file,flux,delimiter=',')


def new_folder(object_name,band,image=False):
    if image == True:
        newpath = str(object_name)+'_'+str(band) + '_image'
        if not os.path.exists(newpath):
            os.makedirs(newpath)







def multiple_txt(path):
    # returns a list of files path. paths correpond to the path of Band data
    fyles = sorted(glob.glob(path + '*.txt'))
    #print ( 'number of fits files: ', str(len(fyles)))
    return fyles



def ratio_mean_dmsat(sat_path,dm_path,start,end):
    # "sat_path" and "dm_path"is the sub path to 
    # example: star_path = "C:/Python34/GPIcode/aperture_calibration/star"
    # "start" and "end" is the slice range
    sat_paths = multiple_txt(sat_path)
    #print(np.shape(sat_paths))
    dm_paths = multiple_txt(dm_path)
    #print(dm_paths)
    ratio = np.array([])

    for j in range(start,end): # slice 
        temp_dm = []
        temp_sat = []

        for i in range(0,4):
            #print(i)
            #print(dm_paths[i])
            dmi = open_txt(dm_paths[i])[j]
            sati = open_txt(sat_paths[i])[j]
            temp_dm.append(dmi)
            temp_sat.append(sati)
        temp_dm = np.array(temp_dm)
        tem_sat = np.array (temp_sat)
        mean_dm = np.mean(temp_dm, axis = 0)
        mean_sat = np.mean(temp_sat,axis = 0)
        div = mean_dm /mean_sat
        ratio = np.append(ratio,div)
        #print(ratio)
        #print(np.size(ratio))



       
    # std, mean , and error for dm 
    std_dm = np.std(temp_dm)
    mein_dm = np.mean(temp_dm)
    err_dm = std_dm**2/mein_dm**2
    print(std_dm,'stdv_dm')
    print (mein_dm, 'mean_dm')
    print(err_dm, 'error')


    # std, mean , and error for sat
    std_sat = np.std(temp_sat)
    mein_sat = np.mean(temp_sat)
    err_sat = std_sat**2/mein_sat**2
    print(std_sat,'stdv_sat')
    print (mein_sat, 'mean_sat')
    print(err_sat, 'error')


    #std, mean and error for dm and sat
    stds = np.std(ratio)
    meany = np.mean(ratio)
    error = stds/meany

    print(stds,'std_ratio')
    print(meany,'mean_ratio')
    print(error,'error_ratio')
    return meany , err_dm , err_sat

def ratio_mean_dmstar(star_path,dm_path,start,end):
    star_paths = multiple_txt(star_path)
    print(star_paths)
    
    dm_paths = multiple_txt(dm_path)
    ratio = np.array([])
    #datai = no_coronagraph_fluxes()
    #ratio = np.array([])
    
    for j in range(start,end):
        temp_dm = []
        temp_star = []

        for i in range(0,4):

            dmi = open_txt(dm_paths[i])[j]
            stari = open_txt(star_paths[0])[j]
            temp_dm.append(dmi)
            temp_star.append(stari)
        temp_dm = np.array(temp_dm)
        temp_star = np.array(temp_star)
        mean_dm = np.mean(temp_dm,axis=0)
        mean_star = np.mean(temp_star,axis=0)
        div = mean_dm/mean_star
        ratio = np.append(ratio,div)
        
    # mean and stdv for dm spots
    std_dm = np.std(temp_dm)
    mein_dm = np.mean(temp_dm)
    err_dm = std_dm**2/mein_dm**2
    print(std_dm,'stdv_dm')
    print (mein_dm, 'mean_dm')
    print(err_dm, 'error_dm')
    
    # mean and stdv for star
    std_star = np.std(temp_star)
    mein_star = np.mean(temp_star)
    err_star = std_star**2/mein_star**2
    
    print(std_star,'stdv_star')
    print(mein_star,'mean_star')
    print(err_star,'error_star')

    # mean and stdv of ratio
    stds = np.std(ratio)
    meany = np.mean(ratio)
    print(stds,'std_ratio')
    print(meany,'mean_ratio')
    print(stds/meany,'error_ratio')

    
    
    return meany, err_dm, err_star







def testmask(band_path,center_x,center_y,index,slyce,object_radi,anu_radi):
#auto_slice1(band_path,approx_center_path,start_file,end_file,xy_index,center_x=False, center_y=False, object_radi=10,anu_radi=20):
    radii = radiu(center_x,center_y)[0]
    img_raw = get_info1(band_path,index)[1][slyce]
    mask = img_raw
    mask = np.ma.masked_where(radii>object_radi,mask)

    sky = img_raw
    sky = np.ma.masked_where(radii>anu_radi,sky)
    sky = np.ma.masked_where(radii<=object_radi,sky)

    return [sky,mask]

def display_testmask(band_path,center_x,center_y,index,slyce,object_radi,anu_radi):
    data = testmask(band_path,center_x,center_y,index,slyce,object_radi,anu_radi)
    sky = data[0]
    mask = data[1]

    
    plt.figure(1)
    plt.imshow(sky,origin='lower',interpolation='nearest',cmap='gnuplot',vmin=0,vmax=500)
    plt.colorbar()
    plt.title('Anulus')
    plt.xlabel('x [pixel]')
    plt.ylabel('y [pixel]')

    plt.figure(2)
    plt.imshow(mask,origin='lower',interpolation='nearest',cmap='gnuplot',vmin=0,vmax=500)
    plt.colorbar()
    plt.title('Mask')
    plt.xlabel('x [pixel]')
    plt.ylabel('y [pixel]')


    return plt.show()




def all_files_same_slice(band_path,slyce):
    for i in range(0,33):
        print(i)
        img = get_info1(band_path,i)[1][slyce]
        plt.figure(i)
        plt.imshow(img, cmap='Blues',vmin=-48,vmax=90)
        plt.title('file number: ' +str(i), fontsize=20)
        plt.colorbar()

    plt.tight_layout()
    return plt.show()

dm2 =   "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/2_dm"
dm3 =   "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/3_dm"
dm4 =   "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/4_dm"
dm5 =   "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/5_dm"
dm6 =   "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/6_dm"
dm7 =   "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/7_dm"
dm8 =   "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/8_dm" 


sat2 = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/2_sat"
sat3 = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/3_sat"
sat4 = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/4_sat"
sat5 = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/5_sat"
sat6 = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/6_sat"
sat7 = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/7_sat"
sat8 = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/coronagraph_files/8_sat"


nc_2star = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/No_coronograph_files/2_star"
nc_3star = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/No_coronograph_files/3_star"
nc_4star = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/No_coronograph_files/4_star"

nc_2dm   = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/No_coronograph_files/2_dm"
nc_3dm   = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/No_coronograph_files/3_dm"
nc_4dm   = "C:/Python34/GPIcode/aperture_calibration/h_band/no_skyval_sub/No_coronograph_files/4_dm"



def pixel_map(image,x,y):

    image[np.isnan(image)] = np.nanmedian(image)
    return ndimage.map_coordinates(image, (y,x),cval=np.nan)


def gen_xy(size):
# contains the index of row and coloumn     
    s = np.array([size,size])
    x,y = np.meshgrid(np.arange(s[1]),np.arange(s[0]))
    return x,y

def twoD_Gaussian(xy, amplitude, xo, yo, sigma, offset):
    x,y = xy
    xo = float(xo)
    yo = float(yo)    
    a = 1.0 / (2.0 * sigma**2.0)
    b = 1.0 / (2.0 * sigma**2.0)
    g = offset + amplitude*np.exp( - ((a*((x-xo)**2)) + (b*((y-yo)**2))))
    
    return g.ravel()


def return_pos(im, xy_guess,x,y):

    #Fit WD location in slice from guess
   # pad = 10
    #stamp = im[xy_guess[1]-pad:xy_guess[1]+pad, xy_guess[0]-pad:xy_guess[0]+pad]
    #stamp[np.isnan(stamp)] = np.nanmedian(stamp)
    p0 = [np.nanmax(im), xy_guess[0], xy_guess[1], 3.0, 0.0]
    '''
    p0 -guess for the 5 parameter of 2d gaussian
    np.nanmax(stamp) - peak value in the variable stamp
    pad - xc,yc from pixel_cutout size/2 center of the image where the cal or sat spot should be
    3.0 - the full width half max in pixel (width of gaussian)
    0.0 - constant value to shift 
    '''

    #s = np.shape(stamp)
   # x, y = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
    """ x,y - is the same coordinate as pixel cutout after adding xc and yc. this is to keep it relative
        to the size of original image
    """
    popt, pcov = optimize.curve_fit(twoD_Gaussian, (x, y), im.ravel(), p0 = p0,maxfev=15000000)
    #print(popt)
    # xy - pad : is a constant to convert the coordinate back to the original image coordinate frame,
    # in my code i already added the contant term pad --> xc, yc therefore no need to add pad
    # for my code xy = [popt[1],popt[2]]
    #xy = [popt[1] + (xy_guess[0] - pad), popt[2] + (xy_guess[1] - pad)]    

    return popt[1],  popt[2]


def pixel_cutout(image,size,xguess, yguess):#, name2='none',save = False):
    """ combines the above functions in this section. Used to create box cutout centered at 
        the specified spot.
    Args:
        image - a slice of the original data cube
        size  - the size of the sides of the box cutout
        xguess - initial x coordinate guess to center of spot
        yguess - initial y coordinate guess to center of spot
        name1,2 - when save set to True, this will be the name of the file
        save   - option to save image cutout with initial guess and after
                    center has been optimized 
    Return:
        output - box cutout of spot with optimized center 
    """
    size = float(size)
    xguess = float(xguess)
    yguess = float(yguess)
    x,y = gen_xy(20.0)
    x += (xguess-20/2.)
    y += (yguess-20/2.)
   
    output = pixel_map(image,x,y)
    xc,yc = return_pos(output, (xguess,yguess), x,y)
    
    #if save == True:
        # image before center optimization
        #write = pf.writeto(name1, output,clobber = True)

    x,y = gen_xy(size+4)
    x += (xc-np.round((size+4)/2.))
    y += (yc-np.round((size+4)/2.))
    output = pixel_map(image,x,y)


    r = np.sqrt((x-xc)**2 + (y-yc)**2)
    output[np.where(r>size/2)] = np.nan
    #output = np.nan_to_num(output)



    #if save == True:
        # image after center optimization
        #write = pf.writeto(name2, output,clobber = True)

    return output



def loop_pixcut(image,size,centdm,centsat,fname,index=0,save=False,scale=[],residuals=[]):

    dm_img, sat_img = [], []
    spotnum = 0


#=====================plotting procedure============================
    if save == True:
        fig = plt.figure(figsize=(10,10))
        fig.suptitle(fname+'               slice='+str(index),fontsize=25)
        fig.subplots_adjust(wspace=0.2, hspace=0.3)

    if np.shape(centsat) == (1,2):
        spotname = "STAR"
    else:
        spotname = "SAT spot"
# ==================================================================

    
    for i,j in zip_longest(centdm,centsat):

        cutdm = pixel_cutout(image,size,i[0],i[1])
        dm_img.append(cutdm)

        if j != None:
            cutsat = pixel_cutout(image,size,j[0],j[1])
            sat_img.append(cutsat)
        
        #==================plotting procedure==================
        if save == True:

            dm = plt.subplot(4,3,1+spotnum)
            dm.imshow(cutdm,interpolation='nearest',cmap='gnuplot2')
            dm.set_title('DM spot')

            if j != None:
                sat = plt.subplot(4,3,2+spotnum)
                sat.imshow(cutsat,interpolation='nearest',cmap='gnuplot2')
                sat.set_title(spotname)
            spotnum +=3
            # ===================================================

    dm_img = (np.sum(dm_img,axis=0))/len(dm_img)
    sat_img = (np.sum(sat_img,axis=0))/len(sat_img)
    if np.shape(centsat) == (1,2):
        scalef = optimz(sat_img,dm_img)
        resid =(scalef*sat_img) - dm_img

    else:
        scalef = optimz(dm_img,sat_img)
        resid =(scalef*dm_img) - sat_img
    scale.append(scalef)
    residuals.append(resid)

    # ==========================plotting procedure======================
    if save == True:
        ave_dm = plt.subplot(4,3,3)
        ax3 = ave_dm.imshow(dm_img,interpolation='nearest',cmap='gnuplot2')
        fig.colorbar(ax3)
        ave_dm.set_title('Average DM')

        ave_sat = plt.subplot(4,3,6)
        ax2 = ave_sat.imshow(sat_img,interpolation='nearest',cmap='gnuplot2')
        fig.colorbar(ax2)
        ave_sat.set_title('Average '+ spotname)

        rsd = plt.subplot(4,3,9)
        ax = rsd.imshow(resid/np.nanmax(sat_img),interpolation='nearest',cmap='bwr',vmin=-0.1, vmax= 0.1)
        fig.colorbar(ax)
        rsd.set_title("Residuals")

        scale_plot = plt.subplot(4,3,12)
        scale_plot.plot(scale, marker='.') #Line with small markers
        scale_plot.plot(index, scale[index], marker='o', markersize=10)
        plt.title("scale factor",fontsize=16)
        plt.savefig("psf_fitting/pngs/atest/"+fname+'_size'+str(size)+'_'+'slice'+str(index).zfill(2)+'.png', bbox_inches='tight')
    plt.close('all')
    # ==================================================================


    return dm_img,sat_img,scale,residuals





def slice_loop(path,fnum,size,center_dm,center_sat,save = False,filtersize=10):
    
    """ loops over all slices and creates an average image for the specified spots at each 
        wavelength slice. At the end there is a total of 37 averaged images.
    Args: 
       path - is the path in where the file is located (see multiple_files function)
       fnum - used to index file from multiple file array (see multiple_file function)
       size - the size of the sides of the box cutout
       center_guess - initial guess for center of spot.
       save - option to save image cutout with initial guess and after
                center has been optimized
    Return: 
        box - data cube with 37 averaged images.
        filename - the name of the file which is being averaged.
    """
    info = get_info1(path,fnum) # fnum is the index fits file.
    image = info[1]
    filename = info[2]

    center_dm = open_img(center_dm)
    center_sat = open_img(center_sat)

    ave_dm , ave_sat , scale , residuals, chi2_residuals = [],[],[],[],[]
    imslice = 0
    #print( np.shape(image[3:20]))
    #print(t)


    #for img in image[3:20]:
    for img in image:
        img  = high_pass_filter(img,filtersize)
        centdm = center_dm[imslice]
        centsat = center_sat[imslice]
        data = loop_pixcut(img,size,centdm,centsat,filename,imslice,save,scale,residuals)

        ave_dm.append(data[0])
        ave_sat.append(data[1])
        scale = data[2]
        residual = data[3]
        chi2_resid = reduce_chi2(residual[-1] , 1)
        chi2_residuals.append(chi2_resid)

        imslice +=1

    ave_dm = np.array(ave_dm)
    ave_sat = np.array(ave_sat)
    #print(scale)
    #print(np.shape(scale))
    #np.savetxt('psf_fitting/pngs/residuals/chi2_resid_'+filename+ 'size'+ str(size)+ '.txt',chi2_residuals)

        


    #--------------------------- Creating gif and removing pngs ------------------------
    if save == True:
        time.sleep(20)
        location='psf_fitting/pngs/atest/'
        #"psf_fitting/pngs/atest/"+fname+'_size'+str(size)+'_'+'slice'+str(index).zfill(2)+'.png'
        cmd = 'convert -delay 40 -loop 0 '+location+filename+'_size'+str(size)+'*.png '+location+filename+'size'+str(size).zfill(2)+'.gif'
        os.system(cmd)
        pngs = glob.glob(location+filename+'_size'+str(size)+'*.png')
        [os.remove(png) for png in pngs]
    #------------------------------------------------------------------------------------
    
   
    return scale,filename

def reduce_chi2 (residual,numparams):
    chi2 = 1/(np.size(residual) - numparams - 1) * (np.nansum(residual**2))

    return chi2



def minimize_psf(scale, ave_dm, ave_sat): 
    """ Simply minimize residuals
    Args:
        scale - scale factor 
        ave_dm - average dm for a given slice
        ave_sat - average sat for a given slice 
    return:
        residuals for ave_dm and ave_sat

    """
    return np.nansum(np.abs(((scale*ave_dm) - ave_sat)))


def optimz(ave_dm, ave_sat):

    guess =  np.nanmax(ave_sat) / np.nanmax(ave_dm)
    result = optimize.minimize(minimize_psf, guess, args=(ave_dm, ave_sat), method = 'Nelder-Mead') 
    scale = result.x[0]
    #if result.success == False: 
    #    print( 'this is a warning')
    #    print(result)       
    #scl.append(scale)
    return scale

def open_img(file_location): #opens the matrix containing image
    hdu = pf.open(file_location)
    hdr = hdu[0].header
    #print(hdr)
    img = hdu[0].data
    return img

#def slice_loop(path,fnum,size,center_dm,center_sat,save = False,sname='sat'):

def main_loopit(size,fstart,fstop,save=False,filtersize=10):
    path1 = "saved/Hband/H_reduced_data/Hband/S20130211S0"
    #path1 = "C:/Users/varga_000/Documents/GPI/gpi_pipeline_1.3.0_source/gpi_pipeline_1.3.0_r3839M_source/data/Reduced/k1/S20130212S0"
    #path1 = dir_path1 # K1 BAND
    # HBAND coronograph files 403 - 436, ie 0-34 , No coronograph 440 - 455 , i.e 34-50
    # k1BAND coronagraph file      ie. 0-15 , No coronograph      i.e 15- 33
    #fnum = 34
    scales = []
    #size = 20
    center_dm = "psf_fitting/guess_center/Hdm_center.fits"
    #center_dm = "psf_fitting/guess_center/dmk1_center.fits"
    if fstart == 0:
        spot = "psf_fitting/guess_center/Hsat_center.fits" # SAT CENTER
        #spot = "psf_fitting/guess_center/satk1_center.fits"
    else:
        spot = 'psf_fitting/guess_center/star_center.fits' #STAR center
        #spot = "psf_fitting/guess_center/stark1_center.fits"
    nthreads = mp.cpu_count()
    if nthreads > 16:
    	nthreads = 50
    pool = mp.Pool(nthreads)
    result = [pool.apply_async(slice_loop, args=(path1,fnum,size,center_dm,spot,save,filtersize)) for fnum in range (fstart ,fstop)]
    output = [p.get() for p in result]
    for fnum in range(0,fstop-fstart):
    	scales.append(output[fnum][0])
    pool.close()
    pool.join()

    #scale_path1 = "psf_fitting/scale_factors/"
    #save = False 
    """for fnum in range(fstart,fstop):
        scale = slice_loop(path1,fnum,size,center_dm,spot,save,filtersize)
        print(np.shape(scale[0]))
        scales.append(scale[0])
        print(scale)

        #write = pf.writeto(scale_path+scale[1].zfill(2)+'.fits', np.array(scale[0]),clobber = True)
        #write = np.savetxt(scale_path+str(scale[1]).zfill(2)+'.txt',np.array(scale[0]))
        #print(t)"""

    return scales

def main_loopf():
    svpath = "psf_fitting/scale_factors/scale_r/"
    size = 8
    filtersize = 3
    save = False
    finalmean, finalstd, bsize= [],[],[]

    while filtersize <= 20:
        print('filtersize', filtersize, ' check that its looping over all filtersizes here')
        size = 8
        while size <= 8:
            fname = svpath + 'hscaleplot_'+ 'filtersize'+str(filtersize)+'bsize'+str(size).zfill(2)+'.png'
            cname = svpath+"hc_size"+'_filtersize'+str(filtersize)+'_'+str(size).zfill(2)+'.txt'
            ncname = svpath+"hnc_size"+'_filtersize'+str(filtersize)+'_'+str(size).zfill(2)+'.txt'
            #print(ncname)
            cscale = main_loopit(size,0,34,save,filtersize)
            ncscale = main_loopit(size,34,50,save,filtersize)
    
            np.savetxt(cname,cscale)
            np.savetxt(ncname,ncscale)
            mean,std = scale_plot(cname,ncname,fname,firstslice=0,lastslice=37)
            finalmean.append(mean)
            finalstd.append(std)
            bsize.append(size)

            size +=1
        filtersize += 1
    plt.errorbar(bsize,finalmean , yerr=finalstd)
    plt.xlim(min(bsize)-1, max(bsize)+1)
    #plt.show()


def scale_plot(cname,ncname,fname,firstslice=0,lastslice=36):
    #cscale = np.loadtxt("C:/Python34/GPIcode/psf_fitting/scale_factors/scale_r/Hc_size07.txt")[:,10:34]
    #cscale = np.loadtxt("hbandtest.txt")[:,3:34]

    cscale = np.loadtxt(cname)[:,firstslice:lastslice]
    print(np.shape(cscale),'cscale')
    print(lastslice,'lastslice num')
    ncscale = np.loadtxt(ncname)[:,firstslice:lastslice]
    
    #print(t)
    #ncscale = np.loadtxt("hbandtest2.txt")[:,3:34]
    #ncscale = np.loadtxt("C:/Python34/GPIcode/dmstark1.txt")
    #cscale = np.loadtxt("C:/Python34/GPIcode/dmsatk1.txt")
    #print(np.shape(ncscale),'star')
    #print(np.shape(cscale),'sat')

    cmean = np.nanmean(cscale,axis=0)
    #print(np.shape(cmean))
    #print(t)
    ncmean = np.nanmean(ncscale,axis=0)
    cstd = np.nanstd(cscale,axis=0)
    
    ncstd = np.nanstd(ncscale,axis=0)
    ratio = cmean*ncmean
    previous_h = np.zeros(len(cmean)) + 2.03656e-4   #Hband
    previous_k1 = np.zeros(len(cmean)) + 2.71429e-4 #k1 rob gave me this one
    #previous_k1 = np.zeros(len(cmean)) +  2.695e-4 #k1 jason gave me this one
    lower_err_h = previous_h -0.10895492e-4    #hband
    upper_err_h = previous_h +0.10895492e-4    #hband
    lower_err_k1 = previous_k1 -0.178279e-4  #k1
    upper_err_k1 = previous_k1 +0.178279e-4 # k1

    meanc = [np.random.normal(loc=meani, scale=stdi, size=int(1e6)) for meani,stdi in zip(cmean,cstd)]
    meannc = [np.random.normal(loc=meani, scale=stdi, size=int(1e6)) for meani,stdi in zip(ncmean,ncstd)]
    
    scale = np.multiply(meanc,meannc)
    print(np.shape(scale))
    scalestd = np.nanstd(scale,axis=1)
    scalemean = np.nanmean(scale,axis=1)

    #sclmean = np.nanmean(scalemean)
    #sclstd = np.nanstd(scalemean)
    #print(sclmean,sclstd)

    meansat_starrand  = [np.random.normal(loc=smi, scale=sstdi, size =int(1e6)) for smi,sstdi in zip(scalemean,scalestd)]
    print(np.shape(meansat_starrand))
    sclmen = np.nanmean(meansat_starrand)
    sclstd = np.nanstd(meansat_starrand)
    newratio = np.zeros(len(cmean)) + sclmen
    print(np.shape(sclmen))
    print(np.shape(newratio),'new ratio shape')

    #print(sclmen,'$\pm$',sclstd)



  
    div = scalemean/(scalestd**(-2))
    std = np.sqrt(1/np.sum(scalestd**(-2)))
    #print(std, 'std')
    #print(t)

    
    weightmean = ((1/np.sum(scalestd**(-2)))*(np.sum(np.divide(scalemean,scalestd**2))))
    print(scalemean,'mean')
    print(scalestd,'std')
    print(weightmean,'wmean')
    print(newratio,'new ratio')
    print(np.shape(weightmean))



    #plt.figure(1)
    #print(t)

   

    
    plt.figure()
    f, axarr = plt.subplots(3, sharex=True)
    #axarr[0].set_title('Ratio %6.2f $\pm$ %6.2f x1e-4' % (weightmean*1e4, std*1e4))
    axarr[0].set_title('Ratio %6.2f $\pm$ %6.2f x1e-4' % (sclmen*1e4, sclstd*1e4))

    axarr[0].plot(cscale[0],'b', alpha=0.2,label='sat/dm scale')
    for i in cscale:
        axarr[0].plot(i,'b',alpha=0.2)
    axarr[0].plot(cmean,'k',label='mean scale')
    axarr[0].errorbar(np.arange(len(cmean)),cmean,yerr= cstd, color='k')
    axarr[0].legend(loc='best',prop={'size':7})#='lower right')
    axarr[0].set_ylabel('SAT/DM')

    for j in ncscale:
        axarr[1].plot(j,'r',alpha=0.2)
    axarr[1].plot(ncscale[0],'r', alpha=0.2,label='dm/star scale')
    axarr[1].plot(ncmean,'k',label='mean scale')
    axarr[1].errorbar(np.arange(len(cmean)),ncmean,yerr= ncstd, color='k')
    axarr[1].set_ylabel('DM/STAR')
    axarr[1].legend(loc='best',prop={'size':7})#='upper right')

    axarr[2].errorbar(np.arange(len(cmean)),scalemean,yerr=scalestd,color='g',label="sat/star")
    axarr[2].plot(newratio,'k',label= 'new')
    axarr[2].plot(newratio - sclstd,'k--')
    axarr[2].plot(newratio + sclstd,'k--')
    axarr[2].plot(previous_h,'r',label="prev H")
    axarr[2].plot(lower_err_h,'r--')
    axarr[2].plot(upper_err_h,'r--')
    axarr[2].plot(previous_k1,'b',label="prev K1")
    axarr[2].plot(lower_err_k1,'b--')
    axarr[2].plot(upper_err_k1,'b--')
    axarr[2].legend(loc='best',prop={'size':7})#='upper right')
    plt.show()
    #plt.savefig(fname,bbox_inches='tight',dpi=300)
    plt.close('all')
    return sclmen, sclstd

    
def loop_resid_plot():
    location = "D:/GPIcode/psf_fitting/pngs/residuals/"
    ave_resid = []
    for i in range(20):
        residarray = []
        loc = sorted(glob.glob(location+ '*'+str(i) + ".txt"))
        print(loc)
        for j in loc:
            op_resid = np.loadtxt(j)
            residarray = np.append(residarray, op_resid)
        ave_resid.append(np.nanmean(residarray))
    plt.plot(ave_resid)
    return plt.show


def high_pass_filter(img, filtersize=10):
    """
    A FFT implmentation of high pass filter.

    Args:
        img: a 2D image
        filtersize: size in Fourier space of the size of the space. In image space, size=img_size/filtersize

    Returns:
        filtered: the filtered image
    """
    # mask NaNs
    nan_index = np.where(np.isnan(img))
    img[nan_index] = 0

    transform = fft.fft2(img)

    # coordinate system in FFT image
    u,v = np.meshgrid(fft.fftfreq(transform.shape[1]), fft.fftfreq(transform.shape[0]))
    # scale u,v so it has units of pixels in FFT space
    rho = np.sqrt((u*transform.shape[1])**2 + (v*transform.shape[0])**2)
    # scale rho up so that it has units of pixels in FFT space
    # rho *= transform.shape[0]
    # create the filter
    filt = 1. - np.exp(-(rho**2/filtersize**2))

    filtered = np.real(fft.ifft2(transform*filt))

    # restore NaNs
    filtered[nan_index] = np.nan
    img[nan_index] = np.nan

    return filtered

# ============================== filtering images to see whats they look like ======================

def img_filter(img_cube):
    filtered_cube = []
    for img in img_cube:
        print(np.shape(img))
        img = high_pass_filter(img, filtersize=10):
        filtered_cube.append(img)







def main():
	main_loopf()
if __name__ =='__main__':
	main()


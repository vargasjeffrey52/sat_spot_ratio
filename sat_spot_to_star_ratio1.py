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
#----------------------------------------------------------------------------
#________________________Path and directories and global parameters_____________

#dir_path1 = 'C:/Users/varga_000/Documents/GPI/gpi_pipeline_1.3.0_source/gpi_pipeline_1.3.0_r3839M_source/data/Reduced/130212/S20130212S0'
dir_path1 = 'C:/Users/varga_000/Documents/GPI/gpi_pipeline_1.3.0_source/gpi_pipeline_1.3.0_r3839M_source/data/Reduced/k1/S20130212S0'
#Hband_path = "C:/Users/varga_000/Documents/GPI/gpi_pipeline_1.3.0_source/gpi_pipeline_1.3.0_r3839M_source/data/Reduced/130211/S20130211S0"
hband_path = "saved/Hband/H_reduced_data/Hband/S20130211S0"


def multiple_files(path):
    # returns a list of files path. paths correpond to the path of Band data
    fyles = sorted(glob.glob(path + '*.fits'))
    #print ( 'number of fits files: ', str(len(fyles)))
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
    trash1 = fyle.split("S0",1)
    fyle_number = trash1[1].split("_")[0]

    return fyle_number
   

def open_img(file_location): #opens the matrix containing image
    hdu = pf.open(file_location)
    hdr = hdu[0].header
    print(hdr)
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
    # example: star_path = "C:/Python34/GPIcode/star"
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

dm2 =   "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/2_dm"
dm3 =   "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/3_dm"
dm4 =   "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/4_dm"
dm5 =   "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/5_dm"
dm6 =   "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/6_dm"
dm7 =   "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/7_dm"
dm8 =   "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/8_dm" 


sat2 = "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/2_sat"
sat3 = "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/3_sat"
sat4 = "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/4_sat"
sat5 = "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/5_sat"
sat6 = "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/6_sat"
sat7 = "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/7_sat"
sat8 = "C:/Python34/GPIcode/h_band/no_skyval_sub/coronagraph_files/8_sat"


nc_2star = "C:/Python34/GPIcode/h_band/no_skyval_sub/No_coronograph_files/2_star"
nc_3star = "C:/Python34/GPIcode/h_band/no_skyval_sub/No_coronograph_files/3_star"
nc_4star = "C:/Python34/GPIcode/h_band/no_skyval_sub/No_coronograph_files/4_star"

nc_2dm   = "C:/Python34/GPIcode/h_band/no_skyval_sub/No_coronograph_files/2_dm"
nc_3dm   = "C:/Python34/GPIcode/h_band/no_skyval_sub/No_coronograph_files/3_dm"
nc_4dm   = "C:/Python34/GPIcode/h_band/no_skyval_sub/No_coronograph_files/4_dm"

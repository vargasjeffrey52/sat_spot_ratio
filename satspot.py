# Sat Spot Ratio (PSF - fitting method)
#--------------------------------------------------------

#================= Imports ==============================

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

#=========================================================
#=================== path and directories ================

hband_path = "saved/Hband/H_reduced_data/Hband/S20130211S0" # (Jeff's) location where reduced data located

#=========================================================





def minimize_psf(p, im1, im2, return_res_arr = False):
    """
    Function used by op.minimize to fit PSF
    p[0] = xoffset applied to im2
    p[1] = yoffset applied to im2
    p[2] = flux ratio between im1 and im2 (i.e. value to multiply im2 by to match im1)
    """

    if np.shape(im1) != np.shape(im2):
        print ('Mismatching dimensions')
        sys.exit()

    x, y = np.meshgrid(np.arange(s[1]), np.arange(s[0]))
    # To shift the image in positive x/y direction,
    # the coordinate array read by map_coordinates needs to be shifted in the opposite direction
    x -= p[0]
    y -= p[1]
    im2_shifted = ndimage.mao_coordinates(im2, (y, x), cval = np.nan)

    #Multiply im2_shifted by p[2]
    im2_shifted *= p[2]

    #Compute residual image
    res_arr = im1 - im2_shifted

    #Compute sum of absolute residuals
    res = np.nansum(np.abs(res_arr))

    if return_res_arr is True:
        return res, res_arr
    else:
        return res

def psf_fit(im1, im2, p0, return_res_arr=False):
    """
    Args:
        im1 - first image
        im2 - second image, same dimensions
        p0 - vector of guesses for PSF fit [xoffset, yoffset, flux_ratio]
    Return:
        p - vector of PSF fitting results [xoffset, yoffset, flux_ratio]
    
     Function to peform PSF fitting. Return residual array if true.
     Using scipy.optimize.minimize to do this
    """

    #I've set up some conservative limits on the fitting,
    # the x and y offset should be less than a pixel, and the flux ratio should be within a factor of a hundred
    result = optimize.minimize(minimize_psf, p0, args=(im1, im2), bounds=((-1, 1), (-1, 1), (1e-2, 1e2),))
    p = result.x

    if return_res_arr is True:
        res, res_arr = minimize_psf(p, im1, im2, return_res_arr = True)
        return p, res_arr
    else:
        return p

def fitxy (im, xy_guess):
    """"
    Args:
        im - 2D image
       xy_guess - initial guess of source to fit
    Return:
        xy - Tuple of fitted xy coordinates

     Function which fits the location of a point source within an image
     Same as return_pos in gpi_satspotcalib_wd.py
    """
    
    return xy

def stamp(im, xy, sout):
    """
    Args:
       im - 2D image of arbitrary dimensions (281x281 for GPI)
       xy - 2tuple containing x,y coordinates of spot/point source
        sout - size of output image, with source shifted to (sout/2, sout/2)
    Return:
        stamp - 2d image of size (sout, sout)

     Use ndimage.map_coordinates to create the stamp by interpolation.
     See lines 365-374 in gpi_satspotcalib_wd.py
    """
    
    return stamp




# ----------------------------- FUNCTIONS JEFF ADDED ------------------------------

#============================ Extract fits file information =======================
#==================================================================================


def multiple_files(path):
    """ Used to locate the fits data files.
    Args:
        path - this indicates the directory where the fits files are located.
            example) path = Hband_reduced = "saved/Hband/H_reduced_data/Hband/S20130211S0"
    Returns:
        creates a list of path for all fits files.
    """
    fyles = sorted(glob.glob(path + '*.fits'))
    
    return fyles

def file_number(name):

    """ used to split the file name and extract the last 3 numbers of the file name
    Args: 
        name - String name of the file. Example, "S20130211S0403_spdc.fits"
    Return: 
        last 3 digits in the file name. Example, filename : "S20130211S0403_spdc.fits" --> file_number = 403 
    """
    trash1 = name.split("S0",1)
    file_number = trash1[1].split("_")[0]

    return file_number
   
def get_info1(path,index): 

    """ used to extract information of the fits files
    Args:
        path - is the path in where the file is located (see multiple_files function)
        index - used to index file from multiple file array
    Return:
        hdr - Header in fits file
        img - image data cube with dimension (wavelength,ypixel,xpixel) ex) np.shape(img)>>>(37,281,281)
        num_file - last 3 number in the file name (see function: file_number)
    """
    name = multiple_files(path)[index]
    num_file = file_number(name)

    hdulist = pf.open(name)
    img = hdulist[1].data
    hdr = hdulist[1].header
    img = np.ma.array(img,mask = np.isnan(img))
    return hdr,img,num_file

#===================== create a cutout of spots and find center ===================
#==================================================================================


def pixel_map(image,x,y):
    """ Used to interpolate using the given x,y coordinates
    Args:
        image - is an image slice of the data cube (wavelength,ypixel,xpixel). slice => wavelength
        x     - array of x values
        y     - array of y values 
    return:
        image value at the specified x and y coordinates.

    """

    image[np.isnan(image)] = np.nanmedian(image)
    return ndimage.map_coordinates(image, (y,x),cval=np.nan)



def gen_xy(size):
    """ used to construct a box cut out of the original image, with the box having an area of size x size
    Args:
        size - indicates the size of the box.
    Return:
        x,y  - contains the index of row and column, i.e coordinates of the within the specified box

    """
#      
    s = np.array([size,size])
    x,y = np.meshgrid(np.arange(s[1]),np.arange(s[0]))
    return x,y


def twoD_Gaussian(xy, amplitude, xo, yo, sigma, offset):
    """ Two dimensional Gaussian

    """
    x,y = xy
    xo = float(xo)
    yo = float(yo)    
    a = 1.0 / (2.0 * sigma**2.0)
    b = 1.0 / (2.0 * sigma**2.0)
    g = offset + amplitude*np.exp( - ((a*((x-xo)**2)) + (b*((y-yo)**2))))
    
    return g.ravel()


def return_pos(im, xy_guess,x,y):
    """ used to obtain the optimized coordinates of the center for the boxed cutout image
    Arg:
        im - the box cutout of the original image
        xy_guess - x and y coordinates for initial guess to the center of the box
        x,y - together they indicates index array of coordinates withing the box
    Return:
        popt[1]-[2] - Optimized x and y coordinates for center of box.
    """
    p0 = [np.nanmax(im), xy_guess[0], xy_guess[1], 3.0, 0.0]
    # 3.0 - the full width half max in pixel (width of Gaussian)
    # 0.0 - constant value that shift plane of Gaussian 
    # p0  - guess for the 5 parameter of 2d gaussian (amplitud,xcenter,ycenter,sigmax,sigmay) where the
    #           sigmas are spread of Gaussian.

    popt, pcov = optimize.curve_fit(twoD_Gaussian, (x, y), im.ravel(), p0 = p0)
    print(popt)  

    return popt[1],  popt[2]


def pixel_cutout(image,size,xguess, yguess, name1='none', name2='none',save = False):
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
    x,y = gen_xy(size)
    x += (xguess-size/2.)
    y += (yguess-size/2.)
   
    output = pixel_map(image,x,y)
    xc,yc = return_pos(output, (xguess,yguess), x,y)
    
    if save == True:
        # image before center optimization
        write = pf.writeto(name1, output,clobber = True)

    x,y = gen_xy(size)
    x += (xc-size/2.)
    y += (yc-size/2.)
    output = pixel_map(image,x,y)

    if save == True:
        # image after center optimization
        write = pf.writeto(name2, output,clobber = True)

    return output


def loop_pixcut(image,size,center_guess,imslice=0,save = False):
    """ Loops over same slice cuts out the specified spots. For example, Dm spot
        have 4 location in any given slice which means it will cut out 4 images. It
        then will will average the cut out images to produce 1 averaged image.
    Args:
        image - a slice of the original data cube
        size  - the size of the sides of the box cutout
        center_guess - (x_i,y_i) coordinate array
        imslice - used to name and indicate which slice is the cut being don on
        save  - option to save image cutout with initial guess and after
                center has been optimized
    Return:
        box_img - average image for same slice. 
    """
    box_img = []
    spot = 0
    
    for i in center_guess:

        if save == True:
            # SAVES TO CURRENT DIRECTORY
            spoti = ['A','B','C','D']
            name1 = 'b_opt'+'_sat'+str(spoti[spot])+'s'+ str(imslice)+'.fits'
            name2 = 'a_opt'+'_sat'+str(spoti[spot])+'s'+ str(imslice)+'.fits'
            cutout = pixel_cutout(image,size,i[0],i[1],name1,name2,save)
        else:
            cutout = pixel_cutout(image,size,i[0],i[1])

        box_img.append(cutout)
        spot +=1

    box_img = (np.sum(box_img,axis=0))/len(box_img)
    
    return box_img


def slice_loop(path,fnum,size,center_guess,save = False):
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
    """



    image = get_info1(path,fnum)[1] # 1 returns the img from get_info1 function
    center = open_img(center_guess)
    box = []
    imslice = 0
    
    for img in image:
        cent = center[imslice]
        ave_cut = loop_pixcut(img,size,cent,imslice,save)
        box.append(ave_cut)
        imslice +=1

    box = np.array(box)
    print('shape of cube ave image', np.shape(box))
   
    return box  










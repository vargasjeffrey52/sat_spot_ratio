import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
import os
from scipy import ndimage
from scipy import optimize
import multiprocessing as mp
import numpy.fft as fft

def ratio_dm(list_dm, list_sat, star_pos, dm_pos1, dm_pos2, sat_pos, first_slice = 0, last_slice = 37, high_pass = False, box_size = 8, nudgexy = False, save_gif = False):
    """
    Main function for DM spot data

    Input:
        list_dm - list of files containing star and dm spots
        list_sat - list of files containing dm spots and sat spots
        star_pos - 2 tuple for star pixel position (x, y) - fixed with lambda
        dm_pos1 - (37, 4, 2) array of dm spot positions in star/dm images
        dm_pos2 - (37, 4, 2) array of dm spot positions in dm/sat images (should be the same as dm_pos1)
        sat_pos - (37, 4, 2) array of sat spot positions in dm/sat images

    Returns:
        wl - wavelength axis
        star_dm_ratio - star to dm ratio (n, 37) array
        dm_sat_ratio - dm to sat ratio (n, 37) array
    """

    #Convert star_pos into a (37, 2) array for reasons
    star_pos = np.tile(star_pos, (37, 1))

    list_dm = np.loadtxt(list_dm, dtype=str)
    n_list_dm = len(list_dm)
    list_sat = np.loadtxt(list_sat, dtype=str)
    n_list_sat = len(list_sat)

    #Check all files have same wavelength solution
    header = fits.getheader(list_dm[0], 1)
    wl = (np.arange(37)*header['CD3_3']) + header['CRVAL3']
    
    for i in range(0, n_list_dm):
        header = fits.getheader(list_dm[i], 1)
        if np.sum(wl - ((np.arange(37)*header['CD3_3']) + header['CRVAL3'])) != 0:
            print 'Wavelength axes do not match'
            return 0

    for i in range(0, n_list_sat):
        header = fits.getheader(list_sat[i], 1)
        if np.sum(wl - ((np.arange(37)*header['CD3_3']) + header['CRVAL3'])) != 0:
            print 'Wavelength axes do not match'
            return 0

    #This will be parallelized
    star_dm_ratio = np.zeros((n_list_dm, 37), dtype=np.float64) * np.nan
    dm_sat_ratio = np.zeros((n_list_dm, 37), dtype=np.float64) * np.nan

    pool = mp.Pool()
    kw = {'first_slice': first_slice, 'last_slice': last_slice, 'high_pass': high_pass, 'box_size': box_size, 'nudgexy': nudgexy, 'save_gif': save_gif}
    result1 = [pool.apply_async(slice_loop, (i, list_dm[i], star_pos, dm_pos1, 'ASU', 'DM spot'), kw) for i in range(0, n_list_dm)]  
    kw = {'first_slice': first_slice, 'last_slice': last_slice, 'high_pass': high_pass, 'box_size': box_size, 'nudgexy': nudgexy, 'save_gif': save_gif}
    result2 = [pool.apply_async(slice_loop, (i, list_sat[i], dm_pos2, sat_pos, 'DM spot', 'Sat spot'), kw) for i in range(0, n_list_sat)]  
    
    output = [p.get() for p in result1]
    for i in range(0, n_list_dm):
        star_dm_ratio[output[i][0]] = output[i][1]

    output = [p.get() for p in result2]
    for i in range(0, n_list_dm):
        dm_sat_ratio[output[i][0]] = output[i][1]

    pool.close()
    pool.join()

    return wl, star_dm_ratio, dm_sat_ratio


def ratio_companion():
    #Main function for companion datasets
    #Will have to accept a stellar spectrum
    foo = 1

    return 0


def slice_loop(index, file, xy1, xy2, name1, name2, first_slice = 0, last_slice = 37, high_pass = False, box_size = 8, nudgexy = False, save_gif = False):

    """
        First object should be brighter than the second, xy1 = star, xy2 = dm, xy1 = dm, xy2 = sat.
    """

    stamp1 = np.zeros((37, box_size+4, box_size+4), dtype=np.float64)
    stamp2 = np.zeros((37, box_size+4, box_size+4), dtype=np.float64)
    scales = np.zeros(37, dtype=np.float64) * np.nan

    cube = fits.getdata(file)
    header = fits.getheader(file, 1)
    base_name = os.path.basename(file)

    stamp_cm = 'gnuplot2'
    

    for i in range(first_slice, last_slice):

        if save_gif is True:
            fig = plt.figure(figsize=(9,10))
            fig.suptitle(file+', slice='+str(i),fontsize=14)

        im = cube[i]
        if high_pass is not False:
            im = high_pass_filter(im, high_pass)

        for xy, stamp, name, plt_pos in zip((xy1[i], xy2[i]), (stamp1, stamp2), (name1, name2), ((1, 3), (2, 6))):
            if len(xy) == 2:
                stamp[i] = extract_stamp(im, xy, box_size)
                if save_gif is True:
                    ax = plt.subplot(4, 3, plt_pos[0])
                    ax.imshow(stamp[i], interpolation = 'nearest', cmap = stamp_cm)
                    ax.set_title(name, fontsize = 10)
                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])
            else:
                for j in range(0, 4):
                    this_stamp = extract_stamp(im, xy[j], box_size)
                    if save_gif is True:
                        ax = plt.subplot(4, 3, plt_pos[0]+(j*3))
                        ax.imshow(this_stamp, interpolation = 'nearest', cmap = stamp_cm)
                        ax.set_title(name+' #'+str(j), fontsize = 10)
                        ax.xaxis.set_ticklabels([])
                        ax.yaxis.set_ticklabels([])
                    stamp[i] += this_stamp
                stamp[i] /= 4.0

            if save_gif is True:
                ax = plt.subplot(4, 3, plt_pos[1])
                cb = ax.imshow(stamp[i], interpolation = 'nearest', cmap = stamp_cm)
                cb = fig.colorbar(cb)
                cb.ax.tick_params(labelsize = 8)
                ax.set_title('Average '+name, fontsize = 10)
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

        #Compute scale factor here
        scales[i] = find_scale(stamp1[i], stamp2[i], nudgexy = False)

        if save_gif is True:
            ax = plt.subplot(4, 3, 9)
            cb = ax.imshow(stamp1[i]*scales[i], interpolation = 'nearest', cmap = stamp_cm)
            cb = fig.colorbar(cb)
            cb.ax.tick_params(labelsize = 8)
            ax.set_title(name1+' x '+str(scales[i]), fontsize=10)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            ax = plt.subplot(4, 3, 12)
            cb = ax.imshow(((stamp1[i]*scales[i]) - stamp2[i])/np.nanmax(stamp2[i]), interpolation = 'nearest', cmap = 'bwr', vmin = -0.1, vmax = 0.1)
            cb = fig.colorbar(cb)
            cb.ax.tick_params(labelsize = 8)
            ax.set_title('Fract. Resid.', fontsize=10)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])


            fig.subplots_adjust(wspace=0.10, hspace=0.15)
            plt.savefig('Frames-'+base_name.replace('.fits','')+'-'+str(i).zfill(2)+'.png', dpi = 250, bbox_inches='tight')
            plt.close('all')

    #Create gif here
    if save_gif is True:
        str_box = 's'+str(box_size)
        
        if high_pass is not False:
            str_hp = 'hp'+str(high_pass)
        else:
            str_hp = 'hp0'

        if nudgexy is True:
            str_xy = 'nudge1'
        else:
            str_xy = 'nudge0'


        os.system('convert -delay 25 -loop 0 Frames-'+base_name.replace('.fits','')+'-*.png Animation-'+str_box+'-'+str_hp+'-'+str_xy+'-'+base_name.replace('.fits','')+'.gif')
        for i in range(first_slice, last_slice):
            os.remove('Frames-'+base_name.replace('.fits','')+'-'+str(i).zfill(2)+'.png')

    return index, scales


def find_scale(stamp1, stamp2, nudgexy = False):

    if (np.size(stamp1) != np.size(stamp2)):
        print 'Stamps do not have same dimensions'
        return 0

    guess =  np.nanmax(stamp2) / np.nanmax(stamp1)
    result = optimize.minimize(minimize_psf, guess, args=(stamp1, stamp2), method = 'Nelder-Mead') 
    scale = result.x[0]

    return scale


def minimize_psf(scale, im1, im2): 
    """ Simply minimize residuals
    Args:
        scale - scale factor 
        ave_dm - average dm for a given slice
        ave_sat - average sat for a given slice 
    return:
        residuals for ave_dm and ave_sat

    """
    return np.nansum(np.abs(((scale*im1) - im2)))


def extract_stamp(im, xy, box_size):
    """ Extracts stamp centered on star/spot in image based on initial guess
    Args:
        image - a slice of the original data cube
        xy - initial xy coordinate guess to center of spot
        box_size - size of stamp to be extracted (actually, size of radial mask, box is 4 pixels bigger)
    Return:
        output - box cutout of spot with optimized center 
    """
    
    box_size = float(box_size)
    xguess = float(xy[0])
    yguess = float(xy[1])

    #Exctracts a 20px stamp centered on the guess
    x,y = gen_xy(20.0)
    x += (xguess-20/2.)
    y += (yguess-20/2.)
    output = pixel_map(im,x,y)

    #Fits location of star/spot
    xc,yc = return_pos(output, (xguess,yguess), x,y)
    
    #Extracts a box_size + 4 width stamp centered on exact position
    x,y = gen_xy(box_size+4)
    x += (xc-np.round((box_size+4)/2.))
    y += (yc-np.round((box_size+4)/2.))
    output = pixel_map(im,x,y)

    #Apply radial mask
    r = np.sqrt((x-xc)**2 + (y-yc)**2)
    output[np.where(r>box_size/2)] = np.nan

    return output


def pixel_map(image,x,y):

    image[np.isnan(image)] = np.nanmedian(image)

    return ndimage.map_coordinates(image, (y,x),cval=np.nan)


def gen_xy(size):

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

    p0 = [np.nanmax(im), xy_guess[0], xy_guess[1], 3.0, 0.0]
    popt, pcov = optimize.curve_fit(twoD_Gaussian, (x, y), im.ravel(), p0 = p0, maxfev=15000000)

    return popt[1],  popt[2]


def high_pass_filter(img, filtersize=10):
    """
    A FFT implmentation of high pass filter from pyKLIP.

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

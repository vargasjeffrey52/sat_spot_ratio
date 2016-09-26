import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
import os
from scipy import ndimage
from scipy import optimize
import multiprocessing as mp
import numpy.fft as fft

def ratio_dm(list_dm, list_sat, star_pos, dm_pos1, dm_pos2, sat_pos, first_slice = 0, last_slice = 36, high_pass = False, box_size = 8, nudgexy = False, save_gif = False):
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
    n_dm = np.size(list_dm)

    list_sat = np.loadtxt(list_sat, dtype=str)
    n_sat = np.size(list_sat)

    #Check all files have same wavelength solution
    header = fits.getheader(list_dm[0], 1)
    wl = (np.arange(37)*header['CD3_3']) + header['CRVAL3']
    
    for i in range(0, n_dm):
        header = fits.getheader(list_dm[i], 1)
        if np.sum(wl - ((np.arange(37)*header['CD3_3']) + header['CRVAL3'])) != 0:
            print 'Wavelength axes do not match'
            return 0

    for i in range(0, n_sat):
        header = fits.getheader(list_sat[i], 1)
        if np.sum(wl - ((np.arange(37)*header['CD3_3']) + header['CRVAL3'])) != 0:
            print 'Wavelength axes do not match'
            return 0

    #This will be parallelized
    star_dm_ratio = np.zeros((n_dm, 37), dtype=np.float64) * np.nan
    dm_sat_ratio = np.zeros((n_sat, 37), dtype=np.float64) * np.nan

    #for i in xrange(0, n_dm):
    #    print list_dm[i]
    #    a, b = slice_loop(i, list_dm[i], star_pos, dm_pos1, 'ASU', 'DM spot', first_slice = 2, last_slice = 18, high_pass = 16, box_size = 8, nudgexy = True, save_gif = False)
    #print ajsdkljl

    pool = mp.Pool()
    kw = {'first_slice': first_slice, 'last_slice': last_slice, 'high_pass': high_pass, 'box_size': box_size, 'nudgexy': nudgexy, 'save_gif': save_gif}
    result1 = [pool.apply_async(slice_loop, (i, list_dm[i], star_pos, dm_pos1, 'ASU', 'DM spot'), kw) for i in range(0, n_dm)]  
    kw = {'first_slice': first_slice, 'last_slice': last_slice, 'high_pass': high_pass, 'box_size': box_size, 'nudgexy': nudgexy, 'save_gif': save_gif}
    result2 = [pool.apply_async(slice_loop, (i, list_sat[i], dm_pos2, sat_pos, 'DM spot', 'Sat spot'), kw) for i in range(0, n_sat)]  
    
    output = [p.get() for p in result1]
    for i in range(0, n_dm):
        star_dm_ratio[output[i][0]] = output[i][1]

    output = [p.get() for p in result2]
    for i in range(0, n_sat):
        dm_sat_ratio[output[i][0]] = output[i][1]

    pool.close()
    pool.join()

    #Also combine all dm and sat images together and run again
    avg_dm_cube = np.zeros((n_dm, 37, 281, 281), dtype=np.float64)
    for i in xrange(0, n_dm):
        avg_dm_cube[i] = fits.getdata(list_dm[i], 1)
    avg_dm_cube = np.nanmean(avg_dm_cube, axis=0)
    avg_name = os.path.basename(list_dm[0]).replace('.fits', '_avg.fits')
    index, avg_star_dm_ratio = slice_loop(0, None, star_pos, dm_pos1, 'ASU', 'DM spot', first_slice = first_slice, last_slice = last_slice, high_pass = high_pass, box_size = box_size, nudgexy = nudgexy, save_gif = save_gif, avg_cube = avg_dm_cube, avg_name = avg_name)

    avg_sat_cube = np.zeros((n_sat, 37, 281, 281), dtype=np.float64)
    for i in xrange(0, n_sat):
        avg_sat_cube[i] = fits.getdata(list_sat[i], 1)
    avg_sat_cube = np.nanmean(avg_sat_cube, axis=0)
    avg_name = os.path.basename(list_sat[0]).replace('.fits', '_avg.fits')
    index, avg_dm_sat_ratio = slice_loop(0, None, dm_pos2, sat_pos, 'DM spot', 'Sat spot', first_slice = first_slice, last_slice = last_slice, high_pass = high_pass, box_size = box_size, nudgexy = nudgexy, save_gif = save_gif, avg_cube = avg_sat_cube, avg_name = avg_name)


    return wl, star_dm_ratio, dm_sat_ratio, avg_star_dm_ratio, avg_dm_sat_ratio


def ratio_companion():
    #Main function for companion datasets
    #Will have to accept a stellar spectrum
    foo = 1

    return 0


def slice_loop(index, file, xy1, xy2, name1, name2, first_slice = 0, last_slice = 36, high_pass = False, box_size = 8, nudgexy = False, save_gif = False, avg_cube = None, avg_name = None):

    """
        First object should be brighter than the second, xy1 = star, xy2 = dm, xy1 = dm, xy2 = sat.
    """

    stamp1 = np.zeros((37, box_size+4, box_size+4), dtype=np.float64)
    stamp2 = np.zeros((37, box_size+4, box_size+4), dtype=np.float64)
    scales = np.zeros(37, dtype=np.float64) * np.nan

    if avg_cube is not None:
        cube = np.copy(avg_cube)
        base_name = avg_name
        file = avg_name
    else:
        cube = fits.getdata(file)
        base_name = os.path.basename(file)

    stamp_cm = 'gnuplot2'
    

    for i in range(first_slice, last_slice+1):

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
                    ax.imshow(radial_mask(stamp[i], box_size), interpolation = 'nearest', cmap = stamp_cm)
                    ax.set_title(name, fontsize = 10)
                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])
            else:
                for j in range(0, 4):
                    this_stamp = extract_stamp(im, xy[j], box_size)
                    if save_gif is True:
                        ax = plt.subplot(4, 3, plt_pos[0]+(j*3))
                        ax.imshow(radial_mask(this_stamp, box_size), interpolation = 'nearest', cmap = stamp_cm)
                        ax.set_title(name+' #'+str(j), fontsize = 10)
                        ax.xaxis.set_ticklabels([])
                        ax.yaxis.set_ticklabels([])
                    stamp[i] += this_stamp
                stamp[i] /= 4.0

            if save_gif is True:
                ax = plt.subplot(4, 3, plt_pos[1])
                cb = ax.imshow(radial_mask(stamp[i], box_size), interpolation = 'nearest', cmap = stamp_cm)
                cb = fig.colorbar(cb)
                cb.ax.tick_params(labelsize = 8)
                ax.set_title('Average '+name, fontsize = 10)
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

        #Compute scale factor here
        scales[i], dx, dy, shifted_stamp1 = find_scale(stamp1[i], stamp2[i], box_size, nudgexy = nudgexy)
        if nudgexy is True:
            stamp1[i] = shifted_stamp1

        if save_gif is True:
            ax = plt.subplot(4, 3, 9)
            cb = ax.imshow(radial_mask(stamp1[i]*scales[i], box_size), interpolation = 'nearest', cmap = stamp_cm)
            cb = fig.colorbar(cb)
            cb.ax.tick_params(labelsize = 8)
            ax.set_title(name1+' x '+str(scales[i]), fontsize=10)

            if nudgexy is True:
                dx_str = '%0.3f' % (dx)
                dy_str = '%0.3f' % (dy)
                ax.annotate('dx = '+dx_str, xy=(0.2,0.85), xycoords='axes fraction', fontsize=8)
                ax.annotate('dy = '+dy_str, xy=(0.2,0.80), xycoords='axes fraction', fontsize=8)

            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            ax = plt.subplot(4, 3, 12)
            cb = ax.imshow(radial_mask(((stamp1[i]*scales[i]) - stamp2[i])/np.nanmax(stamp1[i]*scales[i]), box_size), interpolation = 'nearest', cmap = 'bwr', vmin = -0.1, vmax = 0.1)
            cb = fig.colorbar(cb)

            cb.ax.tick_params(labelsize = 8)
            ax.set_title('Fract. Resid.', fontsize=10)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            fig.subplots_adjust(wspace=0.10, hspace=0.15)
            plt.savefig('Frames-'+base_name.replace('.fits','')+'-'+str(i).zfill(2)+'.png', dpi = 200, bbox_inches='tight')
            plt.close('all')

    #Create gif here
    if save_gif is True:
        str_box = 's'+str(box_size).zfill(2)
        
        if high_pass is not False:
            str_hp = 'hp'+str(high_pass).zfill(2)
        else:
            str_hp = 'hp00'

        if nudgexy is True:
            str_xy = 'nudge1'
        else:
            str_xy = 'nudge0'


        os.system('convert -delay 25 -loop 0 Frames-'+base_name.replace('.fits','')+'-*.png Figures/gifs/Animation-'+str_box+'-'+str_hp+'-'+str_xy+'-'+base_name.replace('.fits','')+'.gif')
        for i in range(first_slice, last_slice+1):
            os.remove('Frames-'+base_name.replace('.fits','')+'-'+str(i).zfill(2)+'.png')

    return index, scales


def find_scale(im1, im2, box_size, nudgexy = False):

    if (np.size(im1) != np.size(im2)):
        print 'Stamps do not have same dimensions'
        return 0

    if nudgexy is False:
        guess =  np.nanmax(im2) / np.nanmax(im1)
        result = optimize.minimize(minimize_psf, guess, args=(radial_mask(im1, box_size), radial_mask(im2, box_size), box_size, nudgexy), method = 'Nelder-Mead') 
        scale = result.x[0]
        dx = 0.0
        dy = 0.0
        shifted_im1 = np.copy(im1)
    else:
        guess = (np.nanmax(im2) / np.nanmax(im1), 0.0, 0.0)
        #result = optimize.minimize(minimize_psf, guess, args=(im1, radial_mask(im2, box_size), box_size, nudgexy), bounds = ((guess[0]*0.01, guess[0]*100.), (-0.5, 0.5), (-0.5, 0.5)), method = 'SLSQP') 
        result = optimize.minimize(minimize_psf, guess, args=(im1, radial_mask(im2, box_size), box_size, nudgexy), method = 'Nelder-Mead') 
        scale = result.x[0]
        dx = result.x[1]
        dy = result.x[2]

        x, y = gen_xy(box_size + 4)
        x += dx
        y += dy
        shifted_im1 = ndimage.map_coordinates(im1, (y, x), cval = np.nan)

    return scale, dx, dy, shifted_im1


def minimize_psf(p, im1, im2, box_size, nudgexy): 
    """ Simply minimize residuals
    Args:
        scale - scale factor 
        ave_dm - average dm for a given slice
        ave_sat - average sat for a given slice 
    return:
        residuals for ave_dm and ave_sat
    """
    if nudgexy is False:
        return np.nansum(np.abs(((p*im1) - im2)))
    else:
        x, y = gen_xy(box_size + 4)
        x += p[1]
        y += p[2]
        shifted_im1 = ndimage.map_coordinates(im1, (y, x), cval = np.nan)

        return np.nansum(np.abs(((p[0]*shifted_im1) - im2)))


def radial_mask(im, box_size, return_indx = False):

    new_im = np.copy(im)
    x, y = gen_xy(box_size + 4)
    xc = np.round((box_size + 4) / 2.0)
    yc = np.round((box_size + 4) / 2.0)
    r = np.sqrt((x - xc)**2 + (y - yc)**2)
    new_im[np.where(r > box_size/2.0)] = np.nan

    if return_indx is False:
        return new_im
    else:
        return new_im, np.where(r > box_size/2.0)


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

    #Exctracts a 10px stamp centered on the guess and refines based on maximum pixel location
    for i in range(0, 2):
        x,y = gen_xy(10.0)
        x += (xguess-10/2.)
        y += (yguess-10/2.)
        output = pixel_map(im,x,y)
        xguess = x[np.unravel_index(np.nanargmax(output), np.shape(output))]
        yguess = y[np.unravel_index(np.nanargmax(output), np.shape(output))]

    #Fits location of star/spot
    xc,yc = return_pos(output, (xguess,yguess), x,y)
    
    #Extracts a box_size + 4 width stamp centered on exact position
    x,y = gen_xy(box_size+4)
    x += (xc-np.round((box_size+4)/2.))
    y += (yc-np.round((box_size+4)/2.))
    output = pixel_map(im,x,y)

    return output


def pixel_map(image,x,y):

    image[np.isnan(image)] = np.nanmedian(image)

    return ndimage.map_coordinates(image, (y,x), cval=np.nan)


def gen_xy(size):

    s = np.array([size,size])
    x,y = np.meshgrid(np.arange(s[1], dtype=np.float64),np.arange(s[0], dtype=np.float64))

    return x,y


def twoD_Gaussian(p, data, xy):

    x,y = xy   
    a = 1.0 / (2.0 * p[3]**2.0)
    b = 1.0 / (2.0 * p[3]**2.0)
    g =  p[0]*np.exp( -((a*((x-p[1])**2)) + (b*((y-p[2])**2))))
    
    return np.nansum((data - g)**2.0)


def return_pos(im, xy_guess,x,y):

    p0 = [np.nanmax(im), xy_guess[0], xy_guess[1], 1.25]
    result = optimize.minimize(twoD_Gaussian, p0, args = (im, (x, y)), method = 'Nelder-Mead')

    if result.success is False:
        fits.writeto('test_im.fits', im, clobber=True)

        p  =p0  
        a = 1.0 / (2.0 * p[3]**2.0)
        b = 1.0 / (2.0 * p[3]**2.0)
        g =  p[0]*np.exp( -((a*((x-p[1])**2)) + (b*((y-p[2])**2))))
        fits.writeto('test_model.fits', g, clobber=True)

        print p0
        print np.min(x), np.min(y)
        print result
        print asjkljl
    return result.x[1], result.x[2]


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
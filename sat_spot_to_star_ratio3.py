import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
import os
from scipy import ndimage
from scipy import optimize
import multiprocessing as mp
import numpy.fft as fft
import warnings
import glob

def ratio_dm(list_dm, list_sat, star_pos, dm_pos1, dm_pos2, sat_pos, first_slice = 0, last_slice = 36, high_pass = 0, box_size = 8, nudgexy = False, offset = False, save_gif = False, path = '',order=1):
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

    list_dm = np.genfromtxt(list_dm, dtype=str)
    n_dm = np.size(list_dm)

    list_sat = np.genfromtxt(list_sat, dtype=str)
    n_sat = np.size(list_sat)

    #Check all files have same wavelength solution
    header = fits.getheader(path+list_dm[0], 1)
    wl = (np.arange(37)*header['CD3_3']) + header['CRVAL3']
    
    for i in range(0, n_dm):
        header = fits.getheader(path+list_dm[i], 1)
        if np.sum(wl - ((np.arange(37)*header['CD3_3']) + header['CRVAL3'])) != 0:
            print ('Wavelength axes do not match')
            return 0

    for i in range(0, n_sat):
        header = fits.getheader(path+list_sat[i], 1)
        if np.sum(wl - ((np.arange(37)*header['CD3_3']) + header['CRVAL3'])) != 0:
            print ('Wavelength axes do not match')
            return 0

    #Save some file name strings
    str_box = 's'+str(box_size).zfill(2)
    str_hp = 'hp'+str(high_pass).zfill(2)
    if nudgexy is True:
        str_xy = 'nudge1'
    else:
        str_xy = 'nudge0'
    if offset is True:
        str_offset = 'offset1'
    else:
        str_offset = 'offset0'

    #This will be parallelized
    star_dm_ratio = np.zeros((n_dm, 37), dtype=np.float64) * np.nan
    star_dm_resid = np.zeros((n_dm, 37), dtype=np.float64) * np.nan
    dm_sat_ratio = np.zeros((n_sat, 37), dtype=np.float64) * np.nan
    dm_sat_resid = np.zeros((n_sat, 37), dtype=np.float64) * np.nan

    #foo = slice_loop(0, first_slice, list_dm[0], star_pos, dm_pos1, 'ASU', 'DM spot', high_pass = high_pass, box_size = box_size, nudgexy = nudgexy, offset=offset, save_gif = save_gif, path = path, order = order)
    #print kaljlkj

    pool = mp.Pool()
    kw = {'high_pass': high_pass, 'box_size': box_size, 'nudgexy': nudgexy, 'offset': offset, 'save_gif': save_gif, 'path': path,"order": order}
    result1 = [pool.apply_async(slice_loop, (i, j, list_dm[i], star_pos, dm_pos1, 'ASU', 'DM spot'), kw) for i in range(0, n_dm) for j in range(first_slice, last_slice+1)]  
    kw = {'high_pass': high_pass, 'box_size': box_size, 'nudgexy': nudgexy, 'offset': offset, 'save_gif': save_gif, 'path': path, "order": order}
    result2 = [pool.apply_async(slice_loop, (i, j, list_sat[i], dm_pos2, sat_pos, 'DM spot', 'Sat spot'), kw) for i in range(0, n_sat) for j in range(first_slice, last_slice+1)]  
    
    output = [p.get() for p in result1]
    count = 0
    for i in range(0, n_dm):
        for j in range(first_slice, last_slice+1):
            star_dm_ratio[output[count][0],output[count][1]] = output[count][2]
            star_dm_resid[output[count][0],output[count][1]] = output[count][3]
            count +=1

    output = [p.get() for p in result2]
    count = 0
    for i in range(0, n_sat):
        for j in range(first_slice, last_slice+1):
            dm_sat_ratio[output[count][0], output[count][1]] = output[count][2]
            dm_sat_resid[output[count][0], output[count][1]] = output[count][3]
            count += 1

    #Now plot if save_gif is True
     #Create gif here
    if save_gif is True:
        foo = [pool.apply_async(convert_gif, (path, list_dm[i], str_box, str_hp, str_xy, str_offset, order)) for i in range(0, n_dm)]
        foo = [pool.apply_async(convert_gif, (path, list_sat[i], str_box, str_hp, str_xy, str_offset, order)) for i in range(0, n_sat)]

    #Also combine all dm and sat images together and run again
    avg_dm_cube = np.zeros((n_dm, 37, 281, 281), dtype=np.float64)
    for i in range(0, n_dm):
        avg_dm_cube[i] = fits.getdata(path+list_dm[i], 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_dm_cube = np.nanmean(avg_dm_cube, axis=0)

    avg_name  = path+'diag_avg_dm_cube_'+str(high_pass)+'.fits'
    if (high_pass != 0) and (os.path.isfile(avg_name) is False):
        for i in range(0, 37):
            im = avg_dm_cube[i, :, :]
            avg_dm_cube[i, :, :] = high_pass_filter(im, high_pass)
        fits.writeto(avg_name, avg_dm_cube, clobber=True)

    avg_sat_cube = np.zeros((n_sat, 37, 281, 281), dtype=np.float64)
    for i in range(0, n_sat):
        avg_sat_cube[i] = fits.getdata(path+list_sat[i], 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_sat_cube = np.nanmean(avg_sat_cube, axis=0)

    avg_name = path+'diag_avg_sat_cube_'+str(high_pass)+'.fits'
    if (high_pass != 0) and (os.path.isfile(avg_name) is False):
        for i in range(0, 37):
            im = avg_sat_cube[i, :, :]
            avg_sat_cube[i, :, :] = high_pass_filter(im, high_pass)
        fits.writeto(avg_name, avg_sat_cube, clobber=True)


    kw = {'high_pass': high_pass, 'box_size': box_size, 'nudgexy': nudgexy, 'offset': offset, 'save_gif': True, 'avg_cube': avg_dm_cube, 'avg_name': os.path.basename(list_dm[0]).replace('.fits', '_avg.fits'), 'path': path,"order":order}
    result1 = [pool.apply_async(slice_loop, (0, j, None, star_pos, dm_pos1, 'ASU', 'DM spot'), kw) for j in range(first_slice, last_slice+1)]
    kw = {'high_pass': high_pass, 'box_size': box_size, 'nudgexy': nudgexy, 'offset': offset, 'save_gif': True, 'avg_cube': avg_sat_cube, 'avg_name': os.path.basename(list_sat[0]).replace('.fits', '_avg.fits'), 'path': path,"order": order}
    result2 = [pool.apply_async(slice_loop, (0, j, None, dm_pos2, sat_pos, 'DM spot', 'Sat spot'), kw) for j in range(first_slice, last_slice+1)]

    avg_star_dm_ratio = np.zeros(37, dtype=np.float64) * np.nan
    avg_dm_sat_ratio = np.zeros(37, dtype=np.float64) * np.nan

    output = [p.get() for p in result1]
    count = 0
    for j in range(first_slice, last_slice+1):
        avg_star_dm_ratio[output[count][1]] = output[count][2] 
        count += 1

    output = [p.get() for p in result2]
    count = 0
    for j in range(first_slice, last_slice+1):
        avg_dm_sat_ratio[output[count][1]] = output[count][2]
        count += 1

    pool.close()
    pool.join()

    convert_gif(path, list_dm[0].replace('.fits','_avg.fits'), str_box, str_hp, str_xy, str_offset, order)
    convert_gif(path, list_sat[0].replace('.fits','_avg.fits'), str_box, str_hp, str_xy, str_offset, order)
    if order == 1 :
        order_path = "Figures/"
    else:
        order_path = "Figures/2nd_order/"

    for f in glob.glob(path+ order_path+'Frames-'+str_box+'-'+str_hp+'-'+str_xy+'-'+str_offset+'*.png'):
        os.remove(f)

    return wl, star_dm_ratio, dm_sat_ratio, star_dm_resid, dm_sat_resid, avg_star_dm_ratio, avg_dm_sat_ratio


def ratio_companion():
    #Main function for companion datasets
    #Will have to accept a stellar spectrum
    foo = 1

    return 0

def slice_loop(index, slice, file, xy1, xy2, name1, name2, high_pass = 0, box_size = 8, nudgexy = False, offset = False, save_gif = False, avg_cube = None, avg_name = None, path = '',order =1):

    """
        First object should be brighter than the second, xy1 = star, xy2 = dm, xy1 = dm, xy2 = sat.
    """

    stamp1 = np.zeros((box_size+4, box_size+4), dtype=np.float64)
    stamp2 = np.zeros((box_size+4, box_size+4), dtype=np.float64)

    if avg_cube is not None:
        cube = np.copy(avg_cube)
        base_name = avg_name
        file = avg_name
    else:
        cube = fits.getdata(path+file)
        base_name = os.path.basename(path+file)

    stamp_cm = 'gnuplot2'
    
    i = slice

    if save_gif is True:
        #fig = plt.figure(figsize=(9,10))
        fig, all_ax = plt.subplots(4, 3, figsize=(9, 10))
        fig.suptitle(file+', slice='+str(i),fontsize=14)

    im = cube[i]
    if high_pass != 0:
        im = high_pass_filter(im, high_pass)

    for xy, stamp, name, plt_pos in zip((xy1[i], xy2[i]), (stamp1, stamp2), (name1, name2), (0, 1)):
        if len(xy) == 2:
            stamp[:,:] = extract_stamp(im, xy, box_size)
            if save_gif is True:
                #ax = plt.subplot(4, 3, plt_pos[0])
                ax = all_ax[0][plt_pos]
                ax.imshow(radial_mask(stamp, box_size), interpolation = 'none', cmap = stamp_cm)
                ax.set_title(name, fontsize = 10)
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

                for j in range(1, 4):
                    all_ax[j][plt_pos].xaxis.set_ticklabels([])
                    all_ax[j][plt_pos].yaxis.set_ticklabels([])
                    all_ax[j][plt_pos].axis('off')
        else:
            for j in range(0, 4):
                this_stamp = extract_stamp(im, xy[j], box_size)
                if save_gif is True:
                    #ax = plt.subplot(4, 3, plt_pos[0]+(j*3))
                    ax = all_ax[j][plt_pos]
                    ax.imshow(radial_mask(this_stamp, box_size), interpolation = 'none', cmap = stamp_cm)
                    ax.set_title(name+' #'+str(j), fontsize = 10)
                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])
                stamp[:,:] += this_stamp
            stamp[:,:] /= 4.0

        if save_gif is True:
            #ax = plt.subplot(4, 3, plt_pos[1])
            ax = all_ax[plt_pos][2] #Location of average image
            cb = ax.imshow(radial_mask(stamp, box_size), interpolation = 'none', cmap = stamp_cm)
            cb = fig.colorbar(cb, ax=ax)
            cb.ax.tick_params(labelsize = 8)
            ax.set_title('Average '+name, fontsize = 10)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

    #Compute scale factor here
    scales, offset_value, dx, dy, shifted_stamp1 = find_scale(stamp1, stamp2, box_size, nudgexy = nudgexy, offset = offset)
    if nudgexy is True:
        stamp1[:,:] = shifted_stamp1

    if save_gif is True:
        #ax = plt.subplot(4, 3, 9)
        ax = all_ax[2][2]
        cb = ax.imshow(radial_mask(stamp1*scales, box_size), interpolation = 'none', cmap = stamp_cm)
        cb = fig.colorbar(cb, ax = ax)
        cb.ax.tick_params(labelsize = 8)
        ax.set_title('scale: '+str("{0:.4f}".format(scales))+" offset: "+str("{0:.4f}".format(offset_value)), fontsize=9)

        if nudgexy is True:
            dx_str = '%0.3f' % (dx)
            dy_str = '%0.3f' % (dy)
            ax.annotate('dx = '+dx_str, xy=(0.2,0.85), xycoords='axes fraction', fontsize=8)
            ax.annotate('dy = '+dy_str, xy=(0.2,0.80), xycoords='axes fraction', fontsize=8)

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        #ax = plt.subplot(4, 3, 12)
        ax = all_ax[3][2]
        cb = ax.imshow(radial_mask(((stamp1*scales) - (stamp2-offset_value))/np.nanmax(stamp1*scales), box_size), interpolation = 'none', cmap = 'bwr', vmin = -0.1, vmax = 0.1)
        cb = fig.colorbar(cb, ax = ax)

        cb.ax.tick_params(labelsize = 8)
        ax.set_title('Fract. Resid.', fontsize=10)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        fig.subplots_adjust(wspace=0.10, hspace=0.15)

        if order == 1 :
            order_path = "Figures/"
        else:
            order_path = "Figures/2nd_order/"

        str_box = 's'+str(box_size).zfill(2)
        str_hp = 'hp'+str(high_pass).zfill(2)
        if nudgexy is True:
            str_xy = 'nudge1'
        else:
            str_xy = 'nudge0'
        if offset is True:
            str_offset = 'offset1'
        else:
            str_offset = 'offset0'

        plt.savefig(path+order_path+'Frames-'+str_box+'-'+str_hp+'-'+str_xy+'-'+str_offset+'-'+base_name.replace('.fits','')+'-'+str(i).zfill(2)+'.png', dpi = 100, bbox_inches='tight')
        plt.close('all')

    #Calculate mean of residuals here (currently using sum of absolute residuals)
    residuals = np.nanmean(radial_mask(np.abs((stamp1*scales) - (stamp2-offset_value)), box_size))

    return index, slice, scales, residuals

def find_scale(im1, im2, box_size, nudgexy = False, offset = False):

    if (np.size(im1) != np.size(im2)):
        print ('Stamps do not have same dimensions')
        return 0

    if nudgexy is False:
        if offset is False:
            guess = np.nanmax(im2) / np.nanmax(im1)
            result = optimize.minimize(minimize_psf, guess, args=(radial_mask(im1, box_size), radial_mask(im2, box_size), box_size, nudgexy, offset), method = 'Nelder-Mead', options = {'maxiter': int(1e6) ,'maxfev': int(1e6)})
            if result.status != 0:
                print(result.message)
            scale = result.x[0]
            offset_value = 0.0
            dx = 0.0
            dy = 0.0
            shifted_im1 = np.copy(im1)

        else:
            guess_scale = np.nanmax(im2) / np.nanmax(im1)
            guess_offset = (-1.0)*np.nanmean((guess_scale*im1) - im2)
            guess = (guess_scale, guess_offset)
            result = optimize.minimize(minimize_psf, guess, args=(radial_mask(im1, box_size), radial_mask(im2, box_size), box_size, nudgexy, offset), method = 'Nelder-Mead', options = {'maxiter': int(1e6) ,'maxfev': int(1e6)})
            if result.status != 0:
                print(result.message)
            scale = result.x[0]
            offset_value = result.x[1]
            dx = 0.0
            dy = 0.0
            shifted_im1 = np.copy(im1)
    else:
        #Don't worry about this part - not actually useful!
        if offset is False:
            guess = (np.nanmax(im2) / np.nanmax(im1), 0.0, 0.0)
            #result = optimize.minimize(minimize_psf, guess, args=(im1, radial_mask(im2, box_size), box_size, nudgexy), bounds = ((guess[0]*0.01, guess[0]*100.), (-0.5, 0.5), (-0.5, 0.5)), method = 'SLSQP') 
            result = optimize.minimize(minimize_psf, guess, args=(im1, radial_mask(im2, box_size), box_size, nudgexy), method = 'Nelder-Mead', options = {'maxiter': int(1e6) ,'maxfev': int(1e6)}) 
            scale = result.x[0]
            dx = result.x[1]
            dy = result.x[2]
            offset_value = 0.0

            x, y = gen_xy(box_size + 4)
            x += dx
            y += dy
            shifted_im1 = ndimage.map_coordinates(im1, (y, x), cval = np.nan)
        else:
            guess = (np.nanmax(im2) / np.nanmax(im1), 0.0, 0.0, (-1.0)*np.nanmean((guess_scale*im1) - im2))
            #result = optimize.minimize(minimize_psf, guess, args=(im1, radial_mask(im2, box_size), box_size, nudgexy), bounds = ((guess[0]*0.01, guess[0]*100.), (-0.5, 0.5), (-0.5, 0.5)), method = 'SLSQP') 
            result = optimize.minimize(minimize_psf, guess, args=(im1, radial_mask(im2, box_size), box_size, nudgexy), method = 'Nelder-Mead', options = {'maxiter': int(1e6) ,'maxfev': int(1e6)}) 
            scale = result.x[0]
            dx = result.x[1]
            dy = result.x[2]
            offset_value = result.x[3]

            x, y = gen_xy(box_size + 4)
            x += dx
            y += dy
            shifted_im1 = ndimage.map_coordinates(im1, (y, x), cval = np.nan)


    return scale, offset_value, dx, dy, shifted_im1


def minimize_psf(p, im1, im2, box_size, nudgexy, offset): 
    """ Simply minimize residuals
    Args:
        scale - scale factor 
        ave_dm - average dm for a given slice
        ave_sat - average sat for a given slice 
    return:
        residuals for ave_dm and ave_sat
    """

    if nudgexy is False:
        if offset is True:
            return np.nansum(np.abs(((p[0]*im1) - (im2-p[1]))))
        else:
            return np.nansum(np.abs(((p*im1) - im2)))
    else:
        if offset is True:
            #Don't worry about this part - not actually useful!
            x, y = gen_xy(box_size + 4)
            x += p[1]
            y += p[2]
            shifted_im1 = ndimage.map_coordinates(im1, (y, x), cval = np.nan)

            return np.nansum(np.abs(((p[0]*shifted_im1) - (im2-p[3]))))

        else:
            #Don't worry about this part - not actually useful!
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

    #For debugging
    # if result.success is False:
    #     fits.writeto('test_im.fits', im, clobber=True)

    #     p  =p0  
    #     a = 1.0 / (2.0 * p[3]**2.0)
    #     b = 1.0 / (2.0 * p[3]**2.0)
    #     g =  p[0]*np.exp( -((a*((x-p[1])**2)) + (b*((y-p[2])**2))))
    #     fits.writeto('test_model.fits', g, clobber=True)

    #     print p0
    #     print np.min(x), np.min(y)
    #     print result
    
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
    if filtersize == 0:
        return img
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

def convert_gif(path, name, str_box, str_hp, str_xy, str_offset, order):
    if order == 1 :
        order_path = "Figures/"
    else:
        order_path = "Figures/2nd_order/"

    os.system('convert -delay 25 -loop 0 '+path+order_path+'Frames-'+str_box+'-'+str_hp+'-'+str_xy+'-'+str_offset+'-'+os.path.basename(path+name).replace('.fits','')+'-*.png '+path+order_path+'gifs/Animation-'+str_box+'-'+str_hp+'-'+str_xy+'-'+str_offset+'-'+(os.path.basename(path+name)).replace('.fits','')+'.gif')

    return 0

def save_spot_pos(band):

    #Simple function to save satellite spot position guesses
    #Valid for UCSC lab data

    if band == 'J_Hapod':
        # old data pixel coordinates
        #dm_pos0 = np.array([[129, 161], [117, 131], [148, 119], [159, 149]])
        #dm_pos36 = np.array([[127, 165], [113, 129], [149, 115], [163, 151]])
        dm_pos0 = np.array([[120.500,147.500], [142.750,174.750], [148.000,125.000], [170.000,152.250]])
        dm_pos36 = np.array([[116.250,147.500], [142.250,179.250], [148.250,120.750], [174.000,152.250]])



        dm_pos = np.zeros((37, 4, 2), dtype=np.float64)

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(dm_pos36[j, k] - dm_pos0[j, k])/37.0
                    dm_pos[i, j, k] = dm_pos0[j, k] + (delta * float(i))

        dm_pos = dm_pos.astype(int)

        # old data pixel coordinates
        #sat_pos0 = np.array([[97, 156], [124, 99], [182, 126], [154, 183]])
        #sat_pos36 = np.array([[89, 159], [121, 91], [190, 123], [157, 191]])
        sat_pos0 = np.array([[106.840,165.916], [163.666,188.261], [128.944,108.689], [185.850,130.390]])
        sat_pos36 = np.array([[99.526,168.971], [167.042,194.530], [126.050, 102.420], [192.119,127.497]])

        sat_pos = np.zeros((37, 4, 2), dtype=np.float64)

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(sat_pos36[j, k] - sat_pos0[j, k])/37.0
                    sat_pos[i, j, k] = sat_pos0[j, k] + (delta * float(i))

        sat_pos = sat_pos.astype(int)

        # old data pixel coordinates
        #order_2_sat_pos0 = np.array([[54, 171], [109, 56], [224, 111], [169, 226]])
        #order_2_sat_pos36 = np.array([[38, 177], [103, 40], [240, 105], [174, 242]])
        order_2_sat_pos0 = np.array([[67.215,184.403], [181.509,227.323], [110.618,69.144], [225.394,112.547]])
        order_2_sat_pos36 = np.array([[53.712,190.672], [187.779,240.826], [104.831,55.641], [238.897,106.278]])
        order_2_sat_pos = np.zeros((37, 4, 2), dtype=np.float64)

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(order_2_sat_pos36[j, k] - order_2_sat_pos0[j, k])/37.0
                    order_2_sat_pos[i, j, k] = order_2_sat_pos0[j, k] + (delta * float(i))

        order_2_sat_pos = order_2_sat_pos.astype(int)

        fits.writeto('centers_'+band+'_dm.fits', dm_pos, clobber=True)
        fits.writeto('centers_'+band+'_sat.fits', sat_pos, clobber=True)
        fits.writeto('centers_'+band+'_sat2.fits', order_2_sat_pos, clobber=True)

    if band == 'K2_Hapod':

        #Limited wavelength coverage, channel 6 -- 11 only

        dm_pos6 = np.array([[122.12, 178.78], [99.97, 122.1], [156.00, 100.41], [177.53, 156.07]])
        dm_pos11 = np.array([[122.14, 179.02], [99.68, 121.65], [156.39, 99.95], [177.93, 155.91]])
        dm_pos = np.zeros((37, 4, 2), dtype=np.float64)

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(dm_pos11[j, k] - dm_pos6[j, k])/5.0
                    dm_pos[i, j, k] = dm_pos6[j, k] + (delta * float(i-6))

        dm_pos = dm_pos.astype(int)

        sat_pos6 = np.array([[61.925, 171.16], [107.47, 61.227], [216.51, 107.1], [170.77, 216.92]])
        sat_pos11 = np.array([[60.342, 171.31], [107.3, 60.371], [217.38, 106.71], [171.12, 217.76]])
        sat_pos = np.zeros((37, 4, 2), dtype=np.float64)

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(sat_pos11[j, k] - sat_pos6[j, k])/4.0
                    sat_pos[i, j, k] = sat_pos6[j, k] + (delta * float(i-6))

        sat_pos = sat_pos.astype(int)

        fits.writeto('centers_'+band+'_dm.fits', dm_pos, clobber=True)
        fits.writeto('centers_'+band+'_sat.fits', sat_pos, clobber=True)


    if band == 'Y_Hapod':

        # old data pixel coodinates
        #dm_pos0 = np.array([[131, 160], [122, 133], [147, 124], [157, 149]])
        #dm_pos36 = np.array([[130, 162], [118, 133], [148, 121], [160, 151]])
        dm_pos0 = np.array([[126.500,148.00], [145.361,171.525], [150.109,129.759], [168.521,152.435]])
        dm_pos36 = np.array([[123.234,148.106], [145.070,174.722], [150.303,126.271], [171.815,152.822]])
        dm_pos = np.zeros((37, 4, 2), dtype=np.float64)

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(dm_pos36[j, k] - dm_pos0[j, k])/37.0
                    dm_pos[i, j, k] = dm_pos0[j, k] + (delta * float(i))

        dm_pos = dm_pos.astype(int)

        # old data pixel coordinates
        #sat_pos0 = np.array([[103, 156], [125, 106], [175, 127], [154, 177]])
        #sat_pos36 = np.array([[97, 159], [122, 100], [181, 124], [156, 183]])
        sat_pos0 = np.array([[113.512,163.184], [161.255,181.509], [131.837,114.476], [171.580,133.284]])
        sat_pos36 = np.array([[107.725,166.077], [164.148,187.296], [129.426,109.171], [185.367,130.873]])
        sat_pos = np.zeros((37, 4, 2), dtype=np.float64)

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(sat_pos36[j, k] - sat_pos0[j, k])/37.0
                    sat_pos[i, j, k] = sat_pos0[j, k] + (delta * float(i))

        sat_pos = sat_pos.astype(int)

        # old data pixel coodinates
        #order_2_sat_pos0 = np.array([[68, 170], [111, 70], [210, 113], [168, 213]])
        #order_2_sat_pos36 = np.array([[56, 175], [105, 58], [223, 107], [173, 225]])
        order_2_sat_pos0 = np.array([[80.236,178.616], [175.722,214.785], [116.887,81.683], [213.338,117.852]])
        order_2_sat_pos36 = np.array([[69.144,183.921], [181.509,226.841], [112.065,70.591], [224.912,113.029]])
        order_2_sat_pos = np.zeros((37, 4, 2), dtype=np.float64)

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(order_2_sat_pos36[j, k] - order_2_sat_pos0[j, k])/37.0
                    order_2_sat_pos[i, j, k] = order_2_sat_pos0[j, k] + (delta * float(i))

        order_2_sat_pos = order_2_sat_pos.astype(int)

        fits.writeto('centers_'+band+'_dm.fits', dm_pos, clobber=True)
        fits.writeto('centers_'+band+'_sat.fits', sat_pos, clobber=True)
        fits.writeto('centers_'+band+'_sat2.fits', order_2_sat_pos, clobber=True)

    if band == 'H_Hapod':

        dm_pos0 = np.array([[112.750,144.750], [143.000,181.175], [149.500,114.500], [179.750,151.250]])
        dm_pos36 = np.array([[107.750,144.500], [142.250,187.000], [150.000,109.000], [185.500,153.250]])
        dm_pos = np.zeros((37, 4, 2), dtype=np.float64)

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(dm_pos36[j, k] - dm_pos0[j, k])/37.0
                    dm_pos[i, j, k] = dm_pos0[j, k] + (delta * float(i))

        dm_pos = dm_pos.astype(int)

        sat_pos0 = np.array([[93.778,171.208], [169.200,201.000], [122.400,95.400], [198.000,124.200]])
        sat_pos36 = np.array([[84.000,175.800], [173.400,209.400], [118.800,86.400], [207.600,120.000]])
        sat_pos = np.zeros((37, 4, 2), dtype=np.float64)

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(sat_pos36[j, k] - sat_pos0[j, k])/37.0
                    sat_pos[i, j, k] = sat_pos0[j, k] + (delta * float(i))

        sat_pos = sat_pos.astype(int)


        fits.writeto('centers_'+band+'_dm.fits', dm_pos, clobber=True)
        fits.writeto('centers_'+band+'_sat.fits', sat_pos, clobber=True)
    if band == 'K1_Hapod':

        dm_pos0 = np.array([[103.000,143.500], [139.000,189.000], [148.000,106.000], [185.000,152.000]])
        dm_pos36 = np.array([[101.500,143.500], [139.500,189.500], [148.00,105.500], [186.000,151.500]])
        dm_pos = np.zeros((37, 4, 2), dtype=np.float64)
        #print(np.shape(dm_pos0))
        #print(np.shape(dm_pos36))

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(dm_pos36[j, k] - dm_pos0[j, k])/37.0
                    dm_pos[i, j, k] = dm_pos0[j, k] + (delta * float(i))

        dm_pos = dm_pos.astype(int)

        sat_pos0 = np.array([[78.789,176.687], [172.829,212.856], [114.958,82.165], [208.998,117.852]])
        sat_pos36 = np.array([[77.343,178.133], [173.793,214.785], [114.476,81.201], [210.927,117.370]])
        sat_pos = np.zeros((37, 4, 2), dtype=np.float64)
        print(np.shape(sat_pos0))
        print(np.shape(sat_pos36))

        for i in range(0, 37):
            for j in range(0, 4):
                for k in range(0, 2):
                    delta = float(sat_pos36[j, k] - sat_pos0[j, k])/37.0
                    sat_pos[i, j, k] = sat_pos0[j, k] + (delta * float(i))

        sat_pos = sat_pos.astype(int)


        fits.writeto('centers_'+band+'_dm.fits', dm_pos, clobber=True)
        fits.writeto('centers_'+band+'_sat.fits', sat_pos, clobber=True)



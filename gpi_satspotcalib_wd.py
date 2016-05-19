#Function to analyse GPI satellite spot data

import glob
import numpy as np
from astropy.io import fits
from astropy.io import ascii
from astropy.table import hstack
from pyklip import klip
from scipy import optimize
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import photutils
from derosa.sed import convolve
from scipy.io import readsav
import multiprocessing as mp
from scipy import ndimage


cmap = 'inferno'

def aperture_phot(im, xy, sat_xy, star_xy, wl, app = 1.5, median_filter = False, mask_size = 7.5, wedge_sky = False, fig = False):
    # im - slice to be analyzed
    # xy - 2 tuple of WD coordinates
    # sat_xy - 2,4 tuple of sat spot coordinates
    # wl - wavelenght of slice in microns
    # app - defines aperture size in l/d (calculated below)
    # median_filter - if not False defines size of median filter box.
    # mask_size - pixel size of mask for WD/sat spots when performing median filter
    # wedge sky - if False, compute sky in annulus, if True compute sky in two wedges either side of source
    # fig - if not False, pass figure object for plotting

    gemini_d = 7.7701
    ld = (((wl * 1e-6) / gemini_d) * 206271.0) / 0.014166

    sky_app_in = app + 1.0
    sky_app_out = sky_app_in + 1.0

    if median_filter is not False:
        #First mask all objects using a mask of radius mask_size
        old_im = np.copy(im)
        tmp_im = np.copy(im)
        s = np.shape(im)
        xarr, yarr = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
        rarr = np.sqrt((xarr - xy[0])**2.0 + (yarr - xy[1])**2.0)
        ind = np.where(rarr < mask_size)
        tmp_im[ind] = np.nan

        for i in xrange(0, 4):
            rarr = np.sqrt((xarr - sat_xy[i][0])**2.0 + (yarr - sat_xy[i][1])**2.0)
            ind = np.where(rarr < mask_size)
            tmp_im[ind] = np.nan

        V = np.copy(tmp_im)
        V[tmp_im != tmp_im] = 0
        VV = ndimage.gaussian_filter(V, sigma = 5.0)
        W = (np.copy(V) * 0.0) + 1.0
        W[tmp_im != tmp_im] = 0
        WW = ndimage.gaussian_filter(W, sigma = 5.0)
        filt_im = VV/WW


        #filt_im = ndimage.filters.generic_filter(tmp_im, np.nanmedian, size=median_filter)
        im -= filt_im

    #Perform photometry
    aperture = photutils.CircularAperture((xy[0], xy[1]), r = app * ld)
    sky_aperture = photutils.CircularAnnulus((xy[0], xy[1]), r_in = sky_app_in * ld, r_out=sky_app_out * ld)
    phot_table = hstack([photutils.aperture_photometry(im, aperture), photutils.aperture_photometry(im, sky_aperture)], table_names=['src', 'sky'])
    sky_mean = phot_table['aperture_sum_sky'] / sky_aperture.area()
    wd_phot = phot_table['aperture_sum_src'].data[0]
    wd_sky = sky_mean.data[0]

    #Actually going to compute the sky ourselves since we would ideally want to use a median
    s = np.shape(im)
    xarr, yarr = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
    rarr = np.sqrt((xarr - xy[0])**2.0 + (yarr - xy[1])**2.0)
    ind = np.where((rarr >= (sky_app_in * ld)) & (rarr <= (sky_app_out*ld)))
    wd_sky = np.nanmedian(im[ind])


    dr = (np.pi/180.0)

    if wedge_sky is True:
        old_wd_sky = np.copy(wd_sky)
        wd_sky = 0.0
        #Calculate within an annulus of width 1.5 l/d from either redge of the photometry aperture pm10 deg
        sep = np.sqrt((xy[0] - star_xy[0])**2.0 + (xy[1] - star_xy[0])**2.0)
        pa = np.arctan2((xy[1] - star_xy[1]), (xy[0] - star_xy[0]))
        #(app*l/d) divided by sep gives the difference in PA between the source and the edge of the annulus
        s = np.shape(im)
        xarr, yarr = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
        rarr = np.sqrt((xarr - star_xy[0])**2.0 + (yarr - star_xy[1])**2.0)
        tarr = np.arctan2(yarr-star_xy[1], xarr-star_xy[0])
        ind = np.where((rarr >= (sep - 0.75*ld)) & (rarr <= (sep + 0.75*ld)) & ((tarr >= (pa - ((app*ld)/sep) - 7.5*dr)) & (tarr <= (pa - (app*ld)/sep)) | (tarr <= (pa + ((app*ld)/sep) + 7.5*dr)) & (tarr >= (pa + (app*ld)/sep))))
        wd_sky = np.nanmean(im[ind])
        mark_xarr = xarr[ind]
        mark_yarr = yarr[ind]
    

    wd_phot_nosky = np.copy(wd_phot)
    if median_filter is False:
        wd_phot -= (wd_sky * aperture.area())


    #Plot WD stamp with annuli
    if median_filter is not False:
        ax = fig.add_subplot(6, 4, 1)
        stamp = old_im[(np.floor(xy[1])-20):(np.floor(xy[1])+21), (np.floor(xy[0])-20):(np.floor(xy[0])+21)]
        vmin = np.nanmin(stamp)
        vmax = np.nanmax(stamp)
        ax.imshow(stamp, aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap, vmin = vmin, vmax = vmax)
        ax = fig.add_subplot(6, 4, 2)
        stamp = filt_im[(np.floor(xy[1])-20):(np.floor(xy[1])+21), (np.floor(xy[0])-20):(np.floor(xy[0])+21)]
        ax.imshow(stamp, aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap, vmin = vmin, vmax = vmax)
        
    if median_filter is not False:
        ax = fig.add_subplot(6, 4, 3)
    else:
        ax = fig.add_subplot(6, 2, 1)

    stamp = im[(np.floor(xy[1])-20):(np.floor(xy[1])+21), (np.floor(xy[0])-20):(np.floor(xy[0])+21)]
    ax.imshow(stamp, aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap)
    ax.autoscale(False)
    if wedge_sky is True:
        ax.plot(mark_xarr - (np.floor(xy[0])-20), mark_yarr - (np.floor(xy[1])-20), '.', markeredgecolor='None', markersize=3, color='white', alpha=0.5)

    ax.add_patch(Circle((xy[0] - (np.floor(xy[0])-20), xy[1] - (np.floor(xy[1])-20)), app * ld, fill=False, color='white', linewidth=1.5))
    ax.add_patch(Circle((xy[0] - (np.floor(xy[0])-20), xy[1] - (np.floor(xy[1])-20)), sky_app_in * ld, fill=False, color='white', linewidth=1.5, linestyle='--'))
    ax.add_patch(Circle((xy[0] - (np.floor(xy[0])-20), xy[1] - (np.floor(xy[1])-20)), sky_app_out * ld, fill=False, color='white', linewidth=1.5, linestyle='--'))

    #Now plot radial profile
    s = np.shape(im)
    xarr, yarr = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
    rarr = np.sqrt((xarr - xy[0])**2.0 + (yarr - xy[1])**2.0)

    if median_filter is not False:
        ax = fig.add_subplot(6, 4, 4)
    else:
        ax = fig.add_subplot(6, 2, 2)
    ax.plot(rarr, im, '.', markersize=1, color='k')
    ax.annotate('wd_phot_nosky = %8.2f' % wd_phot_nosky, xy=(0.1, 0.8), xycoords='axes fraction', fontsize=8)
    ax.annotate('sky_phot (per px) = %8.2f' % wd_sky, xy=(0.1, 0.725), xycoords='axes fraction', fontsize=8)
    ax.annotate('phot area (px) = %8.2f (r = %8.2f)' % (aperture.area(), app*ld), xy=(0.1, 0.65), xycoords='axes fraction', fontsize=8)
    ax.annotate('wd_phot = %8.2f' % wd_phot, xy=(0.1, 0.575), xycoords='axes fraction', fontsize=8)


    if wedge_sky is True:
        ax.plot(np.mean((sky_app_in*ld, sky_app_out*ld)), old_wd_sky, '*', color='green', markersize = 10)
    ax.plot(np.mean((sky_app_in*ld, sky_app_out*ld)), wd_sky, '*', color='red', markersize = 10)

    #Now over-plot apertures
    ax.plot((app * ld, app * ld), [-1e5, 1e5], '-', color='blue')
    ax.plot((sky_app_in * ld, sky_app_in * ld), [-1e5, 1e5], '--', color='red')
    ax.plot((sky_app_out * ld, sky_app_out * ld), [-1e5, 1e5], '--', color='red')

    ax.set_xlim(0, 25)
    ax.set_ylim(np.nanmin(stamp), np.nanmax(stamp)*1.1)

    sat_phot = np.zeros(4, dtype=np.float64) * np.nan
    sat_sky = np.zeros(4, dtype=np.float64) * np.nan

    for i in xrange(0, 4):

        #Perform photometry
        aperture = photutils.CircularAperture((sat_xy[i][0], sat_xy[i][1]), r = app * ld)
        sky_aperture = photutils.CircularAnnulus((sat_xy[i][0], sat_xy[i][1]), r_in = sky_app_in * ld, r_out=sky_app_out * ld)
        phot_table = hstack([photutils.aperture_photometry(im, aperture), photutils.aperture_photometry(im, sky_aperture)], table_names=['src', 'sky'])
        sky_mean = phot_table['aperture_sum_sky'] / sky_aperture.area()
        phot = phot_table['aperture_sum_src'].data[0]
        sky = sky_mean.data[0]

        #Actually going to compute the sky ourselves since we would ideally want to use a median
        s = np.shape(im)
        xarr, yarr = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
        rarr = np.sqrt((xarr - sat_xy[i][0])**2.0 + (yarr - sat_xy[i][1])**2.0)
        ind = np.where((rarr >= (sky_app_in * ld)) & (rarr <= (sky_app_out*ld)))
        sky = np.nanmedian(im[ind])

        if wedge_sky is True:
            old_sky = np.copy(sky)
            sky = 0.0
            #Calculate within an annulus of width 1.5 l/d from either redge of the photometry aperture pm10 deg
            sep = np.sqrt((sat_xy[i][0] - star_xy[0])**2.0 + (sat_xy[i][1] - star_xy[1])**2.0)
            pa = np.arctan2((sat_xy[i][1] - star_xy[1]), (sat_xy[i][0] - star_xy[0]))
            #(app*l/d) divided by sep gives the difference in PA between the source and the edge of the annulus
            s = np.shape(im)
            xarr, yarr = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
            rarr = np.sqrt((xarr - star_xy[0])**2.0 + (yarr - star_xy[1])**2.0)
            tarr = np.arctan2(yarr-star_xy[1], xarr-star_xy[0])
            ind = np.where((rarr >= (sep - 0.75*ld)) & (rarr <= (sep + 0.75*ld)) & ((tarr >= (pa - ((app*ld)/sep) - 7.5*dr)) & (tarr <= (pa - (app*ld)/sep)) | (tarr <= (pa + ((app*ld)/sep) + 7.5*dr)) & (tarr >= (pa + (app*ld)/sep))))
            sky = np.nanmean(im[ind])
            mark_xarr = xarr[ind]
            mark_yarr = yarr[ind]
    
        phot_nosky = np.copy(phot)
        if median_filter is False:
            phot -= (sky * aperture.area())

        sat_phot[i] = phot
        sat_sky[i] = sky


        if median_filter is not False:
            ax = fig.add_subplot(6, 4, 5+(i*4))
            stamp = old_im[(np.floor(float(sat_xy[i][1]))-20):(np.floor(float(sat_xy[i][1]))+21), (np.floor(float(sat_xy[i][0]))-20):(np.floor(float(sat_xy[i][0]))+21)]
            vmin = np.nanmin(stamp)
            vmax = np.nanmax(stamp)    
            ax.imshow(stamp, aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap, vmin = vmin, vmax = vmax)
            ax = fig.add_subplot(6, 4, 6+(i*4))
            stamp = filt_im[(np.floor(float(sat_xy[i][1]))-20):(np.floor(float(sat_xy[i][1]))+21), (np.floor(float(sat_xy[i][0]))-20):(np.floor(float(sat_xy[i][0]))+21)]
            ax.imshow(stamp, aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap, vmin = vmin, vmax = vmax)
            
        if median_filter is not False:
            ax = fig.add_subplot(6, 4, 7+(i*4))
        else:
            ax = fig.add_subplot(6, 2, 3+(i*2))

        stamp = im[(np.floor(float(sat_xy[i][1]))-20):(np.floor(float(sat_xy[i][1]))+21), (np.floor(float(sat_xy[i][0]))-20):(np.floor(float(sat_xy[i][0]))+21)]
        ax.imshow(stamp, aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap)
        ax.autoscale(False)
        if wedge_sky is True:
            ax.plot(mark_xarr - (np.floor(sat_xy[i][0])-20), mark_yarr - (np.floor(sat_xy[i][1])-20), '.', markeredgecolor='None', markersize=3, color='white', alpha=0.5)
        ax.add_patch(Circle((sat_xy[i][0] - (np.floor(sat_xy[i][0])-20), sat_xy[i][1] - (np.floor(sat_xy[i][1])-20)), app * ld, fill=False, color='white', linewidth=1.5))
        ax.add_patch(Circle((sat_xy[i][0] - (np.floor(sat_xy[i][0])-20), sat_xy[i][1] - (np.floor(sat_xy[i][1])-20)), sky_app_in * ld, fill=False, color='white', linewidth=1.5, linestyle='--'))
        ax.add_patch(Circle((sat_xy[i][0] - (np.floor(sat_xy[i][0])-20), sat_xy[i][1] - (np.floor(sat_xy[i][1])-20)), sky_app_out * ld, fill=False, color='white', linewidth=1.5, linestyle='--'))

        s = np.shape(im)
        xarr, yarr = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
        rarr = np.sqrt((xarr - sat_xy[i][0])**2.0 + (yarr - sat_xy[i][1])**2.0)

        if median_filter is not False:
            ax = fig.add_subplot(6, 4, 8+(i*4))
        else:
            ax = fig.add_subplot(6, 2, 4+(i*2))    

        ax.plot(rarr, im, '.', markersize=1, color='k')
        ax.annotate('sat_phot_nosky = %8.2f' % phot_nosky, xy=(0.1, 0.8), xycoords='axes fraction', fontsize=8)
        ax.annotate('sky_phot (per px) = %8.2f' % sky, xy=(0.1, 0.725), xycoords='axes fraction', fontsize=8)
        ax.annotate('sat_phot = %8.2f' % phot, xy=(0.1, 0.650), xycoords='axes fraction', fontsize=8)

        if wedge_sky is True:
            ax.plot(np.mean((sky_app_ind*ld, sky_app_out*ld)), old_sky, '*', color='green', markersize = 10)
        ax.plot(np.mean((sky_app_in*ld, sky_app_out*ld)), sky, '*', color='red', markersize = 10)

        #Now over-plot apertures
        ax.plot((app * ld, app * ld), [-1e5, 1e5], '-', color='blue')
        ax.plot((sky_app_in * ld, sky_app_in * ld), [-1e5, 1e5], '--', color='red')
        ax.plot((sky_app_out * ld, sky_app_out * ld), [-1e5, 1e5], '--', color='red')
        
        ax.set_xlim(0, 25)
        ax.set_ylim(np.nanmin(stamp), np.nanmax(stamp)*1.1)


    return wd_phot, wd_sky, sat_phot, sat_sky

def return_pos(im, xy_guess):

    #Fit WD location in slice from guess
    pad = 10
    stamp = im[xy_guess[1]-pad:xy_guess[1]+pad, xy_guess[0]-pad:xy_guess[0]+pad]
    stamp[np.isnan(stamp)] = np.nanmedian(stamp)
    p0 = [np.nanmax(stamp), pad, pad, 3.0, 0.0]
    s = np.shape(stamp)
    x, y = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
    popt, pcov = optimize.curve_fit(twoD_Gaussian, (x, y), stamp.ravel(), p0 = p0)
    xy = [popt[1] + (xy_guess[0] - pad), popt[2] + (xy_guess[1] - pad)]    

    return xy

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma, offset):
    xo = float(xo)
    yo = float(yo)    
    a = 1.0 / (2.0 * sigma**2.0)
    b = 1.0 / (2.0 * sigma**2.0)
    g = offset + amplitude*np.exp( - ((a*((x-xo)**2)) + (b*((y-yo)**2))))
    
    return g.ravel()

def minimize(p, star, err, model):
    #Compute chi2
    return np.nansum(((star - (p*model)) / err)**2.0)

def minimize_psf(p, wd, sat):
    #Simply minimize residuals
    return np.nansum(np.abs(((p*wd) - sat)))

#Code to handle multiprocessing
def calc(i, k, list, app, wd_x, wd_y, wedge, median_filter, mask_size):

    cube = fits.getdata(list[i])
    header = fits.getheader(list[i], 1)

    slice = cube[k, :, :]
    slice_hp = klip.high_pass_filter(slice)

    wd_xy = return_pos(slice_hp, (wd_x[i], wd_y[i]))
    sat_xy = [map(float, header['SATS'+str(k)+'_'+str(j)].lstrip().split()) for j in xrange(0, 4)]
    star_xy = map(float, header['PSFC_'+str(k)].lstrip().split())

    wl = header['CRVAL3'] + (header['CD3_3'] * float(k))

    if median_filter is not False:
        fig = plt.figure(figsize=(14, 16))
    else:
        fig = plt.figure(figsize=(7, 16))

    result = aperture_phot(slice, wd_xy, sat_xy, star_xy, wl, app = app, median_filter = median_filter, mask_size = mask_size, wedge_sky = wedge, fig = fig)

    if wedge is True:
        sky_prefix = 'skywedge'
    else:
        sky_prefix = 'skyannulus'

    if median_filter is not False:
        filter_prefix = 'filter'+str(median_filter)+'px'
    else:
        filter_prefix = 'nofilter'

    fig.savefig('Figs_app/Diag/WD_stamps_'+sky_prefix+'_'+filter_prefix+'_'+'_app'+str(app)+'_'+str(i)+'_'+str(k)+'.pdf', bbox_inches='tight')
    plt.close('all')

    print result[0], result[2], np.mean(result[2])

    return i, k, result[0], np.mean(result[2])

def calc_psf(i, k, list, wd_x, wd_y, filtersize):

    cube = fits.getdata(list[i])
    header = fits.getheader(list[i], 1)

    slice = cube[k, :, :]
    slice_hp = klip.high_pass_filter(slice, filtersize=10)

    wd_xy = return_pos(slice_hp, (wd_x[i], wd_y[i]))
    sat_xy = [map(float, header['SATS'+str(k)+'_'+str(j)].lstrip().split()) for j in xrange(0, 4)]
    star_xy = map(float, header['PSFC_'+str(k)].lstrip().split())

    #Mask star
    slice = cube[k, :, :]
    s = np.shape(slice)
    xarr, yarr = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
    xarr -= star_xy[0]
    yarr -= star_xy[1]
    rarr = np.sqrt((xarr**2.0) + (yarr**2.0))
    ind = np.where(rarr < 20.0)
    #slice[ind] = np.nan

    slice_hp = klip.high_pass_filter(slice, filtersize=filtersize)

    fits.writeto('Figs_psf/Fits/filtered_filtersize'+str(filtersize)+'_'+str(i)+'_'+str(k)+'.fits', slice_hp, clobber=True)


    wl = header['CRVAL3'] + (header['CD3_3'] * float(k))

    #Now extract a stamp for each of the sat spots
    stamps = np.zeros((4, 50, 50), dtype = np.float64)
    s = [50, 50]

    fig = plt.figure(figsize = (10, 16))

    gemini_d = 7.7701
    ld = (((wl * 1e-6) / gemini_d) * 206271.0) / 0.014166

    for j in xrange(0, 4):

        xarr, yarr = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
        xarr -= 25.0
        yarr -= 25.0
        rarr = np.sqrt((xarr*xarr) + (yarr*yarr))
        xarr += sat_xy[j][0]
        yarr += sat_xy[j][1]

        tmp_slice = np.copy(slice)
        tmp_slice[np.isnan(tmp_slice)] = np.nanmedian(tmp_slice)
        this_stamp = ndimage.map_coordinates(tmp_slice, (yarr, xarr), cval = np.nan)

        ax = fig.add_subplot(6, 3, 1 + (j*3))
        ax.imshow(this_stamp, aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap)
        ax.set_title('sat #'+str(j))

        tmp_slice_hp = np.copy(slice_hp)
        tmp_slice_hp[np.isnan(tmp_slice_hp)] = np.nanmedian(tmp_slice_hp)
        this_stamp = ndimage.map_coordinates(tmp_slice_hp, (yarr, xarr), cval = np.nan)

        ax = fig.add_subplot(6, 3, 2 + (j*3))
        ax.imshow(this_stamp, aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap)
        ax.set_title('sat #'+str(j)+' (filt)')

        stamps[j,:,:] = this_stamp


    xarr, yarr = np.meshgrid(np.arange(s[1], dtype=np.float64), np.arange(s[0], dtype=np.float64))
    xarr -= 25.0
    yarr -= 25.0
    rarr = np.sqrt((xarr*xarr) + (yarr*yarr))
    ind = np.where(rarr > (1.5*ld))
    xarr += wd_xy[0]
    yarr += wd_xy[1]

    tmp_slice = np.copy(slice)
    tmp_slice[np.isnan(tmp_slice)] = np.nanmedian(tmp_slice)
    wd_stamp = ndimage.map_coordinates(tmp_slice, (yarr, xarr), cval = np.nan)

    ax = fig.add_subplot(6, 3, 3)
    im = ax.imshow(np.copy(wd_stamp), aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap)
    fig.colorbar(im)
    ax.set_title('WD')

    tmp_slice_hp = np.copy(slice_hp)
    tmp_slice_hp[np.isnan(tmp_slice_hp)] = np.nanmedian(tmp_slice_hp)
    wd_stamp_hp = ndimage.map_coordinates(tmp_slice_hp, (yarr, xarr), cval = np.nan)

    ax = fig.add_subplot(6, 3, 6)
    im = ax.imshow(np.copy(wd_stamp_hp), aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap)
    fig.colorbar(im)
    ax.set_title('WD (filt)')

    #Show the averaged sat spot
    ax = fig.add_subplot(6, 3, 9)
    avg_sat = np.nanmean(stamps, axis=0)
    im = ax.imshow(np.copy(avg_sat), aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap)
    fig.colorbar(im)
    ax.set_title('avg (sats)')

    wd_stamp_hp[ind] = np.nan
    avg_sat[ind] = np.nan

    guess =  np.nanmax(avg_sat) / np.nanmax(wd_stamp_hp)
    result = optimize.minimize(minimize_psf, guess, args=(wd_stamp_hp, avg_sat))
    scale = result.x[0]
    
    ax = fig.add_subplot(6, 3, 12)
    im = ax.imshow((wd_stamp_hp * scale) - avg_sat, aspect = 'auto', origin='lower', interpolation='nearest', cmap = cmap)
    fig.colorbar(im)


    ax.set_title('residuals')

    fig.savefig('Figs_psf/Diag/WD_stamps_filter'+str(filtersize)+'_'+str(i)+'_'+str(k)+'.pdf', bbox_inches='tight')
    plt.close('all')

    print i, k, scale

    return i, k, scale

def main():

    #Routine which plots results
    list = sorted(glob.glob('OldReduction/S*.fits'))
    n = len(list)

    band = 'H'
    header = fits.getheader(list[0], 1)
    wl = header['CRVAL3'] + (header['CD3_3'] * np.arange(37, dtype=np.float64))
    dwl = convolve.spectral_resolution(wl)

    #Load WD spectrum
    wd = readsav('HD8049_SINFONI_HK.sav')
    wd_wl = wd.wd_wl
    wd_fl = wd.wd_fl

    wd_fl *= 1.030428416E-21

    wd_wl = wd_wl[90:len(wd_wl)]
    wd_fl = wd_fl[90:len(wd_fl)]
    wd_dwl = convolve.spectral_resolution(wd_wl)

    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(wd_wl, wd_fl)
    ax1.set_xlabel(r'Wavelength (um)')
    ax1.set_ylabel(r'$F_{\lambda}$')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(wd_wl, wd_dwl)
    ax2.set_xlabel(r'Wavelength (um)')
    ax2.set_ylabel(r'$\Delta \lambda$ (um)')

    #Now use apparent magnitude in Crepp to normalize
    filters = ['MKO_H']
    filter_path = '/export/big_scr7/derosa/SED_Generation/Filters/Processed/'
    
    for i in xrange(0, len(filters)):
        filt = readsav(filter_path+filters[i]+'.sav')
        filter_wl = filt.filter_wl
        filter_fl = filt.filter_fl
        filter_dwl = filt.filter_dwl
        filter_zp = filt.filter_zp

        wl_convol, wd_convol, filter_convol = convolve.convol_and_interp(wd_wl, wd_fl, wd_dwl, filter_wl, filter_fl, filter_dwl)
        top = np.trapz(wd_convol * filter_convol, wl_convol)
        bottom = np.trapz(filter_convol, wl_convol)

        print -2.5*np.log10((top/bottom)/filter_zp)

    
    #Now convert down to GPI resolution
    wd_wl, wd_fl, foo = convolve.convol_and_interp(wd_wl, wd_fl, wd_dwl, wl, wl*0.0, dwl)

    ax1.plot(wd_wl, wd_fl, '-o', color='red')

    fig.savefig('Figs_app/WD_dwl.pdf', bbox_inches='tight')
    plt.close('all')

    #Now similarly for the star
    model_name = '/export/big_scr7/derosa/Model_Spectra/Extraction/BTNextGen-AGSS2009/BTNextGen-AGSS2009-5100-p45-p00-p00.sav'
    model = readsav(model_name)
    model_wl = model.model_wl
    model_fl = model.model_fl
    model_dwl = convolve.spectral_resolution(model_wl)

    #Find scaling factor to best fit 2MASS photo
    filters = ['2MASS_J', '2MASS_H', '2MASS_Ks']
    n_filt = len(filters)
    mags = [7.077, 6.649, 6.523]
    mags_e = [0.027, 0.059, 0.031]
    star_flux = np.zeros(n_filt, dtype=np.float64)
    star_flux_e = np.zeros(n_filt, dtype=np.float64)
    model_flux = np.zeros(n_filt, dtype=np.float64)
    filter_wavelength = np.zeros(n_filt, dtype=np.float64)

    for i in xrange(0, n_filt):
        filt = readsav(filter_path+filters[i]+'.sav')
        filter_wl = filt.filter_wl
        filter_fl = filt.filter_fl
        filter_dwl = filt.filter_dwl
        filter_zp = filt.filter_zp
        filter_wavelength[i] = filt.filter_eff
        wl_convol, model_convol, filter_convol = convolve.convol_and_interp(model_wl, model_fl, model_dwl, filter_wl, filter_fl, filter_dwl)
        top = np.trapz(model_convol * filter_convol, wl_convol)
        bottom = np.trapz(filter_convol, wl_convol)
        model_flux[i] = top/bottom

        foo = np.random.normal(loc=mags[i], scale=mags_e[i], size=int(1e6))
        #Convert to flux units
        foo = (10**(foo / -2.5)) * filter_zp
        star_flux[i] = np.nanmean(foo)
        star_flux_e[i] = np.nanstd(foo)


    guess = np.nanmean(star_flux/model_flux)
    result = optimize.minimize(minimize, guess, args=(star_flux, star_flux_e, model_flux))
    scale = result.x[0]
    model_fl *= scale
    model_flux *= scale

    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ind = np.where((model_wl >= 1.0) & (model_wl <= 3.0))
    ax1.plot(model_wl[ind], model_fl[ind], color='k', alpha=0.5)
    ax1.set_xlim(1.0, 3.0)
    ax1.set_yscale('log')
    ax1.set_xlabel(r'Wavelength (um)')
    ax1.set_ylabel(r'$F_{\lambda}$')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(model_wl[ind], model_dwl[ind])
    ax2.set_xlim(1.0, 3.0)
    ax2.set_xlabel(r'Wavelength (um)')
    ax2.set_ylabel(r'$\Delta \lambda$ (um)')

    model_wl, model_fl, foo = convolve.convol_and_interp(model_wl, model_fl, model_dwl, wl, wl*0.0, dwl)
    ax1.plot(model_wl, model_fl, '-o', color='red', alpha=0.75, markersize=4, markeredgecolor='None')

    ax1.plot(filter_wavelength, model_flux, 'o', mfc='none', color='blue', markersize=10)
    ax1.errorbar(filter_wavelength, star_flux, yerr=star_flux_e, fmt='o', color='blue')

    fig.savefig('Figs_app/Star_dwl.pdf', bbox_inches='tight')
    plt.close('all')

    #guess position for WD in the four images - change if using different data!
    wd_x = [215, 216, 217, 218, 220]
    wd_y = [86, 88, 89, 90, 92]

    app_phot = False
    psf_phot = True

    if app_phot is True:

        wd_phot = np.zeros((37, n), dtype=np.float64)
        sat_phot = np.zeros((37, n), dtype=np.float64)

        app = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

        #Use 'wedges' for sky calculation? - This tends to fail at large sizes for the central aperture
        wedge = False
        if wedge is True:
            sky_prefix = 'skywedge'
        else:
            sky_prefix = 'skyannulus'

        median_filter = 11.0
        mask_size = 5.5
        if median_filter is not False:
            filter_prefix = 'filter'+str(median_filter)+'px'
        else:
            filter_prefix = 'nofilter'

        for j in xrange(0, len(app)):

            pool = mp.Pool()
            result = [pool.apply_async(calc, args = (i, k, list, app[j], wd_x, wd_y, wedge, median_filter, mask_size)) for k in xrange(0, 37) for i in xrange(0, n)]    
            output = [p.get() for p in result]

            for i in xrange(0, int(37*n)):
                this_i = output[i][0]
                this_k = output[i][1]
                wd_phot[this_k, this_i] = output[i][2]
                sat_phot[this_k, this_i] = output[i][3]
            

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot((np.min(wd_wl), np.max(wd_wl)), (4910, 4910), color='k')
            ax.plot((np.min(wd_wl), np.max(wd_wl)), (4910-250, 4910-250), '--', color='k')
            ax.plot((np.min(wd_wl), np.max(wd_wl)), (4910+250, 4910+250), '--', color='k')

            for i in xrange(0, n):
                #We want to plot spot / wd X wd / star
                ax.plot(wd_wl, 1.0/((sat_phot[:,i]/wd_phot[:,i]) * (wd_fl / model_fl)))
            
            ax.set_xlabel(r'Wavelength (um)')
            ax.set_ylabel(r'Spot ratio')
            ax.set_ylim(0, 15000)
            ax.annotate(r'Aperture radius ($\lambda/D$) = '+str(app[j]), xy=(0.1,0.9), xycoords='axes fraction', horizontalalignment='left')

            final_val = np.zeros((37, n), dtype = np.float64)
            for i in xrange(0, n):
                final_val[:,i] = 1.0/((sat_phot[:,i]/wd_phot[:,i]) * (wd_fl / model_fl))

            ax.annotate(r'Spot ratio (avg/std) = %8.2f $\pm$ %8.2f' % (np.nanmean(final_val), np.nanstd(final_val)), xy=(0.1,0.8), xycoords='axes fraction', horizontalalignment='left')

            fig.savefig('Figs_app/WD_sat_ratio_'+sky_prefix+'_'+filter_prefix+'_app'+str(app[j])+'.pdf', bbox_inches='tight')
            plt.close('all')

    if psf_phot is True:

        #For each slice:
        # 1 hp filter
        # 2 shift sats to 25,25 in a 50x50 image
        # 3 shift wd similarly
        # 4 scale to minimize some cost function
    
        ratio = np.zeros((37, n), dtype=np.float64)
        filtersize = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        n_filter = len(filtersize)

        for jj in xrange(0, n_filter):

            if 1 == 1:
                pool = mp.Pool()
                result = [pool.apply_async(calc_psf, args = (i, k, list, wd_x, wd_y, filtersize[jj])) for k in xrange(0, 37) for i in xrange(0, n)]    
                output = [p.get() for p in result]

                for i in xrange(0, int(37*n)):
                    this_i = output[i][0]
                    this_k = output[i][1]
                    ratio[this_k, this_i] = output[i][2]
            else: #debugging
                for ik in xrange(0, 37):
                    for ii in xrange(0, n):
                        result = calc_psf(ii, ik, list, wd_x, wd_y, filtersize[jj])
                        ratio[ik, ii] = result[2]



            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot((np.min(wd_wl), np.max(wd_wl)), (4910, 4910), color='k')
            ax.plot((np.min(wd_wl), np.max(wd_wl)), (4910-250, 4910-250), '--', color='k')
            ax.plot((np.min(wd_wl), np.max(wd_wl)), (4910+250, 4910+250), '--', color='k')

            for i in xrange(0, n):
                #We want to plot spot / wd X wd / star
                ax.plot(wd_wl, 1.0/(ratio[:,i] * (wd_fl / model_fl)))
            
            ax.set_xlabel(r'Wavelength (um)')
            ax.set_ylabel(r'Spot ratio')
            ax.set_ylim(0, 15000)

            final_val = np.zeros((37, n), dtype = np.float64)
            for i in xrange(0, n):
                final_val[:,i] = 1.0/(ratio[:,i] * (wd_fl / model_fl))
                    
            ax.annotate(r'Spot ratio (avg/std) = %8.2f $\pm$ %8.2f' % (np.nanmean(final_val), np.nanstd(final_val)), xy=(0.1,0.8), xycoords='axes fraction', horizontalalignment='left')
            mag = np.random.normal(loc = np.nanmean(final_val), scale = np.nanstd(final_val), size = int(1e6))
            mag = 2.5*np.log10(mag)
            ax.annotate(r'$\Delta$ mag = %8.2f $\pm$ %8.2f' % (np.nanmean(mag), np.nanstd(mag)), xy=(0.1,0.75), xycoords='axes fraction', horizontalalignment='left')

            fig.savefig('Figs_psf/WD_sat_ratio_filtersize'+str(filtersize[jj])+'.pdf', bbox_inches='tight')
            plt.close('all')



if __name__ == '__main__':
    main()




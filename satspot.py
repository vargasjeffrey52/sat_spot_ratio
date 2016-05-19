def minimize_psf(p, im1, im2, return_res_arr = False):
	# Function used by op.minimize to fit PSF
	# p[0] = xoffset applied to im2
	# p[1] = yoffset applied to im2
	# p[2] = flux ratio between im1 and im2 (i.e. value to multiply im2 by to match im1)

	if np.shape(im1) != np.shape(im2):
		print 'Mismatching dimensions'
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

	#Inputs:
	#	im1 - first image
	#	im2 - second image, same dimensions
	#	p0 - vector of guesses for PSF fit [xoffset, yoffset, flux_ratio]
	#Outputs:
	#	p - vector of PSF fitting results [xoffset, yoffset, flux_ratio]
	
	# Function to peform PSF fitting. Return residual array if true.
	# Using scipy.optimize.minimize to do this

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

	#Inputs:
	#	im - 2D image
	#   xy_guess - initial guess of source to fit
	#Outputs:
	#	xy - Tuple of fitted xy coordinates

	# Function which fits the location of a point source within an image
	# Same as return_pos in gpi_satspotcalib_wd.py

	return xy

def stamp(im, xy, sout):

	#Inputs:
	#   im - 2D image of arbitrary dimensions (281x281 for GPI)
	#   xy - 2tuple containing x,y coordinates of spot/point source
	#	sout - size of output image, with source shifted to (sout/2, sout/2)
	#Output:
	#	stamp - 2d image of size (sout, sout)

	# Use ndimage.map_coordinates to create the stamp by interpolation.
	# See lines 365-374 in gpi_satspotcalib_wd.py

	return stamp




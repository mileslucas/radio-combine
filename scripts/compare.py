# Compare.py
# Miles Lucas - mdlucas@nrao.edu

import numpy as np
import argparse

# CASA imports
from casac import casac
ia = casac.image()
tb = casac.table()
log = casac.logsink()

def compare(image_a, image_b, regrid=False, plot=True):
	'''
	Compare the given images

	Params
	------
	image_a: str
		The path to the first image
	image_b: str
		The path to the second image
	regrid: bool (optional)
		If true, regrids the second image to the first image's coordinate system. Default=False
	plot: bool (optional)
		If true, creates plots of the power spectra. Default=True

	Returns
	-------
	psds: ndarray
		This is a 2x2 array with the two power spectrum densities for each image. The format is
		[psd_a, psd_b] where each psd consists of [freq, pow]. 
	'''
	if regrid:
		image_b = regrid_im(image_a, image_b)
	smap_a, amps_a = get_arrays(image_a)
	smap_b, amps_b = get_arrays(image_b)

	r_a, pow_a = get_psd(smap_a, amps_a)
	r_b, pow_b = get_psd(smap_b, amps_b)

	if plot:
		comparison_plot(r_a, pow_a, image_a, r_b, pow_b, image_b)
	

	# Verify Fourier transform property
	idx = r_a == 0
	log.post('Image: {}\nTotal power: {}\nMatches image: {}\n'.format(image_a, np.real(pow_a[idx][0]), np.sum(amps_a) == pow_a[idx]))
	idx = r_b == 0
	log.post('Image: {}\nTotal power: {}\nMatches image: {}\n'.format(image_b, np.real(pow_b[idx][0]), np.sum(amps_b) == pow_b[idx]))
	return [[r_a, pow_a], [r_b, pow_b]]

	

def regrid_im(image_a, image_b):
	'''
	Regrids image_b from image_a

	Params
	------
	image_a: str
		The path to the first (reference) image
	image_b: str
		The path to the second image)

	Returns
	-------
	outname: str
		The path of the regridded image. This should be the same as <image_b>.regrid
	'''

	ia.open(image_a)
	cs = ia.coordsys()
	ia.close()
	ia.open(image_b)
	tokens = image_b.split('.')
	outname = '.'.join(tokens[:-1]) + '.regrid.' + tokens[-1]
	ib = ia.regrid(outfile=outname, csys=cs.torecord(), shape=ia.shape(), overwrite=True)
	ib.done()
	cs.done()
	ia.done()
	ia.close()

	return outname


def get_arrays(image_path):
	'''
	Gets the numpy array for a given image_path
	
	Params
	------
	image_path: str
		The path to the image
	
	Returns
	-------
	smap: dict
		A dictionary with the relevant skymap axis information. 
	pow: ndarray
		A 2 dimensional array of the powers of the fft
	'''

	# Get the Powers
	tb.open(image_path)
	amps = np.squeeze(tb.getcol('map'))
	tb.close()

	# Get the sky map axes
	ia.open(image_path)
	summ = ia.summary()
	ia.close()

	smap = {
		'n_x': summ['shape'][0],
		'd_x': summ['incr'][0],
		'n_y': summ['shape'][1],
		'd_y': summ['incr'][1],
	}

	return smap, amps


def get_psd(smap, amps):
	'''
	Creates a power spectrum density (PSD) from given skymap axis information and 2D amplitudes

	Params
	-------
	smap: dict
		The relevant skymap axis information. There should be four keys: 'n_x' and 'n_y' are
		the length of each axis and 'd_x' and 'd_y' are the increments in degrees of each axis
	amps: ndarray
		The 2-dimensional amplitude map of the image

	Returns
	-------
	uvdist: ndarray
		The square distance of each fourier point from the origin
	power: ndarray
		The unnormalized power of the  fourier transform
	'''
	# Get the frequencies
	us = np.fft.fftfreq(smap['n_x'], smap['d_x'])
	vs = np.fft.fftfreq(smap['n_y'], smap['d_y'])
	fft = np.fft.fft2(amps)
	uvdist = []
	power = []
	for i, u in enumerate(us):
		for j, v in enumerate(vs):
			uvdist.append(np.linalg.norm((u, v)))
			power.append(fft[i, j])

	# This will sort the arrays by uvdist but maintain link between distance and power
	uvdist, power = zip(*sorted(zip(uvdist, power)))	

	return np.array(uvdist), np.array(power)

def comparison_plot(r_a, pow_a, name_a, r_b, pow_b, name_b):
	'''
	Plots comparison of the two power spectrum. This method will throw an exception if 
	there is no display. 
	
	Params
	------
	r_a: array-like
		The distances of the psd for the first image
	pow_a: array-like
		The powers of the psd for the first image
	name_a: str
		The name of the first image or dataset
	r_b: array-like
		The distances of the psd for the second image
	pow_b: array-like
		The powers of the psd for the second image
	name_b: str
		The name of the second image or dataset
	'''
	import matplotlib.pyplot as plt
	import matplotlib as mpl

	# Style
	#mpl.style.use('seaborn')
	style = {
		'figure.figsize': (9, 6),
		'axes.labelsize': 14,
		'axes.titlesize': 16,
		'legend.fontsize': 14,
	}
	mpl.rcParams.update(style)

	line_props = {
		'ls': '',
		'marker': '.',
		'ms': 4,
		'mew': 0,
		'alpha': 0.4,
		'lw': 1,
	}
	# Get bin the data
	bin_width = 100
	x = np.arange(0, min((max(r_a), max(r_b))), bin_width)
	int_y_a = np.interp(x, r_a, np.real(pow_a))
	int_y_b = np.interp(x, r_b, np.real(pow_b))

	# Plots
	grid = plt.GridSpec(2, 3, width_ratios=[1, 1, 2])
	fig = plt.figure(figsize=(18,9))
	ax1 = plt.subplot(grid[0,0])
	plt.semilogy(r_a/1000, pow_a, c='b', **line_props)
	plt.ylabel(name_a)
	plt.title('PSD')
	plt.gca().get_xaxis().set_visible(False)

	plt.subplot(grid[1,0], sharex=ax1, sharey=ax1)
	plt.semilogy(r_b/1000, pow_b, c='g', **line_props)
	plt.ylabel(name_b)

	plt.subplot(grid[0,1], sharex=ax1, sharey=ax1)
	plt.semilogy(x/1000, int_y_a, 'b.', mew=0)
	plt.title('Interpolated PSD')
	plt.gca().get_xaxis().set_visible(False)
	plt.gca().get_yaxis().set_visible(False)
	
	plt.subplot(grid[1,1], sharex=ax1, sharey=ax1)
	plt.semilogy(x/1000, int_y_b, 'g.', mew=0)
	plt.gca().get_yaxis().set_visible(False)

	ax5 = plt.subplot(grid[:, 2], sharex=ax1)
	plt.plot(x/1000, int_y_b / int_y_a, 'o')
	plt.title('Comparison of PSD')
	ax5.yaxis.tick_right()


	
	fig.text(0.04, 0.5, 'Power', fontsize=14, va='center', rotation = 'vertical')
	fig.text(0.5, 0.04, r'UV Distance ($k\lambda$)', ha='center', fontsize=14)
	fig.text(0.96, 0.5, r'Power ratio', va='center', fontsize=14, rotation='vertical')
	
	plt.show()


	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('image_a', help='path to the first image')
	parser.add_argument('image_b', help='path to the second image')
	parser.add_argument('-r', '--regrid', action='store_true', help="regrids image_b to image_a's coordinates")
	parser.add_argument('-n', '--no-plot', dest='plot', action='store_false')
	args = parser.parse_args()
	compare(args.image_a, args.image_b, args.regrid, args.plot)

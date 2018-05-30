# Compare.py
# Miles Lucas - mdlucas@nrao.edu

import numpy as np
import argparse
import re

# CASA imports
from casac import casac
ia = casac.image()
tb = casac.table()

def compare(image_a, image_b, regrid=False, plot=True):
	'''
	Compare the given images
	'''
	if regrid:
		image_b = regrid_im(image_a, image_b)
	smap_a, amps_a = get_arrays(image_a)
	smap_b, amps_b = get_arrays(image_b)

	r_a, pow_a = get_psd(smap_a, amps_a)
	r_b, pow_b = get_psd(smap_b, amps_b)

	if plot:
		comparison_plot(r_a, pow_a, image_a.split('/')[-1], r_b, pow_b, image_b.split('/')[-1])
	

	# Verify Fourier transform property
	idx = r_a == 0
	print 'Image: {}\nTotal power: {}\nMatches image: {}\n'.format(image_a, np.real(pow_a[idx][0]), np.sum(amps_a) == pow_a[idx])
	idx = r_b == 0
	print 'Image: {}\nTotal power: {}\nMatches image: {}\n'.format(image_b, np.real(pow_b[idx][0]), np.sum(amps_b) == pow_b[idx])
	return r_a, pow_a, r_b, pow_b

	

def regrid_im(image_a, image_b):
	'''
	Regrids image_b from image_a


	Returns
	-------
	outname: str
		The path of the regridded image. This should be the same is <image_b>.regrid
	'''

	ia.open(image_a)
	cs = ia.coordsys()
	ia.close()
	ia.open(image_b)
	tokens = image_b.split('.')
	outname = '.'.join(tokens[:-1]) + '.regrid.' + tokens[-1]
	ia.regrid(outfile=outname, csys=cs.torecord(), shape=ia.shape(), overwrite=True)
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
	Turn the 2D data into 1D data by taking the distance from the origin
	'''
	# Get the frequencies
	freq_x = np.fft.fftfreq(smap['n_x'], smap['d_x'])
	freq_y = np.fft.fftfreq(smap['n_y'], smap['d_y'])

	fft = np.fft.fft2(amps)

	dist = []
	power = []
	for i, x in enumerate(freq_x):
		for j, y in enumerate(freq_y):
			dist.append(np.linalg.norm((x, y)))
			power.append(fft[i, j])

	return np.array(dist), np.array(power)

def comparison_plot(r_a, pow_a, name_a, r_b, pow_b, name_b):
	'''
	Plots comparison of the two power spectrum
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
	# Singular Plots
	plt.figure(figsize=(9, 9))
	ax1 = plt.subplot(211)
	plt.semilogy(r_a, pow_a, c='b', **line_props)
	plt.title(name_a)

	ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
	plt.semilogy(r_b, pow_b, c='g', **line_props)
	plt.title(name_b)
	plt.xlabel(r'$f$ (Hz)')
	
	plt.tight_layout()
	plt.show()

	# Plot Both
	plt.figure()
	plt.semilogy(r_a, pow_a, label=name_a, **line_props)
	plt.semilogy(r_b, pow_b, label=name_b, **line_props)
	plt.title('Comparison of PSD')
	plt.xlabel(r'$f$ (Hz)')
	plt.ylabel(r'Power (unnormalized)')
	plt.legend(loc='best')
	plt.show()

	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('image_a', help='path to the first image')
	parser.add_argument('image_b', help='path to the second image')
	parser.add_argument('-r', '--regrid', action='store_true', help="regrids image_b to image_a's coordinates")
	parser.add_argument('-n', '--no-plot', dest='plot', action='store_false')
	args = parser.parse_args()
	compare(args.image_a, args.image_b, args.regrid, args.plot)

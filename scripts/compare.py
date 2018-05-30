# Compare.py
# Miles Lucas - mdlucas@nrao.edu

import numpy as np
# from matplotlib import rc
# import matplotlib.pyplot as plt
import argparse

def compare(image_a, image_b, regrid=False):
	'''
	Compare the given images
	'''
	waves_a, amps_a = get_arrays(image_a)
	waves_b, amps_b = get_arrays(image_b)

	

def regrid(image_a, image_b):
	'''
	Regrids image_b from image_a


	Returns
	-------
	outname: str
		The path of the regridded image. This should be the same is <image_b>.regrid
	'''
	
	ia.open(image_a)
	cs = ia.coordsys()
	ia.open(image_b)
	outname = image_b + '.regrid'
	ia.regrid(outfile=outname, csys=cs.torecord(), shape=ia.shape(), overwrite=True)
	cs.done()
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
	waves: ndarray
		A 2 dimensional array of the wavelengths of the fft
	amps: ndarray
		A 2 dimensional array of the amplitudes of the fft
	'''
	ia.open(image_path)
	ia.fft(real='r.im', amps='a.im')
	ia.close()

	tb.open('r.im')
	waves = np.squeeze(tb.getcol('map'))
	tb.close()

	tb.open('a.im')
	amps = np.squeeze(tb.getcol('map'))
	tb.close()

	ia.removefile('r.im')
	ia.removefile('a.im')

	return waves, amps


def arr_transform(freq_x, freq_y, amps):
	'''
	Turn the 2D data into 1D data by taking the distance from the origin
	'''
	dist = []
	power = []
	for i, x in enumerate(freq_x):
		for j, y in enumerate(freq_y):
			dist.append(np.linalg.norm((x, y)))
			power.append(amps[i, j])

	return np.array(dist), np.array(power)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('image_a', help='path to the first image')
	parser.add_argument('image_b', help='path to the second image')
	parser.add_argument('-r', '--regrid', action='store_true', help="regrids image_b to image_a's coordinates")
	args = parser.parse_args()
	compare(args.image_a, args.image_b, args.regrid)

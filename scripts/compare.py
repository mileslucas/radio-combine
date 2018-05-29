# Compare.py
# Miles Lucas - mdlucas@nrao.edu

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import argparse

def compare(image_a, image_b):
	'''
	Compare the given images
	'''
	

def get_arrays(image_path):
	'''
	Gets the numpy array for a given image_path
	
	Params
	------
	image_path: str
		The path to the image
	
	Returns
	-------
	im_arr: ndarray
		An array of shape (N,M) where N and M are the resolution of the image	     '''
	
	tb.open(image_path)
	raw_data = tb.getcol('map')
	# The original shape may be something like (300,300,1,1,1) so we squeeze
	im_arr = np.squeeze(raw_data)
	return im_arr
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('image_a', help='path to the first image')
	parser.add_argument('image_b', help='path to the second image')
	args = parser.parse_args()
	compare(args.image_a, args.image_b)

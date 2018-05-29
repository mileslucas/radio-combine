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
	pass


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('image_a')
	parser.add_argument('image_b')
	args = parser.parse_args()
	
	compare(args.image_a, args.image_b)

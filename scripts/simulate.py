# simulate.py
# Miles Lucas - mdlucas@nrao.edu

import argparse

# CASA imports
from casac import casac
ia = casac.image()
from task_simobserve import simobserve






if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('path', help='The path and filename prefix. \
		The files will be saved as <path>.sd.im and <path>.int.im')
	args = parser.parse_args()
	

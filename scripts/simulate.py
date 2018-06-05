# simulate.py
# Miles Lucas - mdlucas@nrao.edu

import argparse
import os

# CASA imports
from casac import casac
me = casac.measures()
sm = casac.simulator()
cl = casac.componentlist()
from tclean import tclean
from simutil import simutil

def simulate(path):
	'''
	'''
#	create_points_cl()
	setup_simulator(path)
	clean(path)

def create_points_cl():
	cl.addcomponent(dir='J2000 01h0m0.0 +47.0.0.000', flux=1.0, freq='5GHz', shape='point')
	cl.addcomponent(dir='J2000 00h0m0.0 +47.0.0.000', flux=2.0, freq='5GHz', shape='point')
	cl.addcomponent(dir='J2000 01h0m0.0 +45.0.0.000', flux=2.0, freq='5GHz', shape='point')
	cl.addcomponent(dir='J2000 00h0m0.0 +43.0.0.000', flux=0.5, freq='5GHz', shape='point')
	cl.rename('data/points.cl')
	cl.close()

def setup_simulator(path, config='d'):
	'''
	'''
	sm.open(path + '.int.ms')
	u = simutil()
	configdir = os.getenv('CASAPATH').split()[0] + "/data/alma/simmos/"
	x, y, z, d, padnames, telescope, posobs = u.readantenna(configdir + 
			'vla.{}.cfg'.format(config))
	
	sm.setconfig(telescopename=telescope, x=x, y=y, z=z, dishdiameter=d.tolist(), 
		mount=['alt-az'], antname=padnames, coordsystem='global', 
		referencelocation=posobs)
	
	sm.setspwindow(spwname='CBand', freq='5GHz', deltafreq='50MHz', 
		freqresolution='50MHz', nchannels=1, stokes='RR')

	# Initialize the source and calibrater 
	sm.setfield(sourcename='My cal', sourcedirection=['J2000','00h0m0.0','+45.0.0.000'], 
		calcode='A') 
	sm.setfield(sourcename='My source', sourcedirection=['J2000','01h0m0.0','+47.0.0.000']) 
	sm.setlimits(shadowlimit=0.001, elevationlimit='8.0deg')
	sm.setauto(autocorrwt=0.0)
	sm.settimes(integrationtime='10s', usehourangle=False, referencetime=me.epoch('utc', 'today'))
	
	sm.observe('My cal', 'CBand', starttime='0s', stoptime='300s')
	sm.observe('My source', 'CBand', starttime='310s', stoptime='610s')
	
	sm.setdata(spwid=1, fieldid=1)
	sm.predict(complist='data/points.cl')
	sm.close() 

def clean(path, **tclean_args):
	'''
	'''
	tclean(vis=path + '.int.ms', imagename=path + '.int.im', imsize=256, 
		niter=1000, **tclean_args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('path', help='The path and filename prefix. \
		The files will be saved as <path>.sd.im and <path>.int.im')
	args = parser.parse_args()
	simulate(args.path)

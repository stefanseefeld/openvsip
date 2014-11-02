#!/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2014 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.BSD file.
#
# This demo is based on the k-Ω Beamformer Example by Randall Judd
# (http://portals.omg.org/hpec/files/vsipl/CD/JuddKO.pdf).
# See also the implementations in 
# https://github.com/rrjudd/jvsip/blob/master/examples/komegaExamples

from vsip import random
from vsip.signal import *
from vsip.signal import window
from vsip.signal.fftm import *
from vsip.signal.fir import fir
from vsip.selgen import generation
from vsip.math import elementwise as elm
from vsip.math import reductions as reduce
from vsip.math import matvec
from vsip.signal.freqswap import freqswap
from vsip.selgen.clip import clip
import math
from matplotlib import pyplot

speed=1500       # propagation speed
rate=1024        # sample rate
nts=1024         # length of time series
spacing=1.5      # sensor spacing
sensors=128      # number of sensors
averages=8       # data sets to average

# tones in hertz for sinusoids present in simulated signal
sim_freqs=[450, 300, 150, 50, 55, 95]
# corresponding bearings in degrees for above sinusoids
sim_bearings=[50, 130, 130, 90, 120, 70] 
# number of simulated noise directions
nnoise=64

class time_series(object):
    """ Simulate acoustic data with narrow band point sources from multiple
        directions and isotropic, band-limited noise.
    """
    def __init__(self):
        # length of narrow band simulated noise signal
        L = int(2 * rate/(sensors * spacing/speed) + nts + 1)
        # Kaiser window kernel to reject out of band noise
	kernel = window.kaiser(float, 6, 1)
	# Window-based FIR filter design using above kernel
	self.fir = fir(kernel, symmetry.none, 2*L, 2, obj_state.save, 0, alg_hint.time)
	# band-limited noise vectors
	self.noise = vsip.vector(float, 2*L)
        self.bl_noise = vsip.vector(float, L)
	# Generate white gaussian noise
	self.rand  = vsip.random.rand(float, 7, True)
	# time series indices
	self.t = generation.ramp(float, 0.,1./rate, nts)
        self.t_dt = vsip.vector(float, nts)
	# simulated data matrix in which each row corresponds to a sensor
	self.data = vsip.matrix(float, sensors, nts)
        self.d_t = spacing/speed
    
    # simulate narrow band data composed purely of sinusoids specified earlier
    def nb_sim(self):
        for i in range(len(sim_freqs)):
            # pick a center frequence for sinusoid 
	    f=sim_freqs[i]
            # Calculate effective angle based on its bearing
	    b=self.d_t * math.cos(sim_bearings[i] * math.pi/180.0)
            for j in range(sensors):
		# Introduce phase shifts in selected sinusoid in accordance with the sensor ID
		dt = float(j) * b
                self.t_dt = self.t + dt
                self.t_dt *= 2 * math.pi * f
                self.t_dt = elm.cos(self.t_dt)
                self.t_dt *= 3
		# Multiplex phase shifted sinusoids corresponding to a specific sensor
                self.data[j,:] += self.t_dt
    
    def noise_sim(self):
	#sensor-to-sensor travel time  
	d_t=self.d_t * rate 
	# array travel time at end        
	o_0    = d_t * sensors + 1 
	# angle step        
	a_stp = math.pi/nnoise 
        for j in range(nnoise):
            a_crct = math.cos(float(j) * a_stp)
            # Generate white noise 
	    self.noise = self.rand.randn(self.noise.length())
            # Get it colored by rejecting out-of-band components
	    self.fir(self.noise, self.bl_noise)
            # Adjust noise variance
	    self.bl_noise *= 12./nnoise

            for i in range(sensors):
               offset = int(o_0 + i * d_t * a_crct)
               # Mix noise with data to model a noisy channel
	       self.data[i,:] += self.bl_noise[offset:offset + nts]
            # Subtract average value from mixture to remove any biasing
	    self.data -= reduce.meanval(self.data)

    def reset(self):
        self.data[:] = 0.

    def __call__(self):
        return self.data


class k_omega(object):

    def __init__(self):
	# number of frequency bins
        frequencies=int(nts/2) + 1
	# Space-frequency matrix        
	self.cfreq=vsip.matrix(complex, sensors, frequencies)
	# Space-time matrix        
	self.rfreq=vsip.matrix(float, sensors, frequencies)
	# K-omega output matrix initialized        
	self.gram=vsip.matrix(float, sensors, frequencies, 0)
	# FFT object along time domain         
	self.rcfftm=fftm(float, fwd, sensors, nts, 1, vsip.row, 0, alg_hint.time)
	# FFT object along spatial domain        
	self.ccfftm=fftm(complex, fwd, sensors, frequencies, 1, vsip.col, 0, alg_hint.time)
	# Window taper object around time axis    
	self.ts_taper=window.hanning(float, nts)
	# Window taper object around spatial axis        
	self.array_taper=window.hanning(float, sensors)

    def __call__(self, data):

        # data tapers
	# Reject high frequency components along time axis
        data = matvec.vmmul(vsip.row, self.ts_taper, data)
	# Reject high frequency components along spatial axis        
	data = matvec.vmmul(vsip.col, self.array_taper, data)
        # 2D FFT: charateristic of k-omega beamforming, first around time axis
        self.rcfftm(data, self.cfreq)
	# and then around spatial axis        
	self.ccfftm(self.cfreq)
	# Calculate averaged power spectrum 
        self.rfreq = elm.magsq(self.cfreq)
        self.rfreq *= 1.0/averages
	# Accumulate result in k-omega output        
	self.gram += self.rfreq

    def get(self):
        return self.gram

def main():
    # initialize input/ouput objects
    ts=time_series()
    kw=k_omega()
    for i in range(averages):
	# initialize time series        
	ts.reset()
	# Simulate narrow band acoustic data        
	ts.nb_sim()
	# Simulate colored gaussian noise
        ts.noise_sim()
	# Perform k-omega beamforming
        kw(ts())
    
    # Obtain the result of beamforming	
    gram=kw.get()    
    
    # rearrange to bring zero-frequency component at the center of spectrum 
    for i in range(gram.size(1)):
        gram[:,i] = freqswap(gram[:,i])
    
    # Post-process the beamformer output to make it suitable for charting
    max, idx = reduce.maxval(gram)
    avg = reduce.meanval(gram)
    gram = clip(gram,0.0,max,avg/100000.0,max)
    # plot log-magnitude of beamformed power spectrum
    gram = elm.log10(gram)
    min, idx = reduce.minval(gram)
    gram -= min
    # Normalize the log-magnitude plot
    max, idx = reduce.maxval(gram)
    gram *= 1.0/max
    fig = pyplot.figure(1,figsize=(10,4))
    ax = fig.add_axes([0.10,0.10,0.85,0.80])
    ax.set_yticklabels(['0','0','30','60','90','120','150','180'])
    ax.yaxis.set_ticks_position('right')
    pyplot.imshow(gram)
    # Labeling plot axis appropriately
    pyplot.title(u'K-Ω Beamformer Output')
    pyplot.xlabel('Frequency')
    pyplot.ylabel(r'$\frac{cos(\theta)}{\lambda}$',fontsize=16,rotation='horizontal')
    pyplot.colorbar()
    # Display the plot
    pyplot.show()

if __name__ == '__main__':
    main()

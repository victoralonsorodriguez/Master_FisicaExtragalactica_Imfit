# This file is just some test for the isophote fitting
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.table import Table

import os
import re
import sys
import subprocess
import shutil

import pdb

from py_info import *
from py_conf_imfit import create_conf_imfit
from py_conf_psf import create_conf_psf

from photutils.datasets import make_noise_image
from photutils.isophote import EllipseGeometry
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse
from photutils.isophote import build_ellipse_model




hdu = fits.open('NGC5443_mask2D_new.fits')
hdr_mask = hdu[0].header
img_mask = hdu[0].data

# Changing from 0/1 mask to True/False mask
img_mask = img_mask == 1

# Loading our data
hdu = fits.open('NGC5443_i.fits')
hdr = hdu[0].header
img = hdu[0].data

img = ma.masked_array(img, mask=img_mask)
img_log = np.log(img)

# Defining an initial ellipse 
geometry = EllipseGeometry(x0=530, y0=179, sma=20, eps=0.5,
                           pa=20.0 * np.pi / 180.0)

aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                          geometry.sma * (1 - geometry.eps),
                          geometry.pa)

# We want to fit data with the elliptical geometry
ellipse = Ellipse(img, geometry)
isolist = ellipse.fit_image(step=0.1)

isophote_table = isolist.to_table()
isophote_table.write('table.csv', format='csv', overwrite=True)

fig = plt.figure(figsize=(21, 5))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

plt.subplot(1, 3, 1)
plt.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err,
             fmt='o', markersize=4)
plt.xlabel('Semimajor Axis Length (pix)')
plt.ylabel('Ellipticity')

plt.subplot(1, 3, 2)
plt.errorbar(isolist.sma, isolist.pa / np.pi * 180.0,
             yerr=isolist.pa_err / np.pi * 180.0, fmt='o', markersize=4)
plt.xlabel('Semimajor Axis Length (pix)')
plt.ylabel('PA (deg)')

ax = plt.subplot(1, 3, 3)
plt.errorbar(isolist.sma, isolist.intens, yerr=isolist.int_err, fmt='o',
             markersize=4)
plt.xlabel('Semimajor Axis Length (pix)')
plt.ylabel('Intensity')
ax.set_yscale('log')

# Plotting isophotes

fig, (ax1) = plt.subplots(figsize=(14, 5), nrows=1, ncols=1)
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
ax1.imshow(img_log, origin='lower')
ax1.set_title('Data')

smas = np.linspace(10, 200, 10)
for sma in smas:
    iso = isolist.get_closest(sma)
    x, y, = iso.sampled_coordinates()
    ax1.plot(x, y, color='white')

plt.show()
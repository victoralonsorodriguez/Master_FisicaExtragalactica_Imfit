# This programe runs imfit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits

import os
import re
import sys
import subprocess
import shutil

import pdb

from py_info import *
from py_conf_imfit import create_conf_imfit
from py_conf_psf import create_conf_psf



def open_fits(fits_path):
    
    fits_name = fits_path.split('/')[-1].split('.')[0]
    # Open the fits with Astropy
    hdu = fits.open(f'{fits_path}')
    hdr = hdu[0].header
    img = hdu[0].data
    
    return (hdr,img,fits_name)

def create_galaxy_folder(galaxy):
    
    # Creating a folder for each galaxy
    galaxy_folder = f'{galaxy}'
    galaxy_folder_path = f'{cwd}/{galaxy_folder}'
    
    # If the folder was created previously is remvoved
    if os.path.isdir(f'{galaxy_folder_path}') == True:
        shutil.rmtree(f'{galaxy_folder_path}')
    os.mkdir(f'{galaxy_folder_path}')

    return galaxy_folder_path

def fits_mag_to_flux(fits_path):
    
    hdr,img,fits_name = open_fits(fits_path)

    # Flux from magnitudes to counts
    #flux = 10**((img - 2.5*np.log10(inst**2)-zcal)/-2.5)
    
    # Expoting the new fits
    flux_fits  = f'{fits_name}_counts.fits'
    output_path = f'{cwd}/{galaxy}/{flux_fits}'
    #fits_ouput = fits.writeto(f'{output_path}', flux, header=hdr, overwrite=True)
    
    return output_path
    

def galaxy_index(df_info,galaxy):

    index = df_info.loc[df_info['galaxy'] == galaxy].index.values[0]
    return index

def create_psf(fits_path,galaxy,df_sky,df_psf):
    
    # Open the fits with Astropy
    hdr,img,fits_name = open_fits(fits_path)
    
    # We are going to create a PSF of the size
    # of the galaxy's image
    x_len = img.shape[1]//2
    y_len = img.shape[0]//2
    
    # We need odd shape because we need to center
    # the PSF
    if x_len%2 == 0:
        x_len += 1
    elif y_len%2 == 0:
        y_len += 1
        
    max_len = min(x_len,y_len)
    x_len = max_len
    y_len = max_len
        
    x_center = x_len/2 + 0.5
    y_center = y_len/2 + 0.5
    
    # These are the size parameters for the PSF
    psf_shape = (x_len,y_len)
    psf_center = (x_center,y_center)
    
    # Moffat psf parameters in pixeles
    galaxy_sky_index = galaxy_index(df_sky,galaxy)
    inst_arcsec_pix = df_sky.loc[galaxy_sky_index]['inst']
    
    # Original value in arcsec
    galaxy_psf_index = galaxy_index(df_psf,galaxy)
    fwhm_value_arcsec = df_psf.loc[galaxy_psf_index]['FWHM_i']
    beta_value_arcsec = df_psf.loc[galaxy_psf_index]['Beta_i']
    
    # Changing to pix
    fwhm_value_pix = fwhm_value_arcsec / inst_arcsec_pix
    beta_value_pix = beta_value_arcsec / inst_arcsec_pix
    
    # Creating the configuration script for the psf
    psf_conf_file_name = f'{galaxy}_conf_psf.txt'
    psf_conf_file_path = f'{cwd}/{galaxy}/{psf_conf_file_name}'
    psf_conf_file = open(psf_conf_file_path,'w+')
    create_conf_psf(file=psf_conf_file,
                     galaxy = galaxy,
                     shape = psf_shape,
                     center = psf_center,
                     fwhm = fwhm_value_pix,
                     beta = beta_value_pix
                     )
    psf_conf_file.close()
    
    # Creating the PSF
    output_psf_file = f'{galaxy}_psf.fits'
    output_psf_file_path = f'{cwd}/{galaxy}/{output_psf_file}'

    subprocess.run(['makeimage',
                    psf_conf_file_path,
                    '-o', output_psf_file_path])
    
    hdr,img,fits_name = open_fits(output_psf_file_path)
    
    psf_flux = np.nansum(img)
    psf_norm = img / psf_flux
    
    output_psf_file_norm = f'{galaxy}_psf_norm.fits'
    output_psf_file_norm_path = f'{cwd}/{galaxy}/{output_psf_file_norm}'
    
    fits.writeto(f'{output_psf_file_norm_path}',psf_norm,hdr,overwrite=True)

    return output_psf_file_norm_path


def main(gal_pos,galaxy):
    
    # Creating a folder for each galaxy
    galaxy_folder_path = create_galaxy_folder(galaxy)
    
    # Fits file to analyze
    fits_file = fits_image_list[gal_pos]
    fits_name = fits_file.split('.')[0]
    fits_path = f'{files_path}/{fits_file}'
    
    # Mask file 
    mask_file = fits_mask_list[gal_pos]
    mask_name = mask_file.split('.')[0]
    mask_path = f'{files_path}/{mask_file}'
    
    # Parameters 
    hdr,img,fits_name = open_fits(fits_path)
    
    # obtaining the shape of the fits
    crop_factor = 10
    x_len = img.shape[1]
    y_len = img.shape[0]
    
    x_len_crop = (x_len//crop_factor)
    y_len_crop = (y_len//crop_factor)
    
    # Obtaining the nearest to the center maximun pixel value position to center the model
    img_crop_center = img[x_len//2 - x_len_crop:x_len//2 + x_len_crop, y_len//2 - y_len_crop:y_len//2 + y_len_crop]
    
    max_pix_value_center = np.nanmax(img_crop_center)
    x_center = np.where(img == max_pix_value_center)[0][0]+1
    y_center= np.where(img == max_pix_value_center)[1][0]+1
    x_center = 530
    y_center = 178
    center_dev = 2
    
    # Position angle
    pos_ang = [105,0,180]
    
    # Ellipticity
    ellip = [0.5,0,180]

    # Center intensity
    I_0 = max_pix_value_center
    
    
    
    # Configuration file
    conf_file_name = f'{galaxy}_conf_imfit.txt'
    conf_file_path = f'{galaxy_folder_path}/{conf_file_name}'
    conf_file = open(f'{conf_file_path}','w+')
    create_conf_imfit(file=conf_file,
                     funct=['Sersic','Exponential'],
                     galaxy = galaxy,
                     img_center_x = [x_center,x_center-center_dev,x_center+center_dev],
                     img_center_y = [y_center,y_center-center_dev,x_center+center_dev],
                     pos_ang = pos_ang,
                     ellip = ellip,
                     n=[2,0.6,6],
                     I_e=[120,0,200],
                     r_e=[60,0,100],
                     I_0=[I_0,0,I_0+I_0*0.1],
                     h=[80,0,100])
    conf_file.close()
    
    # Position of the galaxy in the sky info dataframe
    galaxy_sky_index = galaxy_index(df_sky_info,galaxy)
    
    # Gain value
    gain_value = df_sky_info.loc[galaxy_sky_index]['gain']
    
    # Readnoise value
    noise_value = df_sky_info.loc[galaxy_sky_index]['RON']
    
    # Sky value
    sky_value = df_sky_info.loc[galaxy_sky_index]['sky']
    
    # PSF image
    psf_path = create_psf(fits_path,galaxy,df_sky_info,df_psf_info)
    
    # Output files
    fits_output_name = f'{fits_name}_model.fits'
    fits_output_path = f'{galaxy_folder_path}/{fits_output_name}'
    
    # Output files
    residual_output_name = f'{fits_name}_model.fits'
    residual_output_path = f'{galaxy_folder_path}/{residual_output_name}'

    
    # Changing the directory to run imfit
    os.chdir(f'{galaxy}')
    
    # These lines execute imfit as it was a 
    # terminal command 
    subprocess.run(['imfit',
                    fits_path,
                    '-c', conf_file_path,
                    '--mask',mask_path,
                    '--gain=',f'{gain_value}',
                    '--readnoise=',f'{noise_value}',
                    '--sky',f'{sky_value}',
                    '--psf',psf_path,
                    '--save-model=',fits_output_path,
                    '--save-residual=',residual_output_path])
    
    # Returning to the main directory to continue the programe
    os.chdir(f'{cwd}')
    

if __name__ == '__main__':
    
    cwd = os.getcwd()
    galaxy_original_folder = 'galaxy_files'
    files_path = f'{cwd}/{galaxy_original_folder}'
    
    fits_list = []
    fits_image_list = []
    fits_mask_list = []
    galaxy_list = []
    
    for file in sorted(os.listdir(files_path)):
        if '.fits' in file:
        
            fits_list.append(file)
            
            fits_name = file.split('.')[0]
            galaxy = fits_name.split('_')[0]
            
            if galaxy not in galaxy_list:
                galaxy_list.append(galaxy)
            
            if '_i.fits' in file:
                fits_image_list.append(file)
            elif '_mask' in file:
                fits_mask_list.append(file)
                
        elif '.csv' in file and 'info' in file:
            csv_path = f'{files_path}/{file}'
            if 'sky' in file:
                df_sky_info = sky_info(csv_path)
            elif 'psf' in file:
                df_psf_info = psf_info(csv_path)
    
    if len(galaxy_list) != 0:
        for gal_pos,galaxy in enumerate(galaxy_list):
            main(gal_pos,galaxy)        
    
    else:
        
        print('There is no galaxies in the directory')
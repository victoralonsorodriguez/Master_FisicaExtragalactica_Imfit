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
import mmap
import struct

import pdb
import warnings
warnings.filterwarnings('ignore')

from py_conf_imfit import create_conf_imfit
from py_conf_psf import create_conf_psf

from photutils.isophote import EllipseGeometry
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse


def open_fits(fits_path):
    
    fits_name = fits_path.split('/')[-1].split('.')[0]
    # Open the fits with Astropy
    hdu = fits.open(f'{fits_path}')
    hdr = hdu[0].header
    img = hdu[0].data
    
    return (hdr,img,fits_name)

def plot_profiles(galaxy,csv_path_list,fig_name,mag=False,cons=None):
    
    # Creating some figures
    plot_rows = 1
    plot_cols = 3
    
    csv_values_dict = {}
    df_len_list = []
    for csv_label in csv_path_list:
        df = pd.read_csv(csv_label[0])
        csv_values_dict[csv_label[1]] = df
        df_len_list.append(len(df))
        
    max_data_len = min(df_len_list)
    
    profile_to_plot = ['ellipticity','pa','intens']
    profile_axis_label = ['Ellipcity',
                          'Position Angle [deg]',
                          'Intensity [counts]']
    
    fig = plt.figure(figsize=(7*plot_cols, 7*plot_rows))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    for prof_pos,prof in enumerate(profile_to_plot):
        ax = plt.subplot(plot_rows, plot_cols, prof_pos+1)
        plt.xlabel('Semimajor Axis Length [pix]')
        plt.ylabel(f'{profile_axis_label[prof_pos]}')
        for plot_label in csv_values_dict.keys():
            
            x = csv_values_dict[plot_label]['sma'][:max_data_len]
            y = csv_values_dict[plot_label][f'{prof}'][:max_data_len]
            y_error = csv_values_dict[plot_label][f'{prof}_err']
            
            # Isohpohes pos angle is in rad so can be changed to deg
            if profile_axis_label[prof_pos] == 'Position Angle [deg]':
                ang_deg_abs = y * 180 / np.pi
                # Angles measured from x righ-hand axis
                angle_deg = ang_deg_abs % 360
                y = [ang-180 if ang>180 else ang for ang in angle_deg]                
                y_error = csv_values_dict[plot_label][f'{prof}_err'] * 180 / np.pi
            
            # If we want to plot intensisty in magnitudes
            elif mag==True and prof == 'intens':
                y = values_counts_to_mag(y,cons[0],cons[1])
                
            #plt.errorbar(x,y,yerr=y_error,label=label,
            #            marker='.',linewidth=1)
            plt.scatter(x,y,label=plot_label,
                        marker='.',s=10,linewidth=2)
            
        if profile_axis_label[prof_pos] == 'Intensity [counts]':
            if mag == False: 
                ax.set_yscale('log')
            
            else:
                plt.ylabel(f'mu [mag/arcsec^2]')
                ax.invert_yaxis()

        plt.legend(loc='lower right')

    fig_name_final = f'{galaxy}_{fig_name}_profiles_counts.pdf'
    if mag == True:
        fig_name_final = f'{galaxy}_{fig_name}_profiles_mag.pdf'
    fig_path = f'{cwd}/{galaxy}/{fig_name_final}'
    plt.savefig(f'{fig_path}', format='pdf', dpi=1000, bbox_inches='tight')
    

def isophote_fitting(galaxy,img_gal_path,gal_center,img_mask_path=None):
    
    print('Performing the isophote fitting\n')
    
    # Loading the galaxy image
    _,gal_img,gal_img_name = open_fits(img_gal_path)
    gal_img_fit = gal_img
    gal_img_log_or = np.log10(gal_img_fit)
    gal_img_log = np.log10(gal_img_fit)
    
    # Loading the mask if it is required
    if img_mask_path != None:
        # Loading the mas
        _,mask_img,_ = open_fits(img_mask_path)
        # Converting from 0/1 mask to True/False mask
        mask_img = mask_img == 1
    
        # Masked galaxy image
        img_gal_mask = np.ma.masked_array(gal_img, mask=mask_img)
        gal_img_fit = img_gal_mask
    

    # Logaritmic data to show details
    gal_img_log = np.log10(gal_img_fit)
    
    fig, (ax1) = plt.subplots(figsize=(14, 5), nrows=1, ncols=1)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
    ax1.imshow(gal_img_log, origin='lower')
    
    
    fig_name = f'{gal_img_name}_image_analyze.pdf'
    fig_path = f'{cwd}/{galaxy}/{fig_name}'
    plt.savefig(f'{fig_path}', dpi=1000, bbox_inches='tight')
    
    # While loop beacuse we don't want to fix ellipse initial conditions
    # So the loop is active until the programe converges
    isophote_table = []
    pa_ind = 0
    while len(isophote_table) == 0:
        # Defining a elliptical geometry
        pa_range = np.linspace(20,160,50)
        geometry = EllipseGeometry(x0=gal_center[0], y0=gal_center[1],
                                   sma=50, # semimajor axis in pixels
                                   eps=0.8,
                                   pa=pa_range[pa_ind] * np.pi / 180.0) # position angle in radians
        
        aperture = EllipticalAperture((geometry.x0, geometry.y0),
                                      geometry.sma,
                                      geometry.sma * (1 - geometry.eps),
                                      geometry.pa)  
        
        # Fiting the isophotes by using ellipses
        fit_step = 0.01
        #fit_step = 0.1
        ellipse = Ellipse(gal_img_fit, geometry)
        isolist = ellipse.fit_image(step=fit_step,
                                    minit=20,
                                    sclip=2,
                                    nclip=10,
                                    fix_center=True,
                                    fflag=0.5)
        
        # We can generate a table with the results
        isophote_table = isolist.to_table()
        
        pa_ind += 1
        
        if pa_ind == len(pa_range):
            print('Isophote fitting cannot converge')
            break
                
    # Export it as a csv
    isophote_table_name = f'{gal_img_name}_isophote.csv'
    isophote_table_path = f'{cwd}/{galaxy}/{isophote_table_name}'
    isophote_table.write(f'{isophote_table_path}', format='csv', overwrite=True)
    
    # Creating some figures    
    if 'model' not in gal_img_name:
        plot_list = [(isophote_table_path,'Data')]
        plot_profiles(galaxy,plot_list,'i')
    else:
        plot_list = [(isophote_table_path,'Model')]
        plot_profiles(galaxy,plot_list,'i_model')
    
    fig, (ax1) = plt.subplots(figsize=(14, 5), nrows=1, ncols=1)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
    ax1.imshow(gal_img_log_or, origin='lower')
    
    smas = np.linspace(10, 200, 20)
    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        ax1.plot(x, y, color='white',linewidth=0.5)
    
    fig_name = f'{gal_img_name}_ellipses.pdf'
    fig_path = f'{cwd}/{galaxy}/{fig_name}'
    plt.savefig(f'{fig_path}', format='pdf', dpi=1000, bbox_inches='tight')
    plt.close()
    
    return isophote_table_path

def create_galaxy_folder(galaxy):
    
    # Creating a folder for each galaxy
    galaxy_folder = f'{galaxy}'
    galaxy_folder_path = f'{cwd}/{galaxy_folder}'
    
    # If the folder was created previously is remvoved
    if os.path.isdir(f'{galaxy_folder_path}') == True:
        shutil.rmtree(f'{galaxy_folder_path}')
    os.mkdir(f'{galaxy_folder_path}')

    return galaxy_folder_path

def fits_mag_to_counts(fits_path,inst,zcal):
    
    hdr_mag,img_mag,fits_name = open_fits(fits_path)

    # Flux from magnitudes to counts
    fits_flux_img = 10**((img_mag - 2.5*np.log10(inst**2)-zcal)/-2.5)
    
    # Expoting the new fits
    fits_name = fits_name.split('mag')[0]
    fits_flux_name = f'{fits_name}_counts.fits'
    fits_flux_path = f'{cwd}/{galaxy}/{fits_flux_name}'
    fits.writeto(f'{fits_flux_path}', fits_flux_img, header=hdr_mag, overwrite=True)
    
    return fits_flux_path

def fits_counts_to_mag(fits_path,inst,zcal):
    
    hdr_flux,img_flux,fits_name = open_fits(fits_path)

    # Flux from counts to mag/arcsec^2
    fits_mag_img = -2.5*np.log10(img_flux) - 2.5 * np.log10(inst**2) - zcal
    
    # Expoting the new fits
    fits_name = fits_name.split('counts')[0]
    fits_mag_name  = f'{fits_name}mag.fits'
    fits_mag_path = f'{cwd}/{galaxy}/{fits_mag_name}'
    fits.writeto(f'{fits_mag_path}', fits_mag_img, header=hdr_flux, overwrite=True)
    
    return fits_mag_path  

def values_counts_to_mag(val_counts,inst,zcal):
    val_mag = -2.5*np.log10(val_counts) - 2.5 * np.log10(inst**2) - zcal
    
    return val_mag
    
def galaxy_index(df_info,galaxy):

    index = df_info.loc[df_info['galaxy'] == galaxy].index.values[0]
    return index

def create_psf(fits_path,galaxy,df_sky,df_psf):
    
    print('Creating a PSF')
    
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
    fits_psf_name = f'{galaxy}_psf.fits'
    fits_psf_path = f'{cwd}/{galaxy}/{fits_psf_name}'

    subprocess.run(['makeimage',
                    psf_conf_file_path,
                    '-o', fits_psf_path])
    
    # Opening the psf to normalizate it
    hdr_psf,img_psf,fits_name = open_fits(fits_psf_path)
    
    # Dividing between the total flux
    psf_total_flux = np.nansum(img_psf)
    img_psf_norm = img_psf / psf_total_flux
    
    # Saving the results
    fits_psf_norm_name = f'{galaxy}_psf_norm.fits'
    fits_psf_norm_path = f'{cwd}/{galaxy}/{fits_psf_norm_name}'
    
    fits.writeto(f'{fits_psf_norm_path}',img_psf_norm,hdr_psf,overwrite=True)

    return fits_psf_norm_path

def extract_data_hdr(best_file):

    fit_par_list = []
    with open(best_file, "r+b") as f:
        map_file = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        for line in iter(map_file.readline, b''):
            if re.findall(br'FUNCTION\b', line):
                func = str(line.split(br' ')[1].strip(), 'utf-8')  
                fit_par_list.append([func])

    with open(best_file, "r+b") as f:
        map_file = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        fit_par_list.insert(0, ['Center'])  
        func_index = 0  

        for line in iter(map_file.readline, b''):
 
            match = re.search(br'(\S+)\s+([\d\.\-eE]+)\s+#\s+\+/-\s+([\d\.\-eE]+)', line)
            if match:
                par = match.group(1).decode('utf-8') 
                val = float(match.group(2)) 
                err = float(match.group(3))  
                fit_par_list[func_index].append((par, val, err))
            
            elif re.findall(br'FUNCTION\b', line):
                func_index += 1        
        
    
    fit_min_par = {}
    patterns = {
        'chi2': re.compile(br'Reduced value:\s+([\d\.\-eE]+)'),
        'AIC': re.compile(br'AIC:\s+([\d\.\-eE]+)'),
        'BIC': re.compile(br'BIC:\s+([\d\.\-eE]+)')
        }

    with open(best_file, "r+b") as f:
        map_file = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        for line in iter(map_file.readline, b''):
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    fit_min_par[key] = float(match.group(1)) 
    
    return fit_par_list,fit_min_par
    
def sersic_profile(r, I_e, R_e, n):

    b_n = 2 * n - 1 / 3 + 0.009876 / n
    I_r = I_e * np.exp(-b_n * ((r / R_e) ** (1 / n) - 1))
    
    return I_r

def exponential_disk(r, I_0, h):
    I_r = I_0 * np.exp(-r / h)
    return I_r

def main(gal_pos,galaxy):
    
    # Creating a folder for each galaxy
    galaxy_folder_path = create_galaxy_folder(galaxy)
    
    # Fits file to analyze
    fits_file = fits_image_list[gal_pos]
    fits_path = f'{files_path}/{fits_file}'
    
    # Opening the galaxy image 
    hdr_gal,img_gal,fits_name = open_fits(fits_path)

    # obtaining the shape of the fits
    x_len = img_gal.shape[1]
    y_len = img_gal.shape[0]
    
    # Mask file 
    mask_file = fits_mask_list[gal_pos]
    mask_path = f'{files_path}/{mask_file}'
    
    # Opening the mask
    hdr_mask,img_mask,mask_name = open_fits(mask_path)
    
    # Changing from 0/1 mask to True/False mask
    # Values with True are not selected from image
    img_mask = img_mask == 1
    
    # OBTAINING THE CENTER OF THE GALAXY
    
    # Cropping to focus on the galaxy center
    # For a centered image of a galaxy
    crop_factor = 10
    x_len_crop = (x_len//crop_factor)
    y_len_crop = (y_len//crop_factor)
    
    img_crop_center = img_gal[x_len//2 - x_len_crop:x_len//2 + x_len_crop, y_len//2 - y_len_crop:y_len//2 + y_len_crop]
    max_pix_value_center = np.nanmax(img_crop_center)
    
    # For a non centered image galaxy or for a masked one 
    # Applying the mask to the image to remove stars
    img_gal_mask = np.ma.masked_array(img_gal,mask=img_mask)

    # Finding the value in the image
    max_pix_value_center = np.nanmax(img_gal_mask)
    x_center = np.where(img_gal == max_pix_value_center)[1][0]+1
    y_center = np.where(img_gal == max_pix_value_center)[0][0]+1

    # How much deviation we allow for the center
    center_dev = 1
    
    # Initial conditions derivied from a elliptical fitting
    gal_iso_fit_csv_path = isophote_fitting(galaxy,fits_path,(x_center,y_center),
                     img_mask_path=mask_path)
    
    # Position angle
    pos_ang = [105,0,180]
    
    # Ellipticity
    ellip = [0.5,0,180]
    
    # Functions to decompose the profile
    funct_fit = ['Sersic',
                 'Exponential']#,
                 #'FerrersBar2D']
    
    # SERSIC FUNCTION: Bulge
    # Sersic Index
    ser_ind = [3,0.6,6]
    
    # Effective Raidus
    rad_ef = [60,0,200]
    # Intensity at effective radius
    int_ef = [120,0,200]
    
    # EXPONENTIAL FUNCTION: Disk
    # Center intensity
    I_0_disk = max_pix_value_center
    int_cent_disk = [I_0_disk,0,I_0_disk+I_0_disk*0.25]
    
    # Length scale disk
    len_sca_disk = [80,0,200]
    
    # FERRERS 2D BAR
    # Bar profile index
    bar_index = [1,2,5] 
    
    # Radius bar
    bar_rad = [50,0,200]
    
    # Central intensity of the bar
    I_0_bar = max_pix_value_center
    int_cent_bar = [I_0_bar,0,I_0_bar+I_0_bar*0.25]
    
    # Configuration file
    conf_file_name = f'{galaxy}_conf_imfit.txt'
    conf_file_path = f'{galaxy_folder_path}/{conf_file_name}'
    conf_file = open(f'{conf_file_path}','w+')
    create_conf_imfit(file=conf_file,
                     funct=funct_fit,
                     galaxy = galaxy,
                     img_center_x = [x_center,x_center-center_dev,x_center+center_dev],
                     img_center_y = [y_center,y_center-center_dev,y_center+center_dev],
                     pos_ang = pos_ang,
                     ellip = ellip,
                     n=ser_ind,
                     r_e=rad_ef,
                     I_e=int_ef,
                     I_0=int_cent_disk,
                     h=len_sca_disk,
                     n_bar = bar_index,
                     a_bar = bar_rad,
                     c0=int_cent_bar)
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
    fits_model_name = f'{fits_name}_model_counts.fits'
    fits_model_path = f'{galaxy_folder_path}/{fits_model_name}'
    
    # Output files
    residual_model_name = f'{fits_name}_model_residual_counts.fits'
    residual_model_path = f'{galaxy_folder_path}/{residual_model_name}'

    
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
                    '--save-model=',fits_model_path,
                    '--save-residual=',residual_model_path])
    
    # Returning to the main directory to continue the programe
    os.chdir(f'{cwd}')
    
    best_parameters_name = 'bestfit_parameters_imfit.dat'
    best_parameters_path = f'{cwd}/{galaxy}/{best_parameters_name}'
    fit_par_list,fit_min_par = extract_data_hdr(best_parameters_path)
    
    # Instrumental pixel scale
    inst_arcsec_pix = df_sky_info.loc[galaxy_sky_index]['inst']
    
    # Calibration constant
    zcal = df_sky_info.loc[galaxy_sky_index]['zcal']
    
    fits_model_mag_path = fits_counts_to_mag(fits_model_path,inst_arcsec_pix,zcal)
    
    mod_iso_fit_csv_path = isophote_fitting(galaxy,fits_model_path,(x_center,y_center))
    
    # Comparing the data profile with the model profile
    plot_list = [(gal_iso_fit_csv_path,'Data'),
                 (mod_iso_fit_csv_path,'Model')]
    plot_profiles(galaxy,plot_list,'compar')
    plot_profiles(galaxy,plot_list,'compar',mag=True,cons=(inst_arcsec_pix,zcal))
    
    plt.close()

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
                df_sky_info = pd.read_csv(csv_path)
            elif 'psf' in file:
                df_psf_info = pd.read_csv(csv_path)
    
    if len(galaxy_list) != 0:
        for gal_pos,galaxy in enumerate(galaxy_list):
            
            print(f'\nAnalyzing the galaxy {galaxy}\n')
            
            main(gal_pos,galaxy)    
            
            print('\n#-------------------------#\n')    

    else:
        
        print('There is no galaxies in the directory')
    
    print('The analysis for all the galaxies is finished')
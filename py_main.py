# This programe runs imfit

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.signal import deconvolve

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import functools

from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

import os
import re
import sys
import subprocess
import shutil
import mmap
import struct

import pdb
import time
import warnings
warnings.filterwarnings('ignore')

from py_conf_imfit import create_conf_imfit
from py_conf_psf import create_conf_psf

from photutils.isophote import EllipseGeometry
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse

'''#-------------NEEDED FUNCTIONS-------------#'''

def create_galaxy_folder(galaxy):
    
    # Creating a folder for each galaxy
    galaxy_folder = f'{galaxy}'
    galaxy_folder_path = f'{cwd}/{galaxy_folder}'
    
    max_version = 0
    for folder in sorted(os.listdir(f'{cwd}')):
        if re.match(rf'{galaxy}_\d+',folder):
            version = int(folder.split('_')[1])
            if version > max_version:
                max_version = version
                
            if '-ow' in sys.argv or '-ow-path' in sys.argv:
                shutil.rmtree(f'{folder}')
    
    actual_version = max_version + 1
    if '-ow' in sys.argv or '-ow-path' in sys.argv:
        actual_version = 0
    
    galaxy_folder = f'{galaxy}_0{actual_version}'    
    galaxy_folder_path = f'{cwd}/{galaxy_folder}'
    
    os.mkdir(f'{galaxy_folder_path}')
    
    return galaxy_folder_path



def open_fits(fits_path):
    
    fits_name = fits_path.split('/')[-1].split('.')[0]
    # Open the fits with Astropy
    hdu = fits.open(f'{fits_path}')
    hdr = hdu[0].header
    img = hdu[0].data
    
    return (hdr,img,fits_name)

def plot_profiles(galaxy,csv_path_list,fig_name,
                  cons=None,final_plot=False):
    
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    kpc_per_arcsec = 1./cosmo.arcsec_per_kpc_proper(0.005987)

    
    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')  # Use a serif font for LaTeX rendering
    plt.rc('font', size=14)  # Adjust size to your preference
    # Define the LaTeX preamble with siunitx
    plt.rcParams['text.latex.preamble'] = r'''
                \usepackage{siunitx}
                \sisetup{
                  detect-family,
                  separate-uncertainty=true,
                  output-decimal-marker={.},
                  exponent-product=\cdot,
                  inter-unit-product=\cdot,
                }
                \DeclareSIUnit{\cts}{cts}
                '''
    
    
    csv_values_dict = {}
    df_len_list = []
    for csv_label in csv_path_list:
        df = pd.read_csv(csv_label[0])
        csv_values_dict[csv_label[1]] = df
        df_len_list.append(len(df))

    max_data_len = min(df_len_list)

    profile_to_plot = ['ellipticity','pa','intens']
    profile_axis_label = ['Ellipticity',
                          'Position Angle [deg]',
                          'Intensity [counts]']
    
    if final_plot == True:
        
        profile_to_plot = ['intens']
        profile_axis_label = ['Intensity [counts]']
        
    # Creating some figures
    plot_rows = 1
    plot_cols = len(profile_to_plot)
    
    fig = plt.figure(figsize=(6*plot_cols, 5*plot_rows))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    colors = ['gold','deepskyblue','lime','deeppink']
    markers = ['o','v','*','X']
    
    # Every iteration of this loop is a new plot
    for prof_pos,prof in enumerate(profile_to_plot):
        color_index = 0
        y_max = -10**20
        y_min = 10**20 
        ax = plt.subplot(plot_rows, plot_cols, prof_pos+1)
        
        for plot_label in csv_values_dict.keys():

            if plot_label == 'func_df':
                total_int_label_list = []
                for col in csv_values_dict[plot_label].columns:
                        if col != 'sma' and col != 'Total_int':
                            total_int_label_list.append(col)  
                result = ' + '.join(f'{label}' for label in total_int_label_list)

                for col in csv_values_dict[plot_label].columns:
        
                    if col != 'sma':
                        
                        x = csv_values_dict[plot_label]['sma'][:max_data_len]
                        y = csv_values_dict[plot_label][f'{col}'][:max_data_len]
                        
                        if max(y) > y_max:
                            y_max = max(y)
                        if min(y) < y_min:
                            y_min = min(y)

                        if col == 'Total_int':
                            
                            plt.plot(x,y,label=f'{result}',
                                    linewidth=1,color='black',
                                    linestyle='-.',zorder=4)
                            
                        else:
                            plt.scatter(x,y,label=col,
                                        marker=markers[color_index],s=10,
                                        linewidth=0.15,edgecolor='black',
                                        color=colors[color_index],zorder=1)
                            color_index += 1

            else:
                
                x = csv_values_dict[plot_label]['sma'][:max_data_len]
                y = csv_values_dict[plot_label][f'{prof}'][:max_data_len]
                
                # Isohpohes pos angle is in rad so can be changed to deg
                if profile_axis_label[prof_pos] == 'Position Angle [deg]':
                    angle_deg = rad_to_deg(y)
                    angle_deg = angle_deg % 360
                    # Angles measured from x righ-hand axis
                    y = [(ang+90) if (ang<90) 
                         else (ang-90) if (90<ang<270) 
                         else (ang-270) if (ang>270) 
                         else ang for ang in angle_deg]                
                    y_error = csv_values_dict[plot_label][f'{prof}_err'] * 180 / np.pi
                
                if max(y) > y_max:
                        y_max = max(y)
                if min(y) < y_min:
                        y_min = min(y)
                
                z_order = 2
                marker_size = 11
                if plot_label == 'Model':
                    z_order = 3
                    marker_size = 14
                plt.scatter(x,y,label=plot_label,
                            marker=markers[color_index],s=marker_size,
                            linewidth=0.15,edgecolor='black',
                            color=colors[color_index],zorder=z_order)
                color_index += 1

        if prof == 'intens':
            plt.legend(loc='upper right',prop={'size': 10})
        
        else:
            plt.legend(loc='lower right',prop={'size': 10})


        # Customizing the plots       
        # X bottom axis is common
        ax.set_xlabel(r'$Semimajor\ Axis\ Length\ [\mathrm{pix}]$')
        
        ax.set_xticks(np.linspace(np.min(x), np.max(x), 7))
        ax.set_xmargin(0.1)
        ax.minorticks_on()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        ax.grid(True,alpha=0.2)
        
        # TOP X axis is common
        axxtop = ax.secondary_xaxis('top',
                                    functions=(functools.partial(px_to_kpc,inst=cons[0]),
                                               functools.partial(kpc_to_px,inst=cons[0])))
        
        px_ticks = ax.get_xticks()
        arcsec_ticks = px_to_kpc(px_ticks,inst=cons[0])
        axxtop.set_xticks(arcsec_ticks)
        
        axxtop.minorticks_on()
        
        axxtop.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        axxtop.set_xlabel(r'$Semimajor\ Axis\ Length\ [\mathrm{kpc}]$',labelpad=8)
        axxtop.tick_params(axis='x', which='major')
        
        # Y axes are different for each plot
        # For intensisty
        if prof == 'intens':
        
            # Y left axis
            ax.set_ylabel(r'$Intensity\ [\mathrm{counts}]$')
            ax.set_yticks(np.linspace(np.min(y_min), np.max(y_max), 7))
            ax.minorticks_on()
            
            ax.set_yscale('log')
            ax.set_ylim(bottom=0.5)
            
            # Y right axis
            axyrig = ax.secondary_yaxis('right',
                                        functions=(functools.partial(values_counts_to_mag,inst=cons[0],zcal=cons[1]),
                                                   functools.partial(values_mag_to_counts,inst=cons[0],zcal=cons[1])))
            
            counts_ticks = ax.get_yticks()
            mag_ticks = values_counts_to_mag(counts_ticks,inst=cons[0],zcal=cons[1])
            axyrig.set_yticks(mag_ticks)
            axyrig.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axyrig.yaxis.set_minor_formatter(plt.NullFormatter())
            
            axyrig.minorticks_on()
            
            axyrig.set_ylabel(r'$\mu\ [\mathrm{mag/arcsec^2}]$')
            axyrig.tick_params(axis='y', which='major')
            
        elif prof == 'ellipticity':
            
            # Y left axis
            ax.set_ylabel(r'$Ellipticity$')
            ax.set_yticks(np.linspace(np.min(y_min), np.max(y_max), 7))
            ax.minorticks_on()
            
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # Y right axis
            axyrig = ax.secondary_yaxis('right',functions=(ell_to_axrat,axrat_to_ell))
            
            ell_ticks = ax.get_yticks()
            axrat_ticks = ell_to_axrat(ell_ticks)
            axyrig.set_yticks(axrat_ticks)
            axyrig.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axyrig.yaxis.set_minor_formatter(plt.NullFormatter())
            
            axyrig.minorticks_on()
            
            axyrig.set_ylabel(r'$Axis\ ratio$')
            axyrig.tick_params(axis='y', which='major')
            
            
        elif prof == 'pa':
        
            # Y left axis
            ax.set_ylabel(r'$Position\ Angle\ [\mathrm{deg}]$')
            ax.set_yticks(np.linspace(np.min(y_min), np.max(y_max), 7))
            ax.minorticks_on()
            
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # Y right axis
            axyrig = ax.secondary_yaxis('right',functions=(deg_to_rad,rad_to_deg))
            
            deg_ticks = ax.get_yticks()
            rad_ticks = deg_to_rad(deg_ticks)
            axyrig.set_yticks(list(rad_ticks))
            axyrig.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axyrig.yaxis.set_minor_formatter(plt.NullFormatter())
            
            axyrig.minorticks_on()
            
            axyrig.set_ylabel(r'$Position\ Angle\ [\mathrm{rad}]$')
            axyrig.tick_params(axis='y', which='major')
            
            
    if '-png' in sys.argv:
        img_format = 'png'
    else: 
        img_format = 'pdf'

    fig_name_final = f'{galaxy}_{fig_name}_profiles.{img_format}'
    fig_path = f'{galaxy_folder_path}/{fig_name_final}'
    plt.savefig(f'{fig_path}', format=img_format, dpi=1000, bbox_inches='tight')    
    plt.close()    
        
def plot_fit_func(galaxy,fit_par_list,rad_range):

    int_sum = np.zeros(len(rad_range))
    
    fig = plt.figure(figsize=(5, 5))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    ax = plt.subplot(1, 1, 1)

    int_func_df = pd.DataFrame(columns=[])
    int_func_df['sma'] = rad_range


    for func_par in fit_par_list:
        
        if 'Center' in func_par:
            cen_ind_list = next((i for i, item in enumerate(fit_par_list) if item[0] == 'Center'), None)
            
            cen_par_val = {par[0]: par[1] for par in fit_par_list[cen_ind_list][1:]} 
            cen_par_err = {par[0]: par[2] for par in fit_par_list[cen_ind_list][1:]} 
            
            x0_val = cen_par_val['X0']
            y0_val = cen_par_val['Y0']
            
            x0_err = cen_par_err['X0']
            y0_err = cen_par_err['Y0']
        
        elif 'Sersic' in func_par:
            ser_ind_list = next((i for i, item in enumerate(fit_par_list) if item[0] == 'Sersic'), None)
            
            ser_par_val = {par[0]: par[1] for par in fit_par_list[ser_ind_list][1:]} 
            ser_par_err = {par[0]: par[2] for par in fit_par_list[ser_ind_list][1:]} 
            
            n_val = ser_par_val['n']
            I_e_val = ser_par_val['I_e']
            r_e_val = ser_par_val['r_e']
            
            n_err = ser_par_err['n']
            I_e_err = ser_par_err['I_e']
            r_e_err = ser_par_err['r_e']
            
            ser_int_func = sersic_profile(rad_range,n_val,r_e_val,I_e_val)
            int_func_df['Sersic'] = ser_int_func
            
            int_sum = int_sum + ser_int_func
            
            plt.scatter(rad_range,ser_int_func,label='Sersic',
                        marker='o',s=10,
                        linewidth=0.15,edgecolor='black',
                        color='gold',zorder=1)
            
        elif 'Exponential' in func_par:
            exp_ind_list = next((i for i, item in enumerate(fit_par_list) if item[0] == 'Exponential'), None)
            
            exp_par_val = {par[0]: par[1] for par in fit_par_list[exp_ind_list][1:]} 
            exp_par_err = {par[0]: par[2] for par in fit_par_list[exp_ind_list][1:]} 
            
            I_0_val = exp_par_val['I_0']
            h_val = exp_par_val['h']
            
            I_0_err = exp_par_err['I_0']
            h_err = exp_par_err['h']
            
            exp_int_func = exponential_disk(rad_range,I_0_val,h_val)
            int_func_df['Exponential'] = exp_int_func
            
            int_sum = int_sum + exp_int_func
    
            plt.scatter(rad_range,exp_int_func,label='Exponential',
                        marker='v',s=10,
                        linewidth=0.15,edgecolor='black',
                        color='dodgerblue',zorder=1)
    
    if len(fit_par_list) > 2:
        
        int_func_df.insert(loc=1,column='Total_int',value=int_sum)
        
    plt.plot(rad_range,int_sum,label='Total Intensity',
                linewidth=1,color='black',
                linestyle='-.',zorder=3)

    plt.xlabel('Semimajor Axis Length [pix]')
    plt.ylabel('Intensity [counts]')
    plt.yscale('log')
    ax.set_ylim(bottom=0.5)   
    plt.legend(loc='upper right')
    
    if '-png' in sys.argv:
        img_format = 'png'
    else: 
        img_format = 'pdf'
    
    fig_name = f'{galaxy}_fit_func.{img_format}'
    fig_path = f'{galaxy_folder_path}/{fig_name}'
    plt.savefig(f'{fig_path}', format=img_format, dpi=1000, bbox_inches='tight')
    plt.close()
    
    csv_name = f'{galaxy}_fit_funct.csv'
    csv_path = f'{galaxy_folder_path}/{csv_name}'
    int_func_df.to_csv(csv_path,header=True,index=False)
    
    if len(int_func_df) != 0:
        return csv_path
    else:
        print('No fitting functions were plotted')
        return None

def plot_images(gal_img,fig_name,cons,
                path=False,
                ellip=False,isolist=None,
                mag=False,res=False):
    
    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')  # Use a serif font for LaTeX rendering
    plt.rc('font', size=14)  # Adjust size to your preference
    # Define the LaTeX preamble with siunitx
    plt.rcParams['text.latex.preamble'] = r'''
                \usepackage{siunitx}
                \sisetup{
                  detect-family,
                  separate-uncertainty=true,
                  output-decimal-marker={.},
                  exponent-product=\cdot,
                  inter-unit-product=\cdot,
                }
                \DeclareSIUnit{\cts}{cts}
                '''
    
    if path == True:
        
        print('Path was passed to function to plot the image')
        _,gal_img_mag,_ = open_fits(gal_img)
        
    else:
            
        gal_img_mag = gal_img
        
    if mag==False:
        gal_img_mag = values_counts_to_mag(gal_img_mag,cons[0],cons[1])
    
    x_len = gal_img_mag.shape[1]
    y_len = gal_img_mag.shape[0]
    
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(1,1,1)
    if res == True:
        im = ax.imshow(gal_img_mag, origin='lower',cmap='viridis_r',
                       vmin=-1,vmax=1)
    else: 
        im = ax.imshow(gal_img_mag, origin='lower',cmap='viridis_r')
    plt.colorbar(im, pad=0.11)
    fig.axes[1].invert_yaxis()
    fig.axes[1].set(ylabel=r'$\mu\ [\mathrm{mag/arcsec^2}]$')
    fig.axes[1].minorticks_on()
    
    if ellip == True:
    
        smas = np.linspace(10, 200, 20)
        for sma in smas:
            iso = isolist.get_closest(sma)
            x, y, = iso.sampled_coordinates()
            plt.plot(x, y, color='white',linewidth=0.5)
    
    # Customizing the plots       
    # X bottom axis is common
    ax.set_xlabel(r'$X\ dimension\ [\mathrm{pix}]$')
    ax.set_xlim(left=0.0)
    
    ax.set_xticks(np.linspace(0, x_len, 7))
    ax.minorticks_on()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    #ax.grid(True,alpha=0.2)
    
    # TOP X axis is common
    axxtop = ax.secondary_xaxis('top',
                                functions=(functools.partial(px_to_kpc,inst=cons[0]),
                                           functools.partial(kpc_to_px,inst=cons[0])))
    
    px_ticks = ax.get_xticks()
    arcsec_ticks = px_to_kpc(px_ticks,inst=cons[0])
    axxtop.set_xticks(arcsec_ticks)
    
    axxtop.minorticks_on()
    
    axxtop.set_xlabel(r'$X\ dimension\ [\mathrm{kpc}]$',labelpad=8)
    axxtop.tick_params(axis='x', which='major')
    
    # Y left axis
    ax.set_ylabel(r'$Y\ dimension\ [\mathrm{pix}]$')
    ax.set_yticks(np.linspace(0, y_len, 7))
    
    ax.minorticks_on()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    # Y right axis
    
    axyrig = ax.secondary_yaxis('right',
                                functions=(functools.partial(px_to_kpc,inst=cons[0]),
                                           functools.partial(kpc_to_px,inst=cons[0])))
    
    px_ticks = ax.get_yticks()
    arcsec_ticks = px_to_kpc(px_ticks,inst=cons[0])
    axyrig.set_yticks(arcsec_ticks)
    
    axyrig.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axyrig.yaxis.set_minor_formatter(plt.NullFormatter())
    
    axyrig.minorticks_on()
    
    axyrig.set_ylabel(r'$Y\ dimension\ [\mathrm{kpc}]$')
    axyrig.tick_params(axis='y', which='major')
    
    if '-png' in sys.argv:
        img_format = 'png'
    else: 
        img_format = 'pdf'

    fig_name = f'{fig_name}.{img_format}'
    fig_path = f'{galaxy_folder_path}/{fig_name}'
    plt.savefig(f'{fig_path}',format=img_format, dpi=1000, bbox_inches='tight')
    plt.close()

def isophote_fitting(galaxy,img_gal_path,gal_center,img_mask_path=None,cons=None):
    
    print('Performing the isophote fitting\n')
    
    # Loading the galaxy image
    _,gal_img,gal_img_name = open_fits(img_gal_path)
    
    gal_img_fit = gal_img
    
    # Original image
    gal_img_log_or = np.log10(gal_img_fit)
    
    # Loading the mask if it is required
    if img_mask_path != None:
        # Loading the mas
        _,mask_img,_ = open_fits(img_mask_path)
        # Converting from 0/1 mask to True/False mask
        mask_img = mask_img == 1
    
        # Masked galaxy image
        img_gal_mask = np.ma.masked_array(gal_img, mask=mask_img)
        gal_img_fit = img_gal_mask
    
    
    fig_name = f'{gal_img_name}_image_analyze'
    plot_images(gal_img_fit,fig_name,cons=cons)
    
    
    # While loop beacuse we don't want to fix ellipse initial conditions
    # So the loop is active until the programe converges
    isophote_table = []
    pa_ind = 0
    while len(isophote_table) == 0:
        # Defining a elliptical geometry
        pa_range = np.linspace(5,175,60)
        print(f'Atempting to converge with PA={pa_range[pa_ind]:.2f} deg')
        
        geometry = EllipseGeometry(x0=gal_center[0], y0=gal_center[1],
                                   sma=50, # semimajor axis in pixels
                                   eps=0.5,
                                   pa=pa_range[pa_ind] * np.pi / 180.0) # position angle in radians
        
        aperture = EllipticalAperture((geometry.x0, geometry.y0),
                                      geometry.sma,
                                      geometry.sma * (1 - geometry.eps),
                                      geometry.pa)  
        
        # Fiting the isophotes by using ellipses
        fit_step = 0.1
        if '-hr' in sys.argv:
            fit_step = 0.01
            
        ellipse = Ellipse(gal_img_fit, geometry)
        isolist = ellipse.fit_image(step=fit_step,
                                    minit=30,
                                    #sclip=2,
                                    #nclip=10,
                                    fix_center=True,
                                    #fflag=0.5
                                    )
        
        # We can generate a table with the results
        isophote_table = isolist.to_table()
        
        pa_ind += 1

        if pa_ind == len(pa_range):
            print('Isophote fitting cannot converge')
            break
                
    # Export it as a csv
    isophote_table_name = f'{gal_img_name}_isophote.csv'
    isophote_table_path = f'{galaxy_folder_path}/{isophote_table_name}'
    isophote_table.write(f'{isophote_table_path}', format='csv', overwrite=True)
    
    # Creating some figures    
    if 'model' not in gal_img_name:
        plot_list = [(isophote_table_path,'Data')]
        plot_profiles(galaxy,plot_list,'i',cons=cons)
    else:
        plot_list = [(isophote_table_path,'Model')]
        plot_profiles(galaxy,plot_list,'i_model',cons=cons)
    
    
    fig_name = f'{gal_img_name}_ellipses'
    plot_images(gal_img,fig_name,cons=cons,ellip=True,isolist=isolist)
    
    return isophote_table_path



def fits_mag_to_counts(fits_path,inst,zcal):
    
    hdr_mag,img_mag,fits_name = open_fits(fits_path)

    # Flux from magnitudes to counts
    fits_flux_img = 10**((img_mag - 2.5*np.log10(inst**2)-zcal)/-2.5)
    
    # Expoting the new fits
    fits_name = fits_name.split('mag')[0]
    fits_flux_name = f'{fits_name}_counts.fits'
    fits_flux_path = f'{galaxy_folder_path}/{fits_flux_name}'
    fits.writeto(f'{fits_flux_path}', fits_flux_img, header=hdr_mag, overwrite=True)
    
    return fits_flux_path

def fits_counts_to_mag(fits_path,inst,zcal):
    
    hdr_flux,img_flux,fits_name = open_fits(fits_path)

    # Flux from counts to mag/arcsec^2
    img_flux[img_flux<=0] = 1
    fits_mag_img = -2.5*np.log10(img_flux) - 2.5 * np.log10(inst**2) - zcal
    
    # Expoting the new fits
    fits_name = fits_name.split('counts')[0]
    fits_mag_name  = f'{fits_name}_mag.fits'
    fits_mag_path = f'{galaxy_folder_path}/{fits_mag_name}'
    fits.writeto(f'{fits_mag_path}', fits_mag_img, header=hdr_flux, overwrite=True)
    
    return fits_mag_path  

# Changing from counts to magnitudes
def values_counts_to_mag(val_counts,inst,zcal):
    
    if isinstance(val_counts, (list,np.ndarray)):
        
        val_counts[val_counts<=0]=1
        val_mag = -2.5*np.log10(val_counts) - 2.5 * np.log10(inst**2) - zcal
        
    elif isinstance(val_counts, (int, float, np.float64)) and (val_counts<=0 or np.isnan(val_counts)):

        val_counts = 1
        val_mag = -2.5*np.log10(val_counts) - 2.5 * np.log10(inst**2) - zcal
    
    elif isinstance(val_counts, (int, float, np.float64)) and val_counts>0:

        val_mag = -2.5*np.log10(val_counts) - 2.5 * np.log10(inst**2) - zcal
    
    else:
        print(val_counts)
        print(np.isnan(val_counts))
        print(type(val_counts))
        print('No valid value')

    return round_number(val_mag,3)

def round_number(num,dec):
    
    if isinstance(num, (int, float, np.float64)):
        num = round(num,dec)
    
    return num

def values_mag_to_counts(val_mag,inst,zcal):
    
    val_count = 10**((val_mag - 2.5*np.log10(inst**2)-zcal)/-2.5)
    
    return round_number(val_count,3)

# Changing from pixeles to kpc 
def px_to_kpc(px,inst):

    arcsec = px * inst 
    kpc = arcsec_to_kpc(arcsec)

    return round_number(kpc,3)

def kpc_to_px(kpc,inst):
    
    arcsec = kpc_to_arcsec(kpc)
    px = arcsec / inst 
    
    return round_number(px,3)

def arcsec_to_kpc(arcsec):
    
    kpc = arcsec * 0.12
    
    return round_number(kpc,3)

def kpc_to_arcsec(kpc):
    
    arcsec = kpc / (0.12)
    
    return round_number(arcsec,3)

# Changing from Ellipticity to Axis ratio
def ell_to_axrat(ell):
    axrat = 1 - ell
    return round_number(axrat,3)

def axrat_to_ell(axrat):
    ell = 1 - axrat
    return round_number(ell,3)

# Changing from degrees to radian
def deg_to_rad(deg):
    
    #rad = (deg % 360) * (np.pi / 180)
    rad = (deg) * (np.pi / 180)

    return round_number(rad,3)

def rad_to_deg(rad):
    
    #deg = (rad % (2*np.pi)) * (180 / np.pi)
    deg = (rad) * (180 / np.pi)
        
    return round_number(deg,3)

def deg_isopho_to_imfit(deg):
    
    
    
    pass
    
def galaxy_index(df_info,galaxy):

    index = df_info.loc[df_info['galaxy'] == galaxy].index.values[0]
    return index

def create_psf(fits_path,galaxy,df_sky,df_psf):
    
    print('\nCreating a PSF\n')
    
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
    fwhm_value_pix = round(fwhm_value_arcsec / inst_arcsec_pix,3)
    beta_value_pix = round(beta_value_arcsec / inst_arcsec_pix,3)
    
    # Creating the configuration script for the psf
    psf_conf_file_name = f'{galaxy}_conf_psf.txt'
    psf_conf_file_path = f'{galaxy_folder_path}/{psf_conf_file_name}'
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
    fits_psf_path = f'{galaxy_folder_path}/{fits_psf_name}'

    if '--imfit-nopath' in sys.argv:

        subprocess.run(['./makeimage',
                        psf_conf_file_path,
                        '-o', fits_psf_path])
        
    else:
        
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
    fits_psf_norm_path = f'{galaxy_folder_path}/{fits_psf_norm_name}'
    
    fits.writeto(f'{fits_psf_norm_path}',img_psf_norm,hdr_psf,overwrite=True)

    return fits_psf_norm_path

def extract_data_hdr(galaxy,best_file,cons):

    fit_par_list = []
    results_df = pd.DataFrame(columns=['galaxy'])
    
    new_row = {col: (galaxy if col == 'galaxy' else np.nan) for col in results_df.columns}
    results_df.loc[len(results_df)] = new_row
    
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
                
                if fit_par_list[func_index][0] == 'Center':
                    extra_str = '_cen'
                elif fit_par_list[func_index][0] == 'Sersic':
                    extra_str = '_ser'
                elif fit_par_list[func_index][0] == 'Exponential':
                    extra_str = '_exp'

                if len(results_df.index) != 1:
                    
                    if par == 'I_e' or par == 'I_0':
                    
                        val_mag = values_counts_to_mag(val,cons[0],cons[1])
                        err_mag = values_counts_to_mag(err,cons[0],cons[1])
                        
                        results_df[f'{par}{extra_str}_counts'].iloc[-1] = val
                        results_df[f'{par}{extra_str}_mag'].iloc[-1] = val_mag
                        
                        results_df[f'{par}_err{extra_str}_counts'].iloc[-1] = err
                        results_df[f'{par}_err{extra_str}_mag'].iloc[-1] = err_mag
                        
                    elif par == 'r_e' or par == 'h':
                        
                        val_kpc = px_to_kpc(val,cons[0])
                        err_kpc = px_to_kpc(err,cons[0])
                        
                        results_df[f'{par}{extra_str}_pix'].iloc[-1] = val
                        results_df[f'{par}{extra_str}_kpc'].iloc[-1] = val_kpc
                        
                        results_df[f'{par}_err{extra_str}_pix'].iloc[-1] = err
                        results_df[f'{par}_err{extra_str}_kpc'].iloc[-1] = err_kpc
                    
                    else:
                        
                        results_df[f'{par}{extra_str}'].iloc[-1] = val
                        results_df[f'{par}_err{extra_str}'].iloc[-1] = err
                        
                else:
                    
                    if par == 'I_e' or par == 'I_0':
                    
                        val_mag = values_counts_to_mag(val,cons[0],cons[1])
                        err_mag = values_counts_to_mag(err,cons[0],cons[1])
                        
                        results_df[f'{par}{extra_str}_counts'] = val
                        results_df[f'{par}{extra_str}_mag'] = val_mag
                        
                        results_df[f'{par}_err{extra_str}_counts'] = err
                        results_df[f'{par}_err{extra_str}_mag'] = err_mag
                        
                    elif par == 'r_e' or par == 'h':
                        
                        val_kpc = px_to_kpc(val,cons[0])
                        err_kpc = px_to_kpc(err,cons[0])
                        
                        results_df[f'{par}{extra_str}_pix'] = val
                        results_df[f'{par}{extra_str}_kpc'] = val_kpc
                        
                        results_df[f'{par}_err{extra_str}_pix'] = err
                        results_df[f'{par}_err{extra_str}_kpc'] = err_kpc
                    
                    
                    else:
                        
                        results_df[f'{par}{extra_str}'] = val
                        results_df[f'{par}_err{extra_str}'] = err
                        
                
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
    
                    if len(results_df.index) != 1:
                    
                        results_df[f'{key}'].iloc[-1] = round(fit_min_par[key],3)

                    else:
                    
                        results_df[f'{key}'] = round(fit_min_par[key],3)
        
    
    
    csv_name = f'{galaxy}_i_results.csv'
    csv_path = f'{galaxy_folder_path}/{csv_name}'
    results_df.to_csv(csv_path,header=True,index=False)
    
    return fit_par_list,fit_min_par,csv_path
    
'''#-----ANALYTICAL FUNCTIONS-----#'''

def sersic_profile(r, n, R_e, I_e):

    b_n = 2 * n - 1 / 3 + 0.009876 / n
    I_r = I_e * np.exp(-b_n * ((r / R_e) ** (1 / n) - 1))
    
    return I_r

def sersic_profile_log(r, n, R_e, a):
    if 0.6<n<6 and 1<R_e<100 and 1<a<6.9:
        b_n = 2 * n - 1 / 3 + 0.009876 / n
        I_r_log = a + (-b_n * ((r / R_e) ** (1 / n) - 1))
        
    elif 0.6<n<6 and 1<R_e<100:
        b_n = 2 * n - 1 / 3 + 0.009876 / n
        I_r_log = a + (-b_n * ((r / R_e) ** (1 / n) - 1))

    else:
        I_r_log = 10e10

    return I_r_log

def exponential_disk(r, I_0, h):
    I_r = I_0 * np.exp(-r / h)
    return I_r

def exponential_disk_linear(x, m, b):
    y = m * x + b
    return y

def two_line_model(x, a1, b1, a2, b2, x_break):
    
    return np.where(x < x_break, a1 * x + b1, a2 * x + b2)

def calculate_chi_squared(x, y_obs, y_err, model, params):

    y_model = model(x, *params)  
    residuals = (y_obs - y_model) / y_err  
    chi_squared = np.sum(residuals**2)  
    return chi_squared

def initial_conditions(df, x_col, y_col,y_err_col):
    
    print('Computing the initial conditions')
    
    # PRIMERA ETAPA: SEPARACIÓN POR DOS RECTAS
    print(f'\nDos rectas')
    x_data_dos = df[x_col].values
    y_data_dos = np.log(df[y_col].values)
    y_err_dos = np.log(df[y_err_col].values)
    
    # Estimaciones iniciales para los parámetros: a1, b1, a2, b2, x_break
    initial_guess_dos = [1, 0, -1, 0, max(x_data_dos) * (1/3)]
    
    # Ajustar el modelo
    popt_dos, pcov_dos = curve_fit(two_line_model, x_data_dos, y_data_dos, p0=initial_guess_dos,sigma=y_err_dos)
    a1, b1, a2, b2, x_break_dos = popt_dos
    
    I_0_dos = np.exp(b2)
    h_dos = -1 / a2
    
    print(f'I_0: {I_0_dos}')
    print(f'h: {h_dos}')
    print(f'X break point: {x_break_dos}')

    # Crear los ajustes
    y_fit_dos = two_line_model(x_data_dos, a1, b1, a2, b2, x_break_dos)
    
    chi_squared_dos = calculate_chi_squared(x_data_dos, y_data_dos, y_err_dos, two_line_model, popt_dos)
    print(f"Chi^2 lineas: {chi_squared_dos}")

    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data_dos, y_data_dos, label='Datos', color='black', s=10)
    plt.plot(x_data_dos, y_fit_dos, label=f'Ajuste: Dos líneas con x_break = {x_break_dos:.2f}', color='blue', lw=2)
    
     
    
    # Añadir líneas ajustadas por separado
    break_pos = list(x_data_dos).index(min(x_data_dos[x_data_dos >= x_break_dos]))
    print(f'X break point pos: {break_pos}')
    
    plt.plot(x_data_dos[x_data_dos < x_break_dos], a1 * x_data_dos[x_data_dos < x_break_dos] + b1, 
             label=f'Línea 1: y = {a1:.2f}x + {b1:.2f}', color='red', lw=2)

    plt.plot(x_data_dos[x_data_dos >= x_break_dos], a2 * x_data_dos[x_data_dos >= x_break_dos] + b2, 
             label=f'Línea 2: y = {a2:.2f}x + {b2:.2f}', color='green', lw=2)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Ajuste de dos líneas rectas con punto de corte automático')
    plt.legend()
    plt.grid(True)
    
    if '-png' in sys.argv:
        img_format = 'png'
    else: 
        img_format = 'pdf'
        
    fig_name = f'{galaxy}_lines_fitting_00.{img_format}'
    fig_path = f'{galaxy_folder_path}/{fig_name}'
    plt.savefig(f'{fig_path}', format=img_format, dpi=1000, bbox_inches='tight')   
    
    # SEGUNDA ETAPA: AJUSTE DEL DISCO
    print(f'\nDisco')
    # Nwe fittin
    x_data_disk = df[x_col].values[break_pos:]
    y_data_disk = np.log(df[y_col].values[break_pos:])
    y_err_disk = np.log(df[y_err_col].values[break_pos:])

    initial_guess_disk = [-1, np.mean(x_data_disk)]
    
    # Ajustar el modelo
    popt_disk, pcov_disk = curve_fit(exponential_disk_linear, x_data_disk, y_data_disk, 
                                     p0=initial_guess_disk,sigma=y_err_disk,nan_policy='omit')
    m_disk, b_disk = popt_disk
    
    I_0_disk = np.exp(b_disk)
    h_disk = -1/m_disk

    print(f'I_0: {I_0_disk}')
    print(f'h: {h_disk}')
    
    # Crear los ajustes
    y_fit_disk = exponential_disk_linear(x_data_disk, m_disk, b_disk)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_data_disk,y_fit_disk,color='gold')
    plt.scatter(x_data_disk, y_data_disk, label='Datos', color='black', s=10)
    
    if '-png' in sys.argv:
        img_format = 'png'
    else: 
        img_format = 'pdf'
    
    fig_name = f'{galaxy}_lines_fitting_01_disk.{img_format}'
    fig_path = f'{galaxy_folder_path}/{fig_name}'
    plt.savefig(f'{fig_path}', format=img_format, dpi=1000, bbox_inches='tight')   
    
    chi_squared_disk = calculate_chi_squared(x_data_disk, y_data_disk, y_err_disk, exponential_disk_linear, popt_disk)
    print(f"Chi^2 disk: {chi_squared_disk}")
    
    
    # TERCERA ETAPA: AJUSTE DEL BULBO
    print(f'\nBulbo log')
    # Nwe fittin
    break_pos_bul = int(break_pos * 2/3)
    x_data_bul = df[x_col].values[:break_pos_bul]
    y_data_bul = np.log(df[y_col].values[:break_pos_bul])
    y_err_bul = np.log(df[y_err_col].values[:break_pos_bul])
    
    if '-hr' in sys.argv:
        
        x_data_bul = [val for val in x_data_bul if list(x_data_bul).index(val)%2 == 0]
        y_data_bul = [val for val in y_data_bul if list(y_data_bul).index(val)%2 == 0]
        y_err_bul = [val for val in y_err_bul if list(y_err_bul).index(val)%2 == 0]

    initial_guess_bul = [3,40,3]
    
    # Ajustar el modelo
    popt_bul, pcov_bul = curve_fit(sersic_profile_log, x_data_bul, y_data_bul, p0=initial_guess_bul,sigma=y_err_bul)
    n_bul, r_e_bul, a = popt_bul
    
    I_e_bul = np.exp(a)
    
    print(f'n log: {n_bul}')
    print(f'R_e log: {r_e_bul}')
    print(f'I_e log: {I_e_bul}')

    # Crear los ajustes
    y_fit_bul = sersic_profile_log(x_data_bul, n_bul, r_e_bul, a)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_data_bul,y_fit_bul,color='lime',label='Sersic')
    plt.scatter(x_data_bul, y_data_bul, label='Datos', color='black', s=10)
    
    if '-png' in sys.argv:
        img_format = 'png'
    else: 
        img_format = 'pdf'
    
    fig_name = f'{galaxy}_lines_fitting_02_bulge.{img_format}'
    fig_path = f'{galaxy_folder_path}/{fig_name}'
    plt.savefig(f'{fig_path}', format=img_format, dpi=1000, bbox_inches='tight')   
    
    chi_squared_bul = calculate_chi_squared(x_data_bul, y_data_bul, y_err_bul, sersic_profile_log, popt_bul)
    print(f'Chi^2 bulge: {chi_squared_bul}')

    return break_pos, break_pos_bul, round_number(I_0_disk,3), round_number(h_disk,3), round_number(n_bul,3), round_number(r_e_bul,3), round_number(I_e_bul,3)

def integrate_luminosity(galaxy,func_csv_path,fit_par_csv_path,cons):
    
    fit_func_df = pd.read_csv(func_csv_path)
    results_df = pd.read_csv(fit_par_csv_path)
    
    results = {}
    x_data = fit_func_df['sma']
    
    for column in fit_func_df.columns:
        if column != 'sma':
            y_data = fit_func_df[column]

            integral = simps(y_data, x=x_data)
            results[column] = integral
    
    if len(results) != 1:
        for col, val in results.items():
            if col != 'Total_int':
                
                tot_int = results['Total_int']
                tot_int_mag = values_counts_to_mag(tot_int,cons[0],cons[1])
                val_mag = values_counts_to_mag(val,cons[0],cons[1])
                
                rat = val / tot_int 
                
                print(f'{col}: {val:.2f} / {tot_int:.2f} = {rat:.2f}%\n')
                
                if col == 'Sersic':
                    rat_str = 'B/T'
                elif col == 'Exponential':
                    rat_str = 'D/T'
                
                if len(results_df.index) != 1:
                    
                    results_df[f'Total_lum_counts'].iloc[-1] = round_number(tot_int,3)
                    results_df[f'Total_lum_mag'].iloc[-1] = round_number(tot_int_mag,3)
                    results_df[f'{col}_lum_counts'].iloc[-1] = round_number(val,3)
                    results_df[f'{col}_lum_mag'].iloc[-1] = round_number(val_mag,3)
                    results_df[f'{rat_str}'].iloc[-1] = round_number(rat,3)

                else:
                    
                    results_df[f'Total_lum_counts'] =  round_number(tot_int,3)
                    results_df[f'Total_lum_mag'] = round_number(tot_int_mag,3)
                    results_df[f'{col}_lum_counts'] = round_number(val,3)
                    results_df[f'{col}_lum_mag'] = round_number(val_mag,3)
                    results_df[f'{rat_str}'] = round_number(rat,3)

    results_df.to_csv(fit_par_csv_path,header=True,index=False)

    return fit_par_csv_path



'''#-------------MAIN FUNCTION-------------'''

def main(gal_pos,galaxy):
    
    start_time_gal = time.time()
    
    # Creating a folder for each galaxy
    global galaxy_folder_path
    galaxy_folder_path = create_galaxy_folder(galaxy)
    
    
    # Position of the galaxy in the sky info dataframe
    galaxy_sky_index = galaxy_index(df_sky_info,galaxy)
    
    # Instrumental pixel scale
    inst_arcsec_pix = df_sky_info.loc[galaxy_sky_index]['inst']
    
    # Calibration constant
    zcal = df_sky_info.loc[galaxy_sky_index]['zcal']
    
    # Fits file to analyze
    fits_file = fits_image_list[gal_pos]
    fits_path = f'{files_path}/{fits_file}'
    
    # Opening the galaxy image 
    hdr_gal,img_gal,fits_name = open_fits(fits_path)
    
    fig_name = f'{fits_name}'
    plot_images(img_gal,fig_name,cons=(inst_arcsec_pix,zcal))

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
    gal_iso_fit_csv_path = isophote_fitting(galaxy,fits_path,
                                            (x_center,y_center),
                                            img_mask_path=mask_path,
                                            cons=(inst_arcsec_pix,zcal))
    gal_iso_fit_df = pd.read_csv(gal_iso_fit_csv_path)

                 
    # PSF image
    psf_path = create_psf(fits_path,galaxy,df_sky_info,df_psf_info)

    # Functions to decompose the profile
    funct_fit = ['Sersic',
                 'Exponential']#,
                 #'FerrersBar2D']
    
    # Obtaining the initial conditions
    break_pos, break_pos_bul, I_0_disk_in, h_disk_in, n_bul_in, r_e_bul_in, I_e_bul_in = initial_conditions(gal_iso_fit_df, 'sma', 'intens', 'intens_err')
    
    # SERSIC FUNCTION: Bulge
    # Position angle
    pos_ang_mean_ser = np.mean(gal_iso_fit_df['pa'][:break_pos_bul])
    pos_ang_mean_ser = 120
    pos_ang_ser = [105,0,180]
    
    # Ellipticity
    ellip_mean_ser = np.mean(gal_iso_fit_df['ellipticity'][:break_pos_bul])
    #ellip_mean_ser = 0.5
    ellip_ser = [0.5,0.1,0.9]
    ellip_ser = [ellip_mean_ser,0.05,0.95]
    
    # Sersic Index
    ser_ind_in = n_bul_in
    ser_ind_in = 3
    ser_ind = [3,0.6,6]
    ser_ind = [n_bul_in,0.6,6]
    
    # Effective Raidus
    rad_ef_in = r_e_bul_in
    rad_ef_in = 20
    rad_ef = [20,0,200]
    rad_ef = [r_e_bul_in,0,200]
    
    # Intensity at effective radius
    int_ef_in = I_e_bul_in
    int_ef_in = 120
    int_ef = [120,0,500]
    int_ef = [I_e_bul_in,0,I_e_bul_in+I_e_bul_in//2]
    
    # EXPONENTIAL FUNCTION: Disk
    # Position angle
    pos_ang_mean_disk = np.mean(gal_iso_fit_df['pa'][break_pos:])
    pos_ang_mean_disk = 105
    pos_ang_disk = [105,0,180]
    
    # Ellipticity
    ellip_mean_disk = np.mean(gal_iso_fit_df['ellipticity'][break_pos:])
    #ellip_mean_disk = 0.5
    ellip_disk = [0.5,0.1,0.9]
    ellip_disk = [ellip_mean_disk,0.05,0.95]
                  
    
    # Center intensity
    I_0_disk = I_0_disk_in
    I_0_disk = max_pix_value_center
    int_cent_disk = [I_0_disk,0,I_0_disk+I_0_disk*0.25]
    int_cent_disk = [I_0_disk_in,0,I_0_disk_in+I_0_disk_in//2]
    
    # Length scale disk
    len_sca_in = h_disk_in
    len_sca_in = 80
    len_sca_disk = [80,0,200]
    len_sca_disk = [h_disk_in,0,200]
    
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
                     pa_ser = pos_ang_ser,
                     ell_ser= ellip_ser,
                     n=ser_ind,
                     r_e=rad_ef,
                     I_e=int_ef,
                     pa_disk=pos_ang_disk,
                     ell_disk=ellip_disk,
                     I_0=int_cent_disk,
                     h=len_sca_disk,
                     n_bar = bar_index,
                     a_bar = bar_rad,
                     c0=int_cent_bar)
    conf_file.close()

    
    # Gain value
    gain_value = df_sky_info.loc[galaxy_sky_index]['gain']
    
    # Readnoise value
    noise_value = df_sky_info.loc[galaxy_sky_index]['RON']
    
    # Sky value
    sky_value = df_sky_info.loc[galaxy_sky_index]['sky']
    
    # Output files
    fits_model_name = f'{fits_name}_model_counts.fits'
    fits_model_path = f'{galaxy_folder_path}/{fits_model_name}'
    
    # Output files
    residual_model_name = f'{fits_name}_model_residual_counts.fits'
    residual_model_path = f'{galaxy_folder_path}/{residual_model_name}'
    
    # Changing the directory to run imfit
    os.chdir(f'{galaxy_folder_path}')
    
    # These lines execute imfit as it was a 
    # terminal command 
    if '--imfit-nopath' in sys.argv:
        subprocess.run(['./imfit',
                        fits_path,
                        '-c', conf_file_path,
                        '--mask',mask_path,
                        '--gain=',f'{gain_value}',
                        '--readnoise=',f'{noise_value}',
                        '--sky',f'{sky_value}',
                        '--psf',psf_path,
                        '--save-model=',fits_model_path,
                        '--save-residual=',residual_model_path])
        
    else:
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
    best_parameters_path = f'{galaxy_folder_path}/{best_parameters_name}'
    fit_par_list,fit_min_par,fit_par_csv_path = extract_data_hdr(galaxy,best_parameters_path,
                                                                 cons=(inst_arcsec_pix,zcal))    
    
    sma_min = min(pd.read_csv(gal_iso_fit_csv_path)['sma'])
    sma_max = max(pd.read_csv(gal_iso_fit_csv_path)['sma'])
    sma_len = len(pd.read_csv(gal_iso_fit_csv_path)['sma'])
    
    int_func_csv_path = plot_fit_func(galaxy,fit_par_list,
                                      rad_range=np.linspace(sma_min,sma_max,sma_len))
    
    integrate_luminosity(galaxy,int_func_csv_path,fit_par_csv_path,cons=(inst_arcsec_pix,zcal))

    # Creating a residual image in magnitudes as observed data (mag) - model (mag)
    
    # Original data from counts to mangitudes
    img_gal_mag_path = fits_counts_to_mag(fits_path,inst_arcsec_pix,zcal)
    hdr_gal_mag,img_gal_mag,_ = open_fits(img_gal_mag_path)
    
    # Model from counts to mangitudes
    img_mod_mag_path = fits_counts_to_mag(fits_model_path,inst_arcsec_pix,zcal)
    _,img_mod_mag,_ = open_fits(img_mod_mag_path)
    
    img_res_mag = img_gal_mag - img_mod_mag
    img_res_mag_name = f'{galaxy}_i_model_residual_mag.fits'
    img_res_mag_path = f'{galaxy_folder_path}/{img_res_mag_name}'
    fits.writeto(f'{img_res_mag_path}',img_res_mag,header=hdr_gal_mag,overwrite=True)
    
    img_res_mag_name = f'{galaxy}_i_model_residual_mag'
    plot_images(img_res_mag,img_res_mag_name,cons=(inst_arcsec_pix,zcal),mag=True,res=True)
    
    
    mod_iso_fit_csv_path = isophote_fitting(galaxy,fits_model_path,
                                            (x_center,y_center),
                                            cons=(inst_arcsec_pix,zcal))
    
    
    # Comparing the data profile with the model profile
    plot_list = [(mod_iso_fit_csv_path,'Model'),
                 (gal_iso_fit_csv_path,'Original Data')]
    
    plot_profiles(galaxy,plot_list,'compar',cons=(inst_arcsec_pix,zcal))
    
    plot_list = [(int_func_csv_path,'func_df'),
                 (mod_iso_fit_csv_path,'Model'),
                 (gal_iso_fit_csv_path,'Original Data')]
    
    plot_profiles(galaxy,plot_list,'all',cons=(inst_arcsec_pix,zcal),final_plot=True)
    plt.close()
    
    
    end_time_gal = time.time()
    gal_time = end_time_gal - start_time_gal
    
    
    results_df = pd.read_csv(fit_par_csv_path)
    
    if len(results_df.index) != 1:
        results_df[f'Total_time_sec'].iloc[-1] = round(gal_time,2)
    else:    
        results_df[f'Total_time_sec'] = round(gal_time,2)
        
    results_df.to_csv(fit_par_csv_path,header=True,index=False)
    
    results_global_csv_name = f'csv_results_global.csv'
    results_global_csv_path = f'{cwd}/{results_global_csv_name}'

    if results_global_csv_name not in os.listdir(cwd):
        
        results_global_df = pd.DataFrame(columns=results_df.columns)
        results_global_df.loc[0] = results_df.loc[0]

    else: 
        
        results_global_df = pd.read_csv(results_global_csv_path)
        
        # Removing the previus galaxy data to ovwerwrite data
        if galaxy in results_global_df['galaxy'].values and ('-ow' in sys.argv or '-ow-csv' in sys.argv):
            
            print('\nThis galaxy is already analized. Data will be overwritted')

            #gal_ind_val = galaxy_index(results_global_df,galaxy)
            #results_global_df.loc[gal_ind_val] = results_df.loc[0]
            
            results_global_df = results_global_df[~results_global_df.isin([galaxy]).any(axis=1)]
            
        print('Adding these results to global csv')
        
        results_global_df = pd.concat([results_global_df,results_df],
                                        ignore_index=True)
        
            
    results_global_df = results_global_df.sort_values(by='galaxy')
    results_global_df.to_csv(results_global_csv_path,header=True,index=False)
    


if __name__ == '__main__':
    

    # to compute the total time
    start_time = time.time()
    
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
    
    
    # Computing the required time
    end_time = time.time()
    total_time = end_time - start_time
    

    
    print('\nThe analysis for all the galaxies is finished')
    
    print(f'Total computing time was {(total_time//60):.2f} minutes and {(total_time%60):.2f} seconds\n') 
    
    print('\n#--------------------------------------------------#\n')    
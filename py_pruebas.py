import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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




# Changing from pixeles to kpc 
def px_to_kpc(px,inst):

    arcsec = px * inst 
    kpc = arcsec_to_kpc(arcsec)
    return kpc

def kpc_to_px(kpc,inst):
    
    arcsec = kpc_to_arcsec(kpc)
    px = arcsec / inst 
    return px

def arcsec_to_kpc(arcsec):
    
    kpc = arcsec * 0.12
    
    return kpc

def kpc_to_arcsec(kpc):
    
    arcsec = kpc / (0.12)
    
    return arcsec

# Changing from counts to magnitudes
def values_counts_to_mag(val_counts,inst,zcal):
    val_mag = -2.5*np.log10(val_counts) - 2.5 * np.log10(inst**2) - zcal
    
    return val_mag

def values_mag_to_counts(val_mag,inst,zcal):
    
    val_count = 10**((val_mag - 2.5*np.log10(inst**2)-zcal)/-2.5)
    
    return val_count

# Changing from Ellipticity to Axis ratio
def ell_to_axrat(ell):
    axrat = 1 - ell
    return axrat

def axrat_to_ell(axrat):
    ell = 1 - axrat
    return ell

# Changing from degrees to radian
def deg_to_rad(deg):
    rad = deg * (np.pi / 180)
    return rad

def rad_to_deg(rad):
    deg = rad * (180 / np.pi)
    return deg


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
                                    linestyle='-.',zorder=3)
                            
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
                    ang_deg_abs = y * 180 / np.pi
                    # Angles measured from x righ-hand axis
                    angle_deg = ang_deg_abs % 360
                    y = [(ang+90) if (ang<90) 
                         else (ang-90) if (90<ang<270) 
                         else (ang-270) if (ang>270) 
                         else ang for ang in angle_deg]                
                    y_error = csv_values_dict[plot_label][f'{prof}_err'] * 180 / np.pi
                
                if max(y) > y_max:
                        y_max = max(y)
                if min(y) < y_min:
                        y_min = min(y)
                
                marker_size = 11
                if plot_label == 'Model':
                    marker_size = 14
                plt.scatter(x,y,label=plot_label,
                            marker=markers[color_index],s=marker_size,
                            linewidth=0.15,edgecolor='black',
                            color=colors[color_index],zorder=2)
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
        arcsec_ticks = px_to_kpc(px_ticks)
        axxtop.set_xticks(arcsec_ticks)
        
        axxtop.minorticks_on()
        
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
            axyrig.set_yticks(rad_ticks)
            axyrig.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axyrig.yaxis.set_minor_formatter(plt.NullFormatter())
            
            axyrig.minorticks_on()
            
            axyrig.set_ylabel(r'$Position\ Angle\ [\mathrm{rad}]$')
            axyrig.tick_params(axis='y', which='major')
            

    fig_name_final = f'{galaxy}_{fig_name}_profiles_counts.pdf'
    fig_path = f'{cwd}/{galaxy}/{fig_name_final}'
    plt.savefig(f'{fig_path}', format='pdf', dpi=1000, bbox_inches='tight')






def main():
    
    plot_list = [('/Users/victor/Master_local/Fisica_extragalactica/Master_FisicaExtragalactica_PerfilesBrillo/NGC5443/NGC5443_i_model_counts_isophote.csv',
                  'Model'),
                 ('/Users/victor/Master_local/Fisica_extragalactica/Master_FisicaExtragalactica_PerfilesBrillo/NGC5443/NGC5443_i_isophote.csv',
                  'Original Data')]
    
    plot_profiles('NGC5443',plot_list,'compar',cons=(0.369,-23))
    
    plot_list = [('/Users/victor/Master_local/Fisica_extragalactica/Master_FisicaExtragalactica_PerfilesBrillo/NGC5443/NGC5443_fit_funct.csv',
                  'func_df'),
                 ('/Users/victor/Master_local/Fisica_extragalactica/Master_FisicaExtragalactica_PerfilesBrillo/NGC5443/NGC5443_i_model_counts_isophote.csv',
                  'Model'),
                 ('/Users/victor/Master_local/Fisica_extragalactica/Master_FisicaExtragalactica_PerfilesBrillo/NGC5443/NGC5443_i_isophote.csv',
                  'Original Data')]
    
    plot_profiles('NGC5443',plot_list,'all',cons=(0.369,-23),final_plot=True)
    




if __name__ == '__main__':
    

    
    cwd = os.getcwd()
    
    main() 
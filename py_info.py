# This file constais the sky and psf information

import pandas as pd

def sky_info(path):
    
    sky_info = pd.read_csv(path)
    sky_info.sort_values(by=['galaxy'],ascending=True, inplace=True, ignore_index=True)
    
    return sky_info


def psf_info(path):
    
    psf_info = pd.read_csv(path)
    psf_info.sort_values(by=['galaxy'],ascending=True, inplace=True, ignore_index=True)
    
    return psf_info
    
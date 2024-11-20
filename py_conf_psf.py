
# This file creates the required template for Galfit
def create_conf_psf(file,galaxy,shape,center,fwhm,beta,funct='Moffat',pos_ang=0,ellip=0,I_0=100):
	
        
    file.write(f'# Configuration file for creating a PSF fo the galaxy {galaxy}\n')
    file.write(f'NCOLS {shape[0]} # x-size\n')
    file.write(f'NROWS {shape[1]} # y-size\n')
    file.write(f'X0 {center[0]}\n')
    file.write(f'Y0 {center[1]}\n')
    file.write(f'FUNCTION {funct}\n')
    file.write(f'PA    {pos_ang}\n')
    file.write(f'ell    {ellip} \n')
    file.write(f'I_0    {I_0}   # Counts/pixel \n')
    file.write(f'fwhm    {fwhm} \n')
    file.write(f'beta    {beta} \n')
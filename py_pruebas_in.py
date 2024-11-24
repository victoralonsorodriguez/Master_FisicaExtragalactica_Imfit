import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pdb

import warnings
warnings.filterwarnings('ignore')

def calculate_chi_squared(x, y_obs, y_err, model, params):
    """
    Calcula el valor de chi^2 para un ajuste.
    
    Parámetros:
        x (array): Valores de la variable independiente.
        y_obs (array): Valores observados (dependiente).
        y_err (array): Incertidumbres asociadas a cada valor observado.
        model (callable): Modelo ajustado (función).
        params (list or tuple): Parámetros del modelo ajustado.
    
    Retorna:
        chi_squared (float): Valor de chi^2.
    """
    y_model = model(x, *params)  # Calcula los valores del modelo
    residuals = (y_obs - y_model) / y_err  # Residuos ponderados
    chi_squared = np.sum(residuals**2)  # Suma de cuadrados de los residuos
    return chi_squared

# Definir una función a trozos para modelar dos rectas con un punto de corte
def two_line_model(x, a1, b1, a2, b2, x_break):
    
    return np.where(x < x_break, a1 * x + b1, a2 * x + b2)

def disk_model_lineal(x, m, b):
    y = m * x + b
    return y

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


def sersic_profile(r, n, R_e, I_e):

    b_n = 2 * n - 1 / 3 + 0.009876 / n
    I_r = I_e * np.exp(-b_n * ((r / R_e) ** (1 / n) - 1))

    return I_r




# Ajustar el modelo de dos líneas rectas
def initial_conditions(df, x_col, y_col,y_err_col):
    
    # PRIMERA ETAPA: SEPARACIÓN POR DOS RECTAS
    print(f'\nDos rectas')
    x_data_dos = df[x_col].values
    y_data_dos = np.log(df[y_col].values)
    y_err_dos = np.log(df[y_err_col].values)
    
    # Estimaciones iniciales para los parámetros: a1, b1, a2, b2, x_break
    initial_guess_dos = [1, 0, -1, 0, np.mean(x_data_dos)]
    
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
    
    # SEGUNDA ETAPA: AJUSTE DEL DISCO
    print(f'\nDisco')
    # Nwe fittin
    x_data_disk = df[x_col].values[break_pos:]
    y_data_disk = np.log(df[y_col].values[break_pos:])
    y_err_disk = np.log(df[y_err_col].values[break_pos:])

    initial_guess_disk = [-1, np.mean(x_data_disk)]
    
    # Ajustar el modelo
    popt_disk, pcov_disk = curve_fit(disk_model_lineal, x_data_disk, y_data_disk, p0=initial_guess_disk,sigma=y_err_disk)
    m_disk, b_disk = popt_disk
    
    I_0_disk = np.exp(b_disk)
    h_disk = -1/m_disk

    print(f'I_0: {I_0_disk}')
    print(f'h: {h_disk}')
    
    # Crear los ajustes
    y_fit_disk = disk_model_lineal(x_data_disk, m_disk, b_disk)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_data_disk,y_fit_disk,color='gold')
    plt.scatter(x_data_disk, y_data_disk, label='Datos', color='black', s=10)
    
    chi_squared_disk = calculate_chi_squared(x_data_disk, y_data_disk, y_err_disk, disk_model_lineal, popt_disk)
    print(f"Chi^2 disk: {chi_squared_disk}")
    
    
    # TERCERA ETAPA: AJUSTE DEL BULBO
    print(f'\nBulbo log')
    # Nwe fittin
    x_data_bul = df[x_col].values[:break_pos]
    y_data_bul = np.log(df[y_col].values[:break_pos])
    y_err_bul = np.log(df[y_err_col].values[:break_pos])

    initial_guess_bul = [1,4,20]
    
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
    
    chi_squared_bul = calculate_chi_squared(x_data_bul, y_data_bul, y_err_bul, sersic_profile_log, popt_bul)
    print(f'Chi^2 bulge: {chi_squared_bul}')
    
    # TERCERA ETAPA: AJUSTE DEL BULBO
    print(f'\nBulbo')
    # Nwe fittin
    x_data_bul = df[x_col].values[:break_pos]
    y_data_bul = df[y_col].values[:break_pos]
    y_err_bul = df[y_err_col].values[:break_pos]

    initial_guess_bul = [1,4,20]
    
    # Ajustar el modelo
    popt_bul, pcov_bul = curve_fit(sersic_profile, x_data_bul, y_data_bul, p0=initial_guess_bul,sigma=y_err_bul)
    n_bul, r_e_bul, I_e_bul = popt_bul
    
    print(f'n: {n_bul}')
    print(f'R_e: {r_e_bul}')
    print(f'I_e: {I_e_bul}')

    # Crear los ajustes
    y_fit_bul = sersic_profile(x_data_bul, n_bul, r_e_bul, I_e_bul)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_data_bul,y_fit_bul,color='deeppink',label='Sersic')
    plt.scatter(x_data_bul, y_data_bul, label='Datos', color='black', s=10)
    
    chi_squared_bul = calculate_chi_squared(x_data_bul, y_data_bul, y_err_bul, sersic_profile_log, popt_bul)
    print(f'Chi^2 bulge: {chi_squared_bul}')


    return break_pos

# Ruta del archivo CSV
file_path = './NGC5443/NGC5443_i_isophote.csv'

# Leer datos
df = pd.read_csv(file_path)

# Ajustar y graficar
params = initial_conditions(df, 'sma', 'intens', 'intens_err')
import pandas as pd
from scipy.integrate import simps

def integrate_lum(csv_path):

    fit_func_df = pd.read_csv(csv_path)
    
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
                rat = val / tot_int * 100
                
                print(f'\n{col}: {val:.2f} / {tot_int:.2f} = {rat:.2f}%')
    
    return results

# Ejemplo de uso
if __name__ == "__main__":
    
    # Llama a la función
    integrals = integrate_lum('./NGC5443/NGC5443_fit_funct.csv')

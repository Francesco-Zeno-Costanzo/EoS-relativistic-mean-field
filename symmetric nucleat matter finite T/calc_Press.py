"""
The code calculates the pressure as a function of the number
density for each temperature between Tmin and Tmax e
stores all data in several files that will be read for
find at what temperature the pressure has an inflection
with horizontal tangent
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import Baryon_finite_T as bft

start = time.time()

path  = fr"C:\Users\franc\Documents\magistrale\Tesi\codici\barioni_T_finita\dati_press"


Tmin = 18.5                             #minimum of temperature
Tmax = 19.5                             #maximum of temperature
Tnum = 20                               #number of temperatue
Ture = np.linspace(Tmin, Tmax, Tnum)    #array of temperatue
number_dens = 500                       #number of density
min_dens = 0.01                         #minimum of density
max_dens = 0.2                          #maximun of density


for i, T in enumerate(Ture):

    start1 = time.time()

    #computation of pressure
    _, _, press, _, _, _, _, _, _ = bft.main(min_dens, max_dens, number_dens, T)

    #save on file
    nome = f'\dati{i}.npy'
    np.save(path+nome, press)

    end1 = time.time() - start1

    print(f'{i}-esima temperatura completata in {end1:.3f} secondi')


end = time.time() - start

print(f'Elapsed time = {end:.3f} seconds')
"""
We want to find at which T the pressure has an inflection
with horizontal tangent then we compute the derivative e
we find all the minima of the derivative. After this we find
the zeros of the minima of the derivative as a function of T
because the derivative of a function that has inflection
with horizontal tangent must have minimum equal to zero
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

path  = fr"C:\Users\franc\Documents\magistrale\Tesi\codici\barioni_T_finita\dati_press"

Tmin = 18.5                             #minimum of temperature
Tmax = 19.5                             #maximum of temperature
Tnum = 20                               #number of temperatue
Ture = np.linspace(Tmin, Tmax, Tnum)    #array of temperatue
number_dens = 500                       #number of density
min_dens = 0.01                         #minimum of density
max_dens = 0.2                          #maximun of density

nb = np.linspace(min_dens, max_dens, number_dens) #array of numerical density
P  = np.zeros((Tnum, number_dens))                #matrix for pressure
dP = np.zeros((Tnum, number_dens-1))              #matrix for presuure derivative

for i in range(Tnum):
    nome = f'\dati{i}.npy'
    P[i, :] = np.load(path+nome, allow_pickle=True)

for i in range(Tnum):
    for j in range(number_dens-1):
        dP[i, j] = (P[i, j+1] - P[i, j])/(nb[j+1] - nb[j])

#find all minima and intepolate
min_der = np.array([np.min(dP[i, :]) for i in range(Tnum)])
sp_min = interp1d(Ture, min_der, kind='cubic')
t = np.linspace(Tmin, Tmax, 100000)
#find the zeros
sol = root_scalar(sp_min, method='bisect', bracket=[Tmin, Tmax])
T_c = sol.root
print(f'T_c = {T_c:.3f} MeV')

colors = plt.cm.jet(np.linspace(0, 1, Tnum))
#plot
plt.figure(1)
plt.title('Pressione')
plt.ylabel('P [Mev/$fm^3$]')
plt.xlabel('densità [$fm^{-3}$]')
plt.grid()
plt.xscale('log')
plt.yscale('log')

for i in range(Tnum):
    plt.plot(nb, P[i, :], color=colors[i], label=f'T = {Ture[i]:.3f} MeV')

plt.legend(loc='best')

fig = plt.figure(2)

main_ax = fig.add_subplot()
main_ax.set_title('Pressure derivative')
main_ax.set_xlabel('densità [$fm^{-3}$]', fontsize=10)
main_ax.set_ylabel('dP/dn [Mev/$fm^3$]', fontsize=10)
main_ax.set_ylim(-1, 4)
main_ax.set_xlim(0.02, 0.12)
main_ax.grid()

for i in range(Tnum):
    main_ax.plot(nb[0:-1], dP[i, :], color=colors[i], label=f'T = {Ture[i]:.3f} MeV')

main_ax.legend(loc='best')

right_inset_ax = fig.add_axes([.6, .17, .25, .25])
right_inset_ax.grid()
right_inset_ax.set_title('minimum of pressure derivative')
right_inset_ax.set_ylabel('min(dP/dn) [Mev/$fm^3$]')
right_inset_ax.set_xlabel('temperature [MeV]')

right_inset_ax.plot(Ture, min_der, 'b')
#right_inset_ax.plot(t, sp_min(t),'r')

plt.show()

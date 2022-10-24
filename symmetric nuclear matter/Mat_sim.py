"""
Il codice calcola l'energia per nucleone in materia simmetrica
con il modelo sigma-omega in campo medio relativistico a T = 0
In questo codice si utilizzano i risultati analiti per gli integrali
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

#\hbar/c in MeV fermi
hbc = 197.327

#masse barioni
mass_p = 938
mass_n = 939
m_n = np.array([mass_p, mass_n])/hbc

#masse mesoni
m_sg = 550.1238
m_om = 783.0000
m_rh = 763.0000
m_m = np.array([m_sg, m_om, m_rh])/hbc

#costanti di accopiamento a nb_sat
g_sg_sat = 10.5396
g_om_sat = 13.0189
g_rh_sat = 3.6836 #qui è già in conto il fattore un mezzo
g_sat = np.array([g_sg_sat, g_om_sat, g_rh_sat])

#array per modellare la dipendeza delle costanti di acc
#nell'ordine  sigma,  omega,  rho
a = np.array([1.3881, 1.3892, 0.5647])
b = np.array([1.0943, 0.9240])
c = np.array([1.7057, 1.4620])
d = np.array([0.4421, 0.4775])

def f(x, i):
    """
    funzione che modelliza l'andamento delle
    costati di accopiamento in fuzione di nb
    per il mesone sigma e il mesone omega

    Parameters
    ----------
    x : float or 1darray
        nb/nb_sat
    i : int
        è una flag
        i = 0 sigma
        i = 1 omega

    Returns
    ----------
    g_n : float or 1darray
        g(nb/nb_sat)
    """
    num = 1 + b[i]*(x + d[i])**2
    den = 1 + c[i]*(x + d[i])**2
    g_n = g_sat[i]*a[i] * num/den
    return g_n


def g(x):
    """
    funzione che modelliza l'andamento della
    costate di accopiamento in fuzione di nb
    per il mesone rho

    Parameters
    ----------
    x : float or 1darray
        nb/nb_sat

    Returns
    ----------
    g_n : float or 1darray
        g(nb/nb_sat)
    """
    g_n = g_sat[2] * np.exp(-a[2]*(x-1))
    return g_n


def sistema(V, n, p, kf, g_sg, g_om, g_rh):
    """
    sistema di equazini da risolvere

    Parameters
    ----------
    V : 1darray
        vettore delle incognite
    n : float
        densità neutroni
    p : float
        densità protoni
    kf : float
        impulso di fermi
    g_sg : float
        costante di accopiamento per sigma a densita nb = n+p
    g_om : float
        costante di accopiamento per omega a densita nb = n+p
    g_rh : float
        costante di accopiamento per rho   a densita nb = n+p

    Returns
    ----------
    lista di equazioni da risolvere
    r1 = 0
    r2 = 0
    r3 = 0
    """
    sig, omg, rho = V
    #masse efficaci
    m_eff_p = m_n[0] - sig*g_sg
    m_eff_n = m_n[1] - sig*g_sg
    #energie nucleoni
    ene1 = np.sqrt(kf**2 + m_eff_p**2)
    ene2 = np.sqrt(kf**2 + m_eff_n**2)
    #integrali analitici
    I_p = 1/np.pi**2 * m_eff_p*0.5*(kf*ene1 - m_eff_p**2 * np.log((kf + ene1)/m_eff_p))
    I_n = 1/np.pi**2 * m_eff_n*0.5*(kf*ene2 - m_eff_n**2 * np.log((kf + ene2)/m_eff_n))
    #equazioni da risolvere
    r1 = sig - g_sg/m_m[0]**2 * (I_p + I_n)
    r2 = omg - g_om/m_m[1]**2 * (n + p)
    r3 = rho - g_rh/m_m[2]**2 * (n - p)

    return [r1 , r2, r3]


number_dens = 100
min_dens = 0.01
max_dens = 0.3
nb_sat = 0.152

nb_dens = np.linspace(min_dens, max_dens, number_dens)

#array che conterrano le soluzioni
sigma = np.zeros(number_dens + 1)
omega = np.zeros(number_dens + 1)
rho   = np.zeros(number_dens + 1)

#valori inizali
sigma[0], omega[0], rho[0] = min_dens * g_sat/m_m**2

#array che conterrà l'energia
ene_nb = np.zeros(number_dens)

t0 = time.time()

for i, nb in enumerate(nb_dens): #ciclo sulle densità
    #materia simmetrica
    p = 0.5*nb
    n = 0.5*nb
    #impulso di fermi
    kf = (3*(np.pi**2)*nb/2)**(1/3)
    #costanti al varaire dellla densità
    g_sg = f(nb/nb_sat, 0)
    g_om = f(nb/nb_sat, 1)
    g_rh = g(nb/nb_sat)
    #punto di parteza, dipoende da i per velocizzare la soluzione
    start = (sigma[i], omega[i], rho[i])
    #risolvo il sistema e conservo la soluzione
    sol = root(sistema , start, args=(n, p, kf, g_sg, g_om, g_rh), tol=1e-14, method='hybr')
    sigma[i+1], omega[i+1], rho[i+1] = sol.x

#elimino il valore inizale
sigma = sigma[1:]
omega = omega[1:]
rho   = rho[1:]

#ciclo sulle densità per il calcolo dell'energia
for i, nb in enumerate(nb_dens):

    #impulso di fermi
    kf = (3*(np.pi**2)*nb/2)**(1/3)
    #costanti al varaire dellla densità
    g_sg = f(nb/nb_sat, 0)
    #masse efficaci
    m_eff_p = m_n[0] - sigma[i]*g_sg
    m_eff_n = m_n[1] - sigma[i]*g_sg
    #energie nucleoni
    ene1 = np.sqrt(kf**2 + m_eff_p**2)
    ene2 = np.sqrt(kf**2 + m_eff_n**2)
    #energia mesoni
    ene_m = 0.5*(m_m[0]**2*sigma[i]**2 + m_m[1]**2*omega[i]**2 + m_m[2]**2*rho[i]**2)
    #energie totali nucleoni
    ene_p = (kf*ene1*(ene1**2 - 0.5*m_eff_p**2) - 0.5*m_eff_p**4 * np.log((kf + ene1)/m_eff_p))/(4*np.pi**2)
    ene_n = (kf*ene2*(ene2**2 - 0.5*m_eff_n**2) - 0.5*m_eff_n**4 * np.log((kf + ene2)/m_eff_n))/(4*np.pi**2)
    #energia totale sistema
    ene_nb[i] =  ene_m + ene_p + ene_n


E_over_A =  hbc*(ene_nb/nb_dens - np.mean(m_n))
print(f'minimo di E/A = {np.min(E_over_A):.3f} MeV')

dt = time.time() - t0
print(f'tempo impeigato: {dt}')

#plot energia
plt.figure(1)
plt.title('Energia per nucleone')
plt.ylabel('E/A [Mev]')
plt.xlabel('densita [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, E_over_A)
plt.show()
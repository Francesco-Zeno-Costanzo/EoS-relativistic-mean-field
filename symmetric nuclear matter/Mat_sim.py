"""
In this code we compute the binding energy per nucleon in
symmetric nuclear matter with the use of sigma-omega model
in relativistic mean field at T = 0
This code uses analytical results for the energy integral and others
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

#\hbar/c in MeV fermi
hbc = 197.327

#barion masses
mass_p = 938
mass_n = 939
m_n = np.array([mass_p, mass_n])/hbc

#meson masses
m_sg = 550.1238
m_om = 783.0000
m_rh = 763.0000
m_m = np.array([m_sg, m_om, m_rh])/hbc

#coupling constant at nb_sat
g_sg_sat = 10.5396
g_om_sat = 13.0189
g_rh_sat = 3.6836 #here there is a 1/2 factor
g_sat = np.array([g_sg_sat, g_om_sat, g_rh_sat])

#array to model the dependency of the coupling constants
#in order     sigma,  omega,  rho
a = np.array([1.3881, 1.3892, 0.5647])
b = np.array([1.0943, 0.9240])
c = np.array([1.7057, 1.4620])
d = np.array([0.4421, 0.4775])

def f(x, i):
    """
    function that models the trend of
    coupling constants depending on nb
    for the sigma meson and the omega meson

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
    function that models the trend of
    coupling constants depending on nb
    for the rho meson

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


def system(V, n, p, kf, g_sg, g_om, g_rh):
    """
    system to solve

    Parameters
    ----------
    V : 1darray
        array of the variables
    n : float
        neutron density
    p : float
        proton density
    kf : float
        fermi momentum
    g_sg : float
        coupling constant for sigma at density nb = n+p
    g_om : float
        coupling constant for omega at density nb = n+p
    g_rh : float
        coupling constant for rho   at density nb = n+p

    Returns
    ----------
    list of equation to solve
    r1 = 0
    r2 = 0
    r3 = 0
    """
    sig, omg, rho = V
    #effective masses
    m_eff_p = m_n[0] - sig*g_sg
    m_eff_n = m_n[1] - sig*g_sg
    #nucleon energy
    ene1 = np.sqrt(kf**2 + m_eff_p**2)
    ene2 = np.sqrt(kf**2 + m_eff_n**2)
    #analitycs integrals
    I_p = 1/np.pi**2 * m_eff_p*0.5*(kf*ene1 - m_eff_p**2 * np.log((kf + ene1)/m_eff_p))
    I_n = 1/np.pi**2 * m_eff_n*0.5*(kf*ene2 - m_eff_n**2 * np.log((kf + ene2)/m_eff_n))
    #equation to solve
    r1 = sig - g_sg/m_m[0]**2 * (I_p + I_n)
    r2 = omg - g_om/m_m[1]**2 * (n + p)
    r3 = rho - g_rh/m_m[2]**2 * (n - p)

    return [r1 , r2, r3]

#parameters
number_dens = 100
min_dens = 0.01
max_dens = 0.3
nb_sat = 0.152

nb_dens = np.linspace(min_dens, max_dens, number_dens)

#array that will contain solutions
sigma = np.zeros(number_dens + 1)
omega = np.zeros(number_dens + 1)
rho   = np.zeros(number_dens + 1)

#initial values
sigma[0], omega[0], rho[0] = min_dens * g_sat/m_m**2

#array that will contain energy
ene_nb = np.zeros(number_dens)

t0 = time.time()

for i, nb in enumerate(nb_dens): #loop over density
    #symmetric matter
    p = 0.5*nb
    n = 0.5*nb
    #fermi momentum
    kf = (3*(np.pi**2)*nb/2)**(1/3)
    #coupling constants as the density varies
    g_sg = f(nb/nb_sat, 0)
    g_om = f(nb/nb_sat, 1)
    g_rh = g(nb/nb_sat)
    #starting point, it depends on i to speed up the solution
    start = (sigma[i], omega[i], rho[i])
    #solve the system and keep the solution
    sol = root(system , start, args=(n, p, kf, g_sg, g_om, g_rh), tol=1e-14, method='hybr')
    sigma[i+1], omega[i+1], rho[i+1] = sol.x

#delete the initial value
sigma = sigma[1:]
omega = omega[1:]
rho   = rho[1:]

#loop over density to compute the energy
for i, nb in enumerate(nb_dens):

    #fermi momentum
    kf = (3*(np.pi**2)*nb/2)**(1/3)
    #coupling constants as the density varies
    g_sg = f(nb/nb_sat, 0)
    #effective masses
    m_eff_p = m_n[0] - sigma[i]*g_sg
    m_eff_n = m_n[1] - sigma[i]*g_sg
    #nucleon energy
    ene1 = np.sqrt(kf**2 + m_eff_p**2)
    ene2 = np.sqrt(kf**2 + m_eff_n**2)
    #meson energy
    ene_m = 0.5*(m_m[0]**2*sigma[i]**2 + m_m[1]**2*omega[i]**2 + m_m[2]**2*rho[i]**2)
    #total nucleon energy
    ene_p = (kf*ene1*(ene1**2 - 0.5*m_eff_p**2) - 0.5*m_eff_p**4 * np.log((kf + ene1)/m_eff_p))/(4*np.pi**2)
    ene_n = (kf*ene2*(ene2**2 - 0.5*m_eff_n**2) - 0.5*m_eff_n**4 * np.log((kf + ene2)/m_eff_n))/(4*np.pi**2)
    #total energy of the system
    ene_nb[i] =  ene_m + ene_p + ene_n


E_over_A =  hbc*(ene_nb/nb_dens - np.mean(m_n))
print(f'minimum of E/A = {np.min(E_over_A):.3f} MeV')

dt = time.time() - t0
print(f'elapsed time: {dt}')

#plot energy
plt.figure(1)
plt.title('Binding energy per nucleon')
plt.ylabel('E/A [Mev]')
plt.xlabel('density [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, E_over_A)
plt.show()
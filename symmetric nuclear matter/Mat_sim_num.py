"""
In this code we compute the binding energy per nucleon in
symmetric nuclear matter with the use of sigma-omega model
in relativistic mean field at T = 0
All of the integrals are done totally numerically
"""
import time
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root

#\hbar/c in MeV fermi
hbc = 197.327

#baryon masses
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


def f(n, i, derive=False):
    """
    function that models the behaviour of the
    coupling constants depending on nb
    for the sigma meson and the omega meson
    It returns the derivative depending on the value of derive

    Parameters
    ----------
    n : float or 1darray
        total baryon density
    i : int
        it is a flag
        i = 0 sigma
        i = 1 omega
    derive : boolen
        it is a flag if True returns the derivative

    Return
    ----------
    derive=False
        g_n : float or 1darray
            g(nb/nb_sat)
    derive=True
        dg_n :float or 1d array
            dg/dnb (nb/nb_sat)
    """
    nb_sat = 0.152
    x = n/nb_sat

    if not derive :
        num = 1 + b[i]*(x + d[i])**2
        den = 1 + c[i]*(x + d[i])**2
        g_n = g_sat[i]*a[i] * num/den
        return g_n

    else :
        num = 2*a[i]*(b[i] - c[i])*nb_sat**2 * (x + d[i]*nb_sat)
        den = (nb_sat**2 + c[i]*(x + d[i]*nb_sat)**2)**2
        dg_n = g_sat[i] * num/den
        return dg_n


def g(n, derive=False):
    """
    function that models the behaviour of the
    coupling constant depending on nb
    for the rho meson

    Parameters
    ----------
    x : float or 1darray
        nb/nb_sat
    derive : boolen
        it is a flag if True returns the derivative

    Returns
    ----------
    derive=False
        g_n : float or 1darray
            g(nb/nb_sat)
    derive=True
        dg_n :float or 1d array
            dg/dnb (nb/nb_sat)
    """
    nb_sat = 0.152
    x = n/nb_sat

    if not derive :
        g_n = g_sat[2] * np.exp(-a[2]*(x-1))
        return g_n

    else :
        dg_n = - (g_sat[2]*a[2]/nb_sat) * np.exp(a[2]*(1-x))
        return dg_n


def ns(k, M):
    """
    integrals for sigma equation

    Parameters
    ----------
    k : float
        momentum, variable of integration
    M : float
        effective mass m - g_sg * sigma

    Return
    ----------
    x_dot : float
        integrand function
    """

    num = k**2 * M
    den = np.sqrt(k**2 + M**2)

    x_dot = 1/(np.pi**2) * num/den

    return x_dot

def pre_bar(k, M):
    """
    integral for the nucleons pressure

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        effective mass m - g_sg * sigma

    Return
    ----------
    x_dot : float
        integrand function
    """
    eps = k**4 / np.sqrt(k**2 + M**2)
    x_dot = eps/(np.pi**2)

    return x_dot


def ene(k, M):
    """
    integral for nucleon energy

    Parameters
    ----------
    k : float
        momentum, variable of integration
    M : float
        effective mass m - g_sg * sigma

    Return
    ----------
    x_dot : float
        integrand function
    """
    eps = k**2 * np.sqrt(k**2 + M**2)
    x_dot = eps/np.pi**2

    return x_dot


def Inte(f, kf, M):
    """
    compute integral via scipy

    Parameters
    ----------
    f : function
        integran function
    kf : float
        fermi momentum, integration upper bound
    M : float
        effective mass m - g_sg * sigma

    Return
    -----------
    I : float
        \int_0^kf f(k, M) dk
    """
    I, dI = integrate.quad(f, 0, kf, args=(M))

    return I


def sistema(V, n, p, kf, g_sg, g_om, g_rh):
    """
    equations to solve

    Parameters
    ----------
    V : 1darray
        array of the variables
    n : float
        neutrons density
    p : float
        protons density
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
    list of equations to solve
    r1 = 0
    r2 = 0
    r3 = 0
    """
    sig, omg, rho = V
    #effective masses
    m_eff_p = m_n[0] - sig*g_sg
    m_eff_n = m_n[1] - sig*g_sg
    #numerical integrals
    I_p = Inte(ns, kf, m_eff_p)
    I_n = Inte(ns, kf, m_eff_n)
    #equations to solve
    r1 = sig - g_sg/m_m[0]**2 * (I_p + I_n)
    r2 = omg - g_om/m_m[1]**2 * (n + p)
    r3 = rho - g_rh/m_m[2]**2 * (n - p)
    return[r1 , r2, r3]

#parameters
number_dens = 100
min_dens = 0.01
max_dens = 0.2
nb_sat = 0.152

nb_dens = np.linspace(min_dens, max_dens, number_dens)

#solutions arrays
sigma = np.zeros(number_dens + 1)
omega = np.zeros(number_dens + 1)
rho   = np.zeros(number_dens + 1)

#initial values
sigma[0], omega[0], rho[0] = min_dens * g_sat/m_m**2

#energy array
ene_nb = np.zeros(number_dens)

t0 = time.time()

for i, nb in enumerate(nb_dens): #loop over densities
    #symmetrical matter
    p = 0.5*nb
    n = 0.5*nb
    #fermi momentum
    kf = (3*(np.pi**2)*nb/2)**(1/3)
    #coupling constants as the density varies
    g_sg = f(nb, 0)
    g_om = f(nb, 1)
    g_rh = g(nb)
    #starting point, it depends on i to speed up the solution
    start = (sigma[i], omega[i], rho[i])
    #solve the system and keep the solution
    sol = root(sistema , start, args=(n, p, kf, g_sg, g_om, g_rh), tol=1e-10, method='hybr')
    sigma[i+1], omega[i+1], rho[i+1] = sol.x

#delete the initial value
sigma = sigma[1:]
omega = omega[1:]
rho   = rho[1:]


def Energia_totale(nb_dens, sigma, omega, rho):
    """
    Computation of the system's total energy

    Parameters
    ----------
    nb_dens : 1darray
        total baryonic density
    n_pro, n_neu : 1darrray
        array of the protons and neutrons densities
    sigma, omega, rho : 1darray
        array with the solutions for the mesons

    Retunrs
    ----------
    energ : 1darray
        array for the total energy
    """
    number_dens = len(nb_dens)
    #array for the energy
    energ = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):
        #fermi momenta
        kf_n = (3*(np.pi**2)*0.5*nb)**(1/3)
        kf_p = (3*(np.pi**2)*0.5*nb)**(1/3)
        #constants as the density varies
        g_sg = f(nb, 0)
        #effective masses
        m_eff_p = m_n[0] - sigma[i]*g_sg
        m_eff_n = m_n[1] - sigma[i]*g_sg
        #mesons energy
        ene_m = 0.5*(m_m[0]**2*sigma[i]**2 + m_m[1]**2*omega[i]**2 + m_m[2]**2*rho[i]**2)
        #nucleons energies
        ene_p = Inte(ene, kf_p, m_eff_p)
        ene_n = Inte(ene, kf_n, m_eff_n)
        #system's total energy
        energ[i] =  ene_m + ene_p + ene_n

    return energ


def Pressione_totale(nb_dens, sigma, omega, rho):
    """
    Computation of the system's total pressure

    Parameters
    ----------
    nb_dens : 1darray
        total baryonic density
    n_pro, n_neu : 1darrray
        array for the protons and neutrons densities
    sigma, omega, rho : 1darray
        array for the mesons solutions

    Retunrs
    ----------
    press : 1darray
        array for the total pressure
    """
    number_dens = len(nb_dens)
    #array for the pressure
    press = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):
        #
        p = n = 0.5*nb
        #fermi momenta
        kf_n = (3*(np.pi**2)*n)**(1/3)
        kf_p = (3*(np.pi**2)*p)**(1/3)
        #constants as the density varies
        g_sg = f(nb, 0)
        #effective masses
        m_eff_p = m_n[0] - sigma[i]*g_sg
        m_eff_n = m_n[1] - sigma[i]*g_sg
        #constants derivatives as the density varies
        dg_sg = f(nb, 0, derive=True)
        dg_om = f(nb, 1, derive=True)
        dg_rh = g(nb, derive=True)
        #nucleons pressure terms
        pre_p = Inte(pre_bar, kf_p, m_eff_p)
        pre_n = Inte(pre_bar, kf_n, m_eff_n)
        #mesons derivative terms
        I_p = Inte(ns, kf_p, m_eff_p)
        I_n = Inte(ns, kf_n, m_eff_n)
        pre_m1 = dg_om*omega[i]*(n + p) + dg_rh*rho[i]*(n - p) - dg_sg*sigma[i]*(I_p+I_n)
        #mesons pressure term
        pre_m2 = -0.5*(m_m[0]**2*sigma[i]**2 - m_m[1]**2*omega[i]**2 - m_m[2]**2*rho[i]**2)
        #system's total pressure
        press[i] = (1/3)*(pre_p + pre_n) + (n + p)*pre_m1 + pre_m2

    return press

#energy and pressure computation
ene_nb = Energia_totale(nb_dens, sigma, omega, rho)
pre_nb = Pressione_totale(nb_dens, sigma, omega, rho)

#enrgy per nucleon
E_over_A = hbc*(ene_nb/nb_dens - np.mean(m_n))
#pressure
pre_nb *= hbc

print(f'Minimum of E/A = {np.min(E_over_A):.3f} MeV')

dt = time.time() - t0
print(f'Elapsed time: {dt}')

#plot energia
plt.figure(1)
plt.title('Binding energy per nucleon')
plt.ylabel('E/A [Mev]')
plt.xlabel('density [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, E_over_A)

#pressure plot
plt.figure(2)
plt.title('Pressione')
plt.ylabel('P [Mev/$fm^3$]')
plt.xlabel('densita [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, pre_nb)

plt.show()

path = r"C:\Users\franc\Documents\magistrale\Tesi\codici\Materia simmetrica\dati.txt"
file = open(path, "w")
file.write("# nb \t E/A \t P \n")
for nb, ene, pre in zip(nb_dens, E_over_A, pre_nb):
    file.write(f"{nb} \t {ene} \t{pre} \n")

file.close()
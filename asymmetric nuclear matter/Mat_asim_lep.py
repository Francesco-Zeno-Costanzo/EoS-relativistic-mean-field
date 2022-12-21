"""
This code computes the energy per nucleon in asymmetric matter
with the sigma-omega model in relativistic mean field at T = 0
considering beta equilibrium between protons, electrons, neutrons
and muons
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from quad_gauss import adaptive_gaussian_quadrature

#\hbar/c in MeV fermi
hbc = 197.327

#baryon masses
mass_p = 938.27203
mass_n = 939.56536
m_n = np.array([mass_p, mass_n])/hbc

#meson masses
m_sg = 550.1238
m_om = 783.0000
m_rh = 763.0000
m_m = np.array([m_sg, m_om, m_rh])/hbc

#lepton masses
m_el = 0.51099907
m_mu = 105.6583755
m_l = np.array([m_el, m_mu])/hbc

#coupling constants at nb_sat
g_sg_sat = 10.5396
g_om_sat = 13.0189
g_rh_sat = 3.6836 #we already considered 1/2
g_sat = np.array([g_sg_sat, g_om_sat, g_rh_sat])

#array to model the dependence on the coupling constants
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
    integral for the sigma equation

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

    num = k**2 * M
    den = np.sqrt(k**2 + M**2)

    x_dot = 1/(np.pi**2) * num/den

    return x_dot

def ene(k, M):
    """
    integral for the nucleons energy

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
    eps = k**2 * np.sqrt(k**2 + M**2)
    x_dot = eps/(np.pi**2)

    return x_dot


def pre(k, M):
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


def pre_lep(k, M):
    """
    integral for the leptons pressure

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        mass of particle

    Return
    ----------
    x_dot : float
        integrand function
    """
    eps = k**4 / np.sqrt(k**2 + M**2)
    x_dot = eps/(3*np.pi**2)

    return x_dot


def Inte(f, kf, M):
    """
    this function computes the integral via scipy

    Parameters
    ----------
    f : function
        integrand function
    kf : float
        fermi momentum, integral upper bound
    M : float
        mass

    Return
    -----------
    I : float
        \int_0^kf f(k, M) dk
    """
    I = adaptive_gaussian_quadrature(f, 0, kf, args=(M,))

    return I


def sistema(V, nb, g_sg, g_om, g_rh, muon):
    """
    system to solve

    Parameters
    ----------
    V : 1darray
        array of the variables
    nb : float
        total baryonic density
    g_sg : float
        coupling constant for sigma at density nb = n+p
    g_om : float
        coupling constant for omega at density nb = n+p
    g_rh : float
        coupling constant for rho   at density nb = n+p
    muon : boolean
        it is a flag, if True there are muons

    Returns
    ----------
    list of equations to solve
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    r5 = 0
    r6 = 0
    r7 = 0
    """
    sig, omg, rho, n, p, e, m = V
    #fermi momenta
    kf_p = (3*(np.pi**2)*p)**(1/3)
    kf_n = (3*(np.pi**2)*n)**(1/3)
    kf_e = (3*(np.pi**2)*e)**(1/3)
    #effective masses
    m_eff_p = m_n[0] - sig*g_sg
    m_eff_n = m_n[1] - sig*g_sg
    #numerical integrals for sigma
    I_p = Inte(ns, kf_p, m_eff_p)
    I_n = Inte(ns, kf_n, m_eff_n)
    #chemical potentials for baryons and electrons
    mu_p = np.sqrt(kf_p**2 + m_n[0]**2)
    mu_n = np.sqrt(kf_n**2 + m_n[1]**2)
    mu_e = np.sqrt(kf_e**2 + m_l[0]**2)

    #equations to solve
    if muon:
        #there are muons
        kf_m = (3*(np.pi**2)*m)**(1/3)
        mu_m = np.sqrt(kf_m**2 + m_l[1]**2)
        r6 = mu_e - mu_m
        r7 = p - e - m

    else :
        #there are no muons
        r6 = 0 #automatically satisfied, 0 = 0
        r7 = p - e

    r1 = sig - g_sg/m_m[0]**2 * (I_p + I_n)
    r2 = omg - g_om/m_m[1]**2 * (n + p)
    r3 = rho - g_rh/m_m[2]**2 * (n - p)
    r4 = nb - n - p
    r5 = mu_n - mu_p - mu_e + 2*g_rh*rho

    return [r1, r2, r3, r4, r5, r6, r7]


def Energia_totale(nb_dens, n_pro, n_neu, sigma, omega, rho):
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
        kf_n = (3*(np.pi**2)*n_neu[i])**(1/3)
        kf_p = (3*(np.pi**2)*n_pro[i])**(1/3)
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


def Pressione_totale(nb_dens, n_pro, n_neu, n_ele, n_muo, sigma, omega, rho):
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
        #fermi momenta
        kf_n = (3*(np.pi**2)*n_neu[i])**(1/3)
        kf_p = (3*(np.pi**2)*n_pro[i])**(1/3)
        kf_e = (3*(np.pi**2)*n_ele[i])**(1/3)
        kf_m = (3*(np.pi**2)*n_muo[i])**(1/3)
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
        pre_p = Inte(pre, kf_p, m_eff_p)
        pre_n = Inte(pre, kf_n, m_eff_n)
        #leptons pressure term
        pre_e = Inte(pre_lep, kf_e, m_l[0])
        pre_m = Inte(pre_lep, kf_m, m_l[1])
        #mesons derivative terms
        I_p = Inte(ns, kf_p, m_eff_p)
        I_n = Inte(ns, kf_n, m_eff_n)
        pre_m1 = dg_om*omega[i]*(n_neu[i]+n_pro[i]) + dg_rh*rho[i]*(n_neu[i]-n_pro[i]) - dg_sg*sigma[i]*(I_p+I_n)
        #mesons pressure term
        pre_m2 = -0.5*(m_m[0]**2*sigma[i]**2 - m_m[1]**2*omega[i]**2 - m_m[2]**2*rho[i]**2)
        #system's total pressure
        press[i] = (1/3)*(pre_p + pre_n) + (n_neu[i]+n_pro[i])*pre_m1 + pre_m2 + pre_e + pre_m

    return press


#computation parameters
number_dens = 400
min_dens = 0.01
max_dens = 1#0.3
nb_sat = 0.152

nb_dens = np.linspace(min_dens, max_dens, number_dens)

#array for the solutions
sigma = np.zeros(number_dens + 1)
omega = np.zeros(number_dens + 1)
rho   = np.zeros(number_dens + 1)
n_pro = np.zeros(number_dens + 1)
n_neu = np.zeros(number_dens + 1)
n_ele = np.zeros(number_dens + 1)
n_muo = np.zeros(number_dens + 1)

#initial values
sigma[0], omega[0], rho[0] = min_dens * g_sat/m_m**2
n_pro[0] = 0.01 * min_dens
n_neu[0] = 0.99 * min_dens
n_ele[0] = 0.01 * min_dens
n_muo[0] = 0.00 * min_dens

t0 = time.time()
#boolean variable becomes True if there are muons
muon = False

for i, nb in enumerate(nb_dens): #loop over densities
    #constants as the density varies
    g_sg = f(nb, 0)
    g_om = f(nb, 1)
    g_rh = g(nb)
    #starting point depending on i to speed up the computation
    start = (sigma[i], omega[i], rho[i], n_neu[i], n_pro[i], n_ele[i], n_muo[i])
    #we solve the system and keep the solution
    sol = root(sistema , start, args=(nb, g_sg, g_om, g_rh, muon), tol=1e-10, method='hybr')
    sigma[i+1], omega[i+1], rho[i+1], n_neu[i+1], n_pro[i+1], n_ele[i+1], n_muo[i+1] = sol.x

    #it computes muons' fermi momenta squared
    kf_e = (3*(np.pi**2)*n_ele[i+1])**(1/3)
    kf_m_2 = kf_e**2 + m_l[0]**2 - m_l[1]**2

    #if negative they are below the threshold and there are no muons
    if kf_m_2 <0:
        muon = False

    #Otherwise, here is the "electron decaying"
    else :
        muon = True

#we delete the initial value
sigma = sigma[1:]
omega = omega[1:]
rho   = rho[1:]
n_pro = n_pro[1:]
n_neu = n_neu[1:]
n_ele = n_ele[1:]
n_muo = n_muo[1:]

#energy and pressure computation
ene_nb = Energia_totale(nb_dens, n_pro, n_neu, sigma, omega, rho)
pre_nb = Pressione_totale(nb_dens, n_pro, n_neu, n_ele, n_muo, sigma, omega, rho)

#enrgy per nucleon
E_over_A =  hbc*(ene_nb/nb_dens)
#pressure
pre_nb *= hbc


dt = time.time() - t0
print(f'tempo impeigato: {dt}')

#energy plot
plt.figure(1)
plt.title('Energia per nucleone')
plt.ylabel('E/A [Mev]')
plt.xlabel('densita [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, E_over_A)

#abundances plot
plt.figure(2)
plt.title('Abbondanze in funzione della densità totale')
plt.xlabel('densita [$fm^{-3}$]')
plt.plot(nb_dens, n_pro/nb_dens, 'red', label='$n_p/n_b$')
plt.plot(nb_dens, n_neu/nb_dens, 'black',label='$n_n/n_b$')
plt.plot(nb_dens, n_ele/nb_dens, 'blue',label='$n_e/n_b$')
plt.plot(nb_dens, n_muo/nb_dens, 'purple', label='$n_{\mu}/n_b$')
plt.legend(loc='best')
#plt.xscale('log')
plt.yscale('log')
plt.grid()

#pressure plot
plt.figure(3)
plt.title('Pressione')
plt.ylabel('P [Mev/$fm^3$]')
plt.xlabel('densita [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, pre_nb)

plt.show()
##
path = r"C:\Users\franc\Documents\magistrale\Tesi\codici\Materia asimmetrica\dati.txt"
file = open(path, "w")
file.write("# nb \t Yn \t Yp \t Ye \t Ym \t E/A \t P \n")
for nb, n, p, e, m, ene, pre in zip(nb_dens, n_neu, n_pro, n_ele, n_muo, E_over_A, pre_nb):
    file.write(f"{nb} \t {n/nb} \t {p/nb} \t {e/nb} \t {m/nb} \t {ene} \t{pre} \n")

file.close()
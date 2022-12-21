"""
This code computes the energy per nucleon , pressure and entropy
with the sigma-omega model in relativistic mean field at finte T
for symmetric matter
"""

import time
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root

#\hbar/c in MeV fermi
hbc = 197.327

#saturation numerical densitity
nb_sat = 0.152

#baryon masses
mass_p = 938.27203
mass_n = 939.56536
m_n = np.array([mass_p, mass_n])/hbc

#meson masses
m_sg = 550.1238
m_om = 783.0000
m_rh = 763.0000
m_m = np.array([m_sg, m_om, m_rh])/hbc

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
    ------
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

    Return
    ------
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


def fermi_dirac_p(k, M, Ene, mu, T, mis):
    """
    Fermi–Dirac statistics for particle

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        effective mass m - g_sg * sigma
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature
    mis : boolean
         Fermi–Dirac statistics multiplies all integrals
         so the integration misure is included only if
         we want integrate Fermi–Dirac statistics alone
         mis = True -> \int k^2 dk/pi^2  1/(exp(...) + 1)
         mis = False-> \int k^2 dk/pi^2 f(...) 1/(exp(...) + 1)

    Return
    ------
    f_d : float
        Fermi–Dirac statistics for particle
    """

    if mis == True:

        ene_s = np.sqrt(k**2 + M**2) #single particles energy
        Ene_t = Ene + ene_s          #total energy
        f_d = 1/np.pi**2 * k**2 * np.exp(-(Ene_t - mu)/T)/(1 + np.exp(-(Ene_t - mu)/T))

        return f_d

    if mis == False:

        ene_s = np.sqrt(k**2 + M**2) #single particles energy
        Ene_t = Ene + ene_s          #total energy
        f_d = np.exp(-(Ene_t - mu)/T)/(1 + np.exp(-(Ene_t - mu)/T))

        return f_d


def fermi_dirac_a(k, M, Ene, mu, T, mis):
    """
    Fermi–Dirac statistics for anti-particle

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        effective mass m - g_sg * sigma
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature
    mis : boolean
         Fermi–Dirac statistics multiplies all integrals
         so the integration misure is included only if
         we want integrate Fermi–Dirac statistics alone
         mis = True -> \int k^2 dk/pi^2  1/(exp(...) + 1)
         mis = False-> \int k^2 dk/pi^2 f(...) 1/(exp(...) + 1)

    Return
    ------
    f_d : float
        Fermi–Dirac statistics for anti-particle
    """

    if mis == True:

        ene_s = np.sqrt(k**2 + M**2) #single particles energy
        Ene_t = Ene - ene_s          #total energy
        f_d = 1/np.pi**2 * k**2 * np.exp((Ene_t - mu)/T)/(1 + np.exp((Ene_t - mu)/T))

        return f_d

    if mis == False:

        ene_s = np.sqrt(k**2 + M**2) #single particles energy
        Ene_t = Ene - ene_s          #total energy
        f_d = np.exp((Ene_t - mu)/T)/(1 + np.exp((Ene_t - mu)/T))

        return f_d


def ns(k, M, Ene, mu, T):
    """
    integral for the sigma equation

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        effective mass m - g_sg * sigma
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature

    Return
    ----------
    x_dot : float
        integrand function
    """

    num = M * (fermi_dirac_p(k, M, Ene, mu, T, False) + fermi_dirac_a(k, M, Ene, mu, T, False))
    den = np.sqrt(k**2 + M**2)

    x_dot = 1/np.pi**2 * k**2 * num/den

    return x_dot


def ene(k, M, Ene, mu, T):
    """
    integral for the nucleons energy

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        effective mass m - g_sg * sigma
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature

    Return
    ------
    x_dot : float
        integrand function
    """

    eps = 1/np.pi**2 * k**2 * np.sqrt(k**2 + M**2)
    x_dot = eps * (fermi_dirac_p(k, M, Ene, mu, T, False) + fermi_dirac_a(k, M, Ene, mu, T, False))

    return x_dot


def pre(k, M, Ene, mu, T):
    """
    integral for the nucleons pressure

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        effective mass m - g_sg * sigma
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature

    Return
    ------
    x_dot : float
        integrand function
    """
    eps = 1/np.pi**2 * k**4 * 1 / np.sqrt(k**2 + M**2)
    x_dot = eps * (fermi_dirac_p(k, M, Ene, mu, T, False) + fermi_dirac_a(k, M, Ene, mu, T, False))

    return x_dot


def entropy(k, fd, M, Ene, mu, T):
    """
    integral for the nucleons entropy

    Parameters
    ----------
    k : float
        momentum, integration variable
    fd : callable
        fermi dirac for particle or anti-particle
    M : float
        effective mass m - g_sg * sigma
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature

    Return
    ------
    x_dot : float
        integrand function
    """

    entr_1 = fd(k, M, Ene, mu, T, False) * np.log(fd(k, M, Ene, mu, T, False))
    entr_2 = (1 - fd(k, M, Ene, mu, T, False)) * np.log(1 - fd(k, M, Ene, mu, T, False))
    x_dot = 1/np.pi**2 * k**2 * (entr_1 + entr_2)

    return x_dot


def Inte(f, arg):
    """
    this function computes the integral via scipy

    Parameters
    ----------
    f : function
        integrand function
    arg : tule
        extra arguments to pass to f

    Return
    ------
    I : float
        \int_0^cut_off f(k, arg) dk
    """

    cut_off = 19.0

    I, dI = integrate.quad(f, 0, cut_off, args=arg)

    return I


def sistema(V, n, p, g_sg, g_om, g_rh, dg_sg, dg_om, dg_rh, T):
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
    g_sg : float
        coupling constant for sigma at density nb = n+p
    g_om : float
        coupling constant for omega at density nb = n+p
    g_rh : float
        coupling constant for rho   at density nb = n+p
    dg_sg : float
        derivative of coupling constant for sigma at density nb = n+p
    dg_om : float
        derivative of coupling constant for omega at density nb = n+p
    dg_rh : float
        derivative of coupling constant for rho   at density nb = n+p
    T : float
        temperature

    Returns
    -------
    list of equations to solve
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    r5 = 0
    """
    sig, omg, rho, mu_p, mu_n = V

    #effective masses
    m_eff_p = m_n[0] - sig*g_sg
    m_eff_n = m_n[1] - sig*g_sg
    #termini di energia
    Sigma_p = g_om*omg - g_rh*rho
    Sigma_n = g_om*omg + g_rh*rho
    Sigma_r = (dg_om/g_om)*(m_m[1]*omg)**2 + (dg_rh/g_rh)*(m_m[2]*rho)**2 - (dg_sg/g_sg)*(m_m[0]*sig)**2
    #numerical integrals for sigma
    I_p = Inte(ns, (m_eff_p, Sigma_p + Sigma_r, mu_p, T))
    I_n = Inte(ns, (m_eff_n, Sigma_n + Sigma_r, mu_n, T))
    #fermi dirac integrals
    I_f_p_p = Inte(fermi_dirac_p, (m_eff_p, Sigma_p + Sigma_r, mu_p, T, True))
    I_f_n_p = Inte(fermi_dirac_p, (m_eff_n, Sigma_n + Sigma_r, mu_n, T, True))
    I_f_p_a = Inte(fermi_dirac_a, (m_eff_p, Sigma_p + Sigma_r, mu_p, T, True))
    I_f_n_a = Inte(fermi_dirac_a, (m_eff_n, Sigma_n + Sigma_r, mu_n, T, True))
    #equation to solve
    r1 = sig - g_sg/m_m[0]**2 * (I_p + I_n)
    r2 = omg - g_om/m_m[1]**2 * (n + p)
    r3 = rho - g_rh/m_m[2]**2 * (n - p)
    r4 = p - (I_f_p_p - I_f_p_a)
    r5 = n - (I_f_n_p - I_f_n_a)

    return [r1, r2, r3, r4, r5]


def Energia_totale(nb_dens, sigma, omega, rho, mu_p, mu_n, T):
    """
    Computation of the system's total energy

    Parameters
    ----------
    nb_dens : 1darray
        total baryonic density
    sigma, omega, rho : 1darray
        array with the solutions for the mesons
    mu_p, mu_p : 1darray
        array with the solutions for the chemical potential
    T : float
        temperature

    Retunrs
    -------
    energ : 1darray
        array for the total energy
    """
    number_dens = len(nb_dens)
    #array for the energy
    energ = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):
        #symmetric matter
        p = 0.5*nb
        n = 0.5*nb
        #constants as the density varies
        g_sg = f(nb, 0)
        g_om = f(nb, 1)
        g_rh = g(nb)
        #constants derivatives as the density varies
        dg_sg = f(nb, 0, derive=True)
        dg_om = f(nb, 1, derive=True)
        dg_rh = g(nb, derive=True)
        #effective masses
        m_eff_p = m_n[0] - sigma[i]*g_sg
        m_eff_n = m_n[1] - sigma[i]*g_sg
        #energy for fermi dirac
        Sigma_p = g_om*omega[i] - g_rh*rho[i]
        Sigma_n = g_om*omega[i] + g_rh*rho[i]
        Sigma_r = (dg_om/g_om)*(m_m[1]*omega[i])**2 + (dg_rh/g_rh)*(m_m[2]*rho[i])**2 - (dg_sg/g_sg)*(m_m[0]*sigma[i])**2
        #mesons energy
        ene_m = 0.5*(m_m[0]**2*sigma[i]**2 + m_m[1]**2*omega[i]**2 + m_m[2]**2*rho[i]**2)
        #nucleons energies
        ene_p = Inte(ene, (m_eff_p, Sigma_p + Sigma_r, mu_p[i], T))
        ene_n = Inte(ene, (m_eff_n, Sigma_n + Sigma_r, mu_n[i], T))
        #system's total energy
        energ[i] =  ene_m + ene_p + ene_n

    return energ


def Pressione_totale(nb_dens, sigma, omega, rho, mu_p, mu_n, T):
    """
    Computation of the system's total pressure

    Parameters
    ----------
    nb_dens : 1darray
        total baryonic density
    sigma, omega, rho : 1darray
        array for the mesons solutions
    mu_p, mu_p : 1darray
        array with the solutions for the chemical potential
    T : float
        temperature

    Retunrs
    -------
    press : 1darray
        array for the total pressure
    """
    number_dens = len(nb_dens)
    #array for the pressure
    press = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):
        #symmetric matter
        p = 0.5*nb
        n = 0.5*nb
        #constants as the density varies
        g_sg = f(nb, 0)
        g_om = f(nb, 1)
        g_rh = g(nb)
        #effective masses
        m_eff_p = m_n[0] - sigma[i]*g_sg
        m_eff_n = m_n[1] - sigma[i]*g_sg
        #constants derivatives as the density varies
        dg_sg = f(nb, 0, derive=True)
        dg_om = f(nb, 1, derive=True)
        dg_rh = g(nb, derive=True)
        #energy for fermi dirac
        Sigma_p = g_om*omega[i] - g_rh*rho[i]
        Sigma_n = g_om*omega[i] + g_rh*rho[i]
        Sigma_r = (dg_om/g_om)*(m_m[1]*omega[i])**2 + (dg_rh/g_rh)*(m_m[2]*rho[i])**2 - (dg_sg/g_sg)*(m_m[0]*sigma[i])**2
        #nucleons pressure terms
        pre_p = Inte(pre, (m_eff_p, Sigma_p + Sigma_r, mu_p[i], T))
        pre_n = Inte(pre, (m_eff_n, Sigma_n + Sigma_r, mu_n[i], T))
        #mesons derivative terms
        pre_m1 = Sigma_r
        #mesons pressure term
        pre_m2 = -0.5*(m_m[0]**2*sigma[i]**2 - m_m[1]**2*omega[i]**2 - m_m[2]**2*rho[i]**2)
        #system's total pressure
        press[i] = (1/3)*(pre_p + pre_n) + (n + p)*pre_m1 + pre_m2

    return press


def Entropia_totale(nb_dens, sigma, omega, rho, mu_p, mu_n, T):
    """
    Computation of the system's total entropy

    Parameters
    ----------
    nb_dens : 1darray
        total baryonic density
    sigma, omega, rho : 1darray
        array for the mesons solutions
    mu_p, mu_p : 1darray
        array with the solutions for the chemical potential
    T : float
        temperature

    Retunrs
    -------
    press : 1darray
        array for the total entropy
    """

    number_dens = len(nb_dens)
    #array for the pressure
    entrop = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):
        #symmetric matter
        p = 0.5*nb
        n = 0.5*nb
        #constants as the density varies
        g_sg = f(nb, 0)
        g_om = f(nb, 1)
        g_rh = g(nb)
        #effective masses
        m_eff_p = m_n[0] - sigma[i]*g_sg
        m_eff_n = m_n[1] - sigma[i]*g_sg
        #constants derivatives as the density varies
        dg_sg = f(nb, 0, derive=True)
        dg_om = f(nb, 1, derive=True)
        dg_rh = g(nb, derive=True)
        #energy for fermi dirac
        Sigma_p = g_om*omega[i] - g_rh*rho[i]
        Sigma_n = g_om*omega[i] + g_rh*rho[i]
        Sigma_r = (dg_om/g_om)*(m_m[1]*omega[i])**2 + (dg_rh/g_rh)*(m_m[2]*rho[i])**2 - (dg_sg/g_sg)*(m_m[0]*sigma[i])**2
        #integral for all species
        I_p_p = Inte(entropy, (fermi_dirac_p, m_eff_p, Sigma_p + Sigma_r, mu_p[i], T))
        I_n_p = Inte(entropy, (fermi_dirac_p, m_eff_n, Sigma_n + Sigma_r, mu_n[i], T))
        I_p_a = Inte(entropy, (fermi_dirac_a, m_eff_p, Sigma_p + Sigma_r, mu_p[i], T))
        I_n_a = Inte(entropy, (fermi_dirac_a, m_eff_n, Sigma_n + Sigma_r, mu_n[i], T))
        #system's total entropy
        entrop[i] = -(I_p_p + I_n_p + I_p_a + I_n_a)

    return entrop


def main(min_dens, max_dens, number_dens, T):
    """
    main function, solution of the system at
    various numerical density

    Parameters
    ----------
    min_dens : float
        minimum of numerical density
    max_dens : float
        maximum of numerical density
    number_dens : float
        number of density between min_dens and max_dens
    T : float
        temperature

    Return
    ------
    nb_dens : 1darray
        array with total baryonic density,
        other output are function of nb_dens
    E_over_A : 1darray
        array with energy for nulceons
    pre_nb : 1darray
        array with pressure
    ent_nb : 1darray
        array with entropy
    sigma, omega, rho : 1darray
        array for the mesons solutions
    mu_pr, mu_ne : 1darray
        array for chemical potential
    """
    T = T / hbc
    nb_dens = np.linspace(min_dens, max_dens, number_dens)

    #array for the solutions
    sigma = np.zeros(number_dens + 1)
    omega = np.zeros(number_dens + 1)
    rho   = np.zeros(number_dens + 1)
    mu_pr = np.zeros(number_dens + 1)
    mu_ne = np.zeros(number_dens + 1)

    #initial values
    sigma[0], omega[0], rho[0] = min_dens * g_sat/m_m**2
    mu_pr[0] = 4.5
    mu_ne[0] = 4.5

    for i, nb in enumerate(nb_dens): #loop over densities
        n = 0.5*nb
        p = 0.5*nb
        #constants as the density varies
        g_sg = f(nb, 0)
        g_om = f(nb, 1)
        g_rh = g(nb)
        #constants derivatives as the density varies
        dg_sg = f(nb, 0, derive=True)
        dg_om = f(nb, 1, derive=True)
        dg_rh = g(nb, derive=True)
        #starting point depending on i to speed up the computation
        start = (sigma[i], omega[i], rho[i], mu_pr[i], mu_ne[i])
        #we solve the system and keep the solution
        sol = root(sistema , start, args=(n, p, g_sg, g_om, g_rh, dg_sg, dg_om, dg_rh, T), tol=1e-10, method='hybr')
        sigma[i+1], omega[i+1], rho[i+1], mu_pr[i+1], mu_ne[i+1] = sol.x


    #we delete the initial value
    sigma = sigma[1:]
    omega = omega[1:]
    rho   = rho[1:]
    mu_pr = mu_pr[1:]
    mu_ne = mu_ne[1:]

    #energy, pressure and entropy computation
    ene_nb = Energia_totale(nb_dens, sigma, omega, rho, mu_pr, mu_ne, T)
    pre_nb = Pressione_totale(nb_dens, sigma, omega, rho, mu_pr, mu_ne, T)
    ent_nb = Entropia_totale(nb_dens, sigma, omega, rho, mu_pr, mu_ne, T)

    #enrgy per nucleon
    E_over_A =  hbc*(ene_nb/nb_dens - np.mean(m_n))
    #pressure
    pre_nb *= hbc
    #entropy
    ent_nb /= nb_dens

    return nb_dens, E_over_A, pre_nb, ent_nb, sigma, omega, rho, mu_pr, mu_ne

if __name__ == '__main__':

    t0 = time.time()

    #computation parameters
    number_dens = 100
    min_dens = 0.01
    max_dens = 0.2
    T = 10

    nb_dens, E_over_A, pre_nb, ent_nb, sigma, omega, rho, mu_pr, mu_ne = main(min_dens, max_dens, number_dens, T)

    #energy plot
    plt.figure(1)

    plt.title(f'Binding energy per nucleon at T = {T:.0f} MeV')
    plt.ylabel('E/A [Mev]')
    plt.xlabel('density [$fm^{-3}$]')
    plt.grid()
    plt.plot(nb_dens, E_over_A, 'k')

    #pressure plot
    plt.figure(2)
    plt.title(f'Pressure at T = {T:.0f} MeV')
    plt.ylabel('P [Mev/$fm^3$]')
    plt.xlabel('density [$fm^{-3}$]')
    plt.grid()
    plt.plot(nb_dens, pre_nb, 'k')

    #entropy plot
    plt.figure(3)
    plt.title(f'Entropy at T = {T:.0f} MeV')
    plt.ylabel('S [1/K]')
    plt.xlabel('density [$fm^{-3}$]')
    plt.grid()
    plt.plot(nb_dens, ent_nb, 'k')

    #chemical potential plot
    plt.figure(4)
    plt.title(f'chemical potential for protons and neutrons at T = {T:.0f} MeV')
    plt.ylabel('chemical potential [fm]')
    plt.xlabel('density [$fm^{-3}$]')
    plt.grid()
    plt.plot(nb_dens, mu_ne, 'k', label='neutrons')
    plt.plot(nb_dens, mu_pr, 'b', label='protrons')
    plt.legend(loc='best')

    plt.show()

    dt = time.time() - t0
    print(f'tempo impeigato: {dt}')
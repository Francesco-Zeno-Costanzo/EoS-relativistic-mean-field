"""
Il codice calcola l'energia per nucleone in materia assimmetrica
con il modelo sigma-omega in campo medio relativistico a T = 0
considerando l'equilibrio beta tra protoni elettroni e neutroni
"""
import time
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root

#\hbar/c in MeV fermi
hbc = 197.327

#masse barioni
mass_p = 938.27203
mass_n = 939.56536
m_n = np.array([mass_p, mass_n])/hbc

#masse mesoni
m_sg = 550.1238
m_om = 783.0000
m_rh = 763.0000
m_m = np.array([m_sg, m_om, m_rh])/hbc

#masse leptoni
m_el = 0.51099907 /hbc

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

def f(n, i, derive=False):
    """
    funzione che modelliza l'andamento delle
    costati di accopiamento in fuzione di nb
    per il mesone sigma e il mesone omega
    A seconda del valore derive restituisce
    la derivata

    Parameters
    ----------
    n : float or 1darray
        densita barionica totale
    i : int
        è una flag
        i = 0 sigma
        i = 1 omega
    derive : boolen
        è una flag se True restituisce la derivata

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
    funzione che modelliza l'andamento della
    costate di accopiamento in fuzione di nb
    per il mesone rho

    Parameters
    ----------
    x : float or 1darray
        nb/nb_sat
    derive : boolen
        è una flag se True restituisce la derivata

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
    integrale per l'equazione di sigma

    Parameters
    ----------
    k : float
        impulso, variabile di integrazione
    M : float
        massa efficace m - g_sg * sigma

    Return
    ----------
    x_dot : float
        funzione da integrare
    """

    num = k**2 * M
    den = np.sqrt(k**2 + M**2)

    x_dot = 1/(np.pi**2) * num/den

    return x_dot

def ene(k, M):
    """
    integrale per l'energia dei nucleoni

    Parameters
    ----------
    k : float
        impulso, variabile di integrazione
    M : float
        massa efficace m - g_sg * sigma

    Return
    ----------
    x_dot : float
        funzione da integrare
    """
    eps = k**2 * np.sqrt(k**2 + M**2)
    x_dot = eps/(np.pi**2)

    return x_dot


def pre(k, M):
    """
    integrale per la pressione dei nucleoni

    Parameters
    ----------
    k : float
        impulso, variabile di integrazione
    M : float
        massa efficace m - g_sg * sigma

    Return
    ----------
    x_dot : float
        funzione da integrare
    """
    eps = k**2 / np.sqrt(k**2 + M**2)
    x_dot = eps/(np.pi**2)

    return x_dot


def Inte(f, kf, M):
    """
    funzione tramite scipy calcola l'integrale

    Parameters
    ----------
    f : function
        funzione da integrare
    kf : float
        impulso di fermi, estremo superiore di integrazione
    M : float
        massa efficace m - g_sg * sigma

    Return
    -----------
    I : float
        \int_0^kf f(k, M) dk
    """
    I, dI = integrate.quad(f, 0, kf, args=(M))

    return I

def sistema(V, nb, g_sg, g_om, g_rh):
    """
    sistema di equazini da risolvere

    Parameters
    ----------
    V : 1darray
        vettore delle incognite
    nb : float
        densità barionica totale
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
    r4 = 0
    r5 = 0
    """
    sig, omg, rho, n, p = V
    #impulsi di fermi
    kf_p = (3*(np.pi**2)*p)**(1/3)
    kf_n = (3*(np.pi**2)*n)**(1/3)
    #potenziali chimici
    mu_e = np.sqrt(kf_p**2 + m_el**2)
    mu_p = np.sqrt(kf_p**2 + m_n[0]**2)
    mu_n = np.sqrt(kf_n**2 + m_n[1]**2)
    #masse efficaci
    m_eff_p = m_n[0] - sig*g_sg
    m_eff_n = m_n[1] - sig*g_sg
    #Integrali numerici per sigma
    I_p = Inte(ns, kf_p, m_eff_p)
    I_n = Inte(ns, kf_n, m_eff_n)
    #equazioni da risolvere
    r1 = sig - g_sg/m_m[0]**2 * (I_p + I_n)
    r2 = omg - g_om/m_m[1]**2 * (n + p)
    r3 = rho - g_rh/m_m[2]**2 * (n - p)
    r4 = nb - n - p
    r5 = mu_n - mu_p - mu_e + 2*g_rh*rho
    return[r1 , r2, r3, r4, r5]

def Energia_totale(nb_dens, n_pro, n_neu, sigma, omega, rho):
    """
    Calcolo dell'energia totale del sistema

    Parameters
    ----------
    nb_dens : 1darray
        densità barionica totale
    n_pro, n_neu : 1darrray
        arradi delle densità di protoni e neutroni
    sigma, omega, rho : 1darray
        array che contengono le soluzioni per i mesoni

    Retunrs
    ----------
    energ : 1darray
        array contenete l'enrgia totale
    """
    number_dens = len(nb_dens)
    #array che conterrà l'energia
    energ = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):
        #impulso di fermi
        kf_n = (3*(np.pi**2)*n_neu[i])**(1/3)
        kf_p = (3*(np.pi**2)*n_pro[i])**(1/3)
        #costanti al varaire dellla densità
        g_sg = f(nb, 0)
        #masse efficaci
        m_eff_p = m_n[0] - sigma[i]*g_sg
        m_eff_n = m_n[1] - sigma[i]*g_sg
        #energia mesoni
        ene_m = 0.5*(m_m[0]**2*sigma[i]**2 + m_m[1]**2*omega[i]**2 + m_m[2]**2*rho[i]**2)
        #energie nucleoni
        ene_p = Inte(ene, kf_p, m_eff_p)
        ene_n = Inte(ene, kf_n, m_eff_n)
        #energia totale sistema
        energ[i] =  ene_m + ene_p + ene_n

    return energ

def Pressione_totale(nb_dens, n_pro, n_neu, sigma, omega, rho):
    """
    Calcolo pressione totale del sistema

    Parameters
    ----------
    nb_dens : 1darray
        densità barionica totale
    n_pro, n_neu : 1darrray
        arradi delle densità di protoni e neutroni
    sigma, omega, rho : 1darray
        array che contengono le soluzioni per i mesoni

    Retunrs
    ----------
    press : 1darray
        array contenete la pressione totale
    """
    number_dens = len(nb_dens)
    #array che conterrà la pressione
    press = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):
        #impulso di fermi
        kf_n = (3*(np.pi**2)*n_neu[i])**(1/3)
        kf_p = (3*(np.pi**2)*n_pro[i])**(1/3)
        #costanti al varaire dellla densità
        g_sg = f(nb, 0)
        #masse efficaci
        m_eff_p = m_n[0] - sigma[i]*g_sg
        m_eff_n = m_n[1] - sigma[i]*g_sg
        #derivate delle costanti al variare della densità
        dg_sg = f(nb, 0, derive=True)
        dg_om = f(nb, 1, derive=True)
        dg_rh = g(nb, derive=True)
        #tempini di pressione dei nucleoni
        pre_p = Inte(pre, kf_p, m_eff_p)
        pre_n = Inte(pre, kf_n, m_eff_n)
        #terimini di derivara dei mesoni
        I_p = Inte(ns, kf_p, m_eff_p)
        I_n = Inte(ns, kf_n, m_eff_n)
        pre_m1 = dg_om*omega[i]*(n_neu[i]+n_pro[i]) + dg_rh*rho[i]*(n_neu[i]-n_pro[i]) - dg_sg*sigma[i]*(I_p+I_n)
        #termine pressione mesoni
        pre_m2 = -0.5*(m_m[0]**2*sigma[i]**2 - m_m[1]**2*omega[i]**2 - m_m[2]**2*rho[i]**2)
        #pressione totale del sistema
        press[i] = (1/3)*(pre_p + pre_n) + (n_neu[i]+n_pro[i])*pre_m1 + pre_m2

    return press

#parametri calcolo
number_dens = 300
min_dens = 0.01
max_dens = 1#0.3
nb_sat = 0.152

nb_dens = np.linspace(min_dens, max_dens, number_dens)

#array che conterrano le soluzioni
sigma = np.zeros(number_dens + 1)
omega = np.zeros(number_dens + 1)
rho   = np.zeros(number_dens + 1)
n_pro = np.zeros(number_dens + 1)
n_neu = np.zeros(number_dens + 1)

#valori inizali
sigma[0], omega[0], rho[0] = min_dens * g_sat/m_m**2
n_pro[0] = 0.001 * min_dens
n_neu[0] = 0.999 * min_dens


t0 = time.time()

for i, nb in enumerate(nb_dens): #ciclo sulle densità
    #costanti al varaire dellla densità
    g_sg = f(nb, 0)
    g_om = f(nb, 1)
    g_rh = g(nb)
    #punto di parteza, dipoende da i per velocizzare la soluzione
    start = (sigma[i], omega[i], rho[i], n_neu[i], n_pro[i])
    #risolvo il sistema e conservo la soluzione
    sol = root(sistema , start, args=(nb, g_sg, g_om, g_rh), tol=1e-10, method='hybr')
    sigma[i+1], omega[i+1], rho[i+1], n_neu[i+1], n_pro[i+1] = sol.x

#elimino il valore inizale
sigma = sigma[1:]
omega = omega[1:]
rho   = rho[1:]
n_pro = n_pro[1:]
n_neu = n_neu[1:]

#calcolo dell'energia e della pressione
ene_nb = Energia_totale(nb_dens, n_pro, n_neu, sigma, omega, rho)
pre_nb = Pressione_totale(nb_dens, n_pro, n_neu, sigma, omega, rho)

#enrgia per nucleone
E_over_A =  hbc*(ene_nb/nb_dens)
#pressione
pre_nb *= hbc


dt = time.time() - t0
print(f'tempo impeigato: {dt}')

#plot energia
plt.figure(1)
plt.title('Energia per nucleone')
plt.ylabel('E/A [Mev]')
plt.xlabel('densita [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, E_over_A)

#plot abbondanze
plt.figure(2)
plt.title('Abbondanze in funzione della densità tottale')
plt.plot(nb_dens, n_pro/nb_dens, label='np/nb')
plt.plot(nb_dens, n_neu/nb_dens, label='nn/nb')
plt.legend(loc='best')
#plt.xscale('log')
plt.yscale('log')
plt.grid()

#plot pressione
plt.figure(3)
plt.title('Pressione')
plt.ylabel('P [Mev/$fm^3$]')
plt.xlabel('densita [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, pre_nb)

plt.show()
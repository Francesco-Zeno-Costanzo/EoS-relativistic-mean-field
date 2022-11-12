"""
code for integration with Gauss quadrature
"""

import math
from scipy.special import roots_legendre

def adaptive_gaussian_quadrature(f, a, b, args=(), tol=1e-8, n=5):
    """
    Splits an integral into the right and left half
    and compares it to the integral on the whole interval
    if tolerance is not satisfied, left and right are
    splitted into smaller intervals and so on until the
    tolerance is satisfied they are added to the sum

    Parameters
    ----------
    f : callable
        function to integrate
    a : float
        beginning of the interval
    b : float
        end of the interval
    args : tuple, optional
        extra arguments to pass to f
    tol : float
        tollerance, default 1e-8
    n : int
        number of nodes, default n=5

    Return
    ----------
    x, Inte : float
        \int_a^b f
    """

    #interval divison
    m = a + (b - a)/2
    #compute integral
    Int = gauss_leg(f, a, b, args, n)
    I_r = gauss_leg(f, m, b, args, n)
    I_l = gauss_leg(f, a, m, args, n)
    #check tollerance
    if abs( Int - (I_l + I_r) ) < tol:
        return Int
    #recursive call
    div1 = adaptive_gaussian_quadrature(f, m, b, args, tol, n)
    div2 = adaptive_gaussian_quadrature(f, a, m, args, tol, n)
    #sum of all integrals in the several intervals
    x = div1 + div2

    return x


def gauss_leg(f, a, b, args=(), n=5):
    """
    Calculation of the integral of f
    with the Gaussian quadrature method

    Parameters
    ----------
    f : callable
        function to integrate
    a : float
        beginning of the interval
    b : float
        end of the interval
    args : tuple, optional
        extra arguments to pass to f
    n : int
        number of nodes, default n=5

    Return
    ----------
    Inte : float
        \int_a^b f
    """

    #b must be grather tha a
    if b < a:
        a, b = b, a
    else : pass

    Inte = 0
    dxdxi = (b - a)/2
    roots, weights = roots_legendre(n)

    for x_i, w_i in zip(roots, weights):
        Inte += w_i * f(x_i * dxdxi + (b + a)/2, *args)

    Inte *= dxdxi

    return Inte


def test():
    """ little test
    """
    def h(x):
        """sine
        """
        return math.sin(x)
    def f(x, a1):
        """gaussian
        """
        return math.exp(-x**2/(2*a1))/(math.sqrt(2*math.pi)*a1)
    def g(x, a1, a2):
        """fermi dirac
        """
        return 1/(math.exp((x - a1)/a2) + 1)

    I = adaptive_gaussian_quadrature(h, 0, math.pi)
    print('Integral value is', I)
    I = adaptive_gaussian_quadrature(f, -1, 1, args=(1,))
    print('Integral value is', I)
    I = adaptive_gaussian_quadrature(g, 0, 20, args=(10, 0.02))
    print('Integral value is', I)


if __name__ == '__main__':

    test()
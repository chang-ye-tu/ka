# -*- coding: utf-8 -*-

from op import *

c1, m1, c2, m2 = 1., 1., 2., 2.
l1, l2, l3, l4 = 1. + 1.j, 3., 1., 2.
omega = 1.
eps, mu, beta = 1., 1., 0.
eps0, mu0, beta0 = 1.4, 1.2, 0.1
k = sqrt(omega**2 * eps * mu)
k0 = sqrt(omega**2 * eps0 * mu0)
gamma = sqrt(1 / (1 - k0**2 * beta0**2))
a_a = k0 * gamma**2 * (1 + k0 * beta0) 
a_b = k0 * gamma**2 * (1 - k0 * beta0)
xi = sqrt((eps0 * mu)/(eps * mu0))
xi_p = (xi + 1/xi) / 2
xi_m = (xi - 1/xi) / 2

n = 16     # sections 
d = [1, 0] # observed direction    
u = zeros((8*n,), dtype=complex)
for l in arange(2*n):
    norm = x(l*pi/n)
    d_nu = (x2p(l*pi/n) * x1(l*pi/n) - x1p(l*pi/n) * x2(l*pi/n)) / (xp(l*pi/n) * norm)
    u[4*l] = -2 * a_a * (l1 * j1(a_a * norm) - l3 * hankel1(1, k * norm)) * d_nu 
    u[4*l+1] = -2 * a_b * (l2 * j1(a_b * norm) - l4 * hankel1(1, k * norm)) * d_nu 
    u[4*l+2] = 2 * l1 * j0(a_a * norm) - 2 * (l3 * xi_p + l4 * xi_m) * hankel1(0, k * norm)
    u[4*l+3] = 2 * l2 * j0(a_b * norm) - 2 * (l3 * xi_m + l4 * xi_p) * hankel1(0, k * norm)

def R(t):
    s = 0.
    for l in arange(1, n):
        s += 1. / l * cos(l * t * pi/n)
    return -2 * pi/n * s - pi/(n**2) * ((-1)**t)

K = zeros((8*n, 8*n), dtype=complex)
for l in arange(2*n):
    for m in arange(2*n):
        K[4*l, 4*m] = m1 * (R(l - m) * Ls1(pi*l/n, pi*m/n, a_a) + pi/n * Ls2(pi*l/n, pi*m/n, a_a)) - a_a/k * c1 * (R(l - m) * Ls1(pi*l/n, pi*m/n, k) + pi/n * Ls2(pi*l/n, pi*m/n, k))
        K[4*l, 4*m+1] = 0
        K[4*l, 4*m+2] = a_a/k * ((R(l - m) * N1(pi*l/n, pi*m/n, a_a) + pi/n * N2(pi*l/n, pi*m/n, a_a)) - (R(l - m) * N1(pi*l/n, pi*m/n, k) + pi/n * N2(pi*l/n, pi*m/n, k)))
        K[4*l, 4*m+3] = 0
        K[4*l+1, 4*m] = 0 
        K[4*l+1, 4*m+1] = m2 * (R(l - m) * Ls1(pi*l/n, pi*m/n, a_b) + pi/n * Ls2(pi*l/n, pi*m/n, a_b)) - a_b/k * c2 * (R(l - m) * Ls1(pi*l/n, pi*m/n, k) + pi/n * Ls2(pi*l/n, pi*m/n, k))
        K[4*l+1, 4*m+2] = 0 
        K[4*l+1, 4*m+3] = a_b/k * ((R(l - m) * N1(pi*l/n, pi*m/n, a_b) + pi/n * N2(pi*l/n, pi*m/n, a_b)) - (R(l - m) * N1(pi*l/n, pi*m/n, k) + pi/n * N2(pi*l/n, pi*m/n, k)))
        K[4*l+2, 4*m] = R(l - m) * (m1 * M1(pi*l/n, pi*m/n, a_a) - c1 * xi_p* M1(pi*l/n, pi*m/n, k)) + pi/n * (m1 * M2(pi*l/n, pi*m/n, a_a) - c1 * xi_p * M2(pi*l/n, pi*m/n, k))
        K[4*l+2, 4*m+1] = -c2 * xi_m * (R(l - m) * M1(pi*l/n, pi*m/n, k) + pi/n * M2(pi*l/n, pi*m/n, k))
        K[4*l+2, 4*m+2] = R(l - m) * (-xi_p * L1(pi*l/n, pi*m/n, k) + a_a/k * L1(pi*l/n, pi*m/n, a_a)) + pi/n * (-xi_p * L2(pi*l/n, pi*m/n, k) + a_a/k * L2(pi*l/n, pi*m/n, a_a))
        K[4*l+2, 4*m+3] = -xi_m * (R(l - m) * L1(pi*l/n, pi*m/n, k) + pi/n * L2(pi*l/n, pi*m/n, k))
        K[4*l+3, 4*m] = -c1 * xi_m * (R(l - m) * M1(pi*l/n, pi*m/n, k) + pi/n * M2(pi*l/n, pi*m/n, k))
        K[4*l+3, 4*m+1] = R(l - m) * (m2 * M1(pi*l/n, pi*m/n, a_b) - c2 * xi_p * M1(pi*l/n, pi*m/n, k)) + pi/n * (m2 * M2(pi*l/n, pi*m/n, a_b) - c2 * xi_p * M2(pi*l/n, pi*m/n, k))
        K[4*l+3, 4*m+2] = -xi_m * (R(l - m) * L1(pi*l/n, pi*m/n, k) + pi/n * L2(pi*l/n, pi*m/n, k))
        K[4*l+3, 4*m+3] = R(l - m) * (-xi_p * L1(pi*l/n, pi*m/n, k) + a_b/k * L1(pi*l/n, pi*m/n, a_b)) + pi/n * (-xi_p * L2(pi*l/n, pi*m/n, k) + a_b/k * L2(pi*l/n, pi*m/n, a_b))

E = zeros((8*n, 8*n), dtype=complex)
for l in arange(2*n):
    E[4*l, 4*l] = m1 + a_a/k * c1 
    E[4*l+1, 4*l+1] = m2 + a_b/k * c2
    E[4*l+2, 4*l+2] = -xi_p - a_a/k
    E[4*l+2, 4*l+3] = -xi_m
    E[4*l+3, 4*l+2] = -xi_m 
    E[4*l+3, 4*l+3] = -xi_p - a_b/k

z = linalg.solve(E + K, u)
 
const = exp(-1j*pi/4) / sqrt(8*pi*k) * pi/n
a_inf, b_inf = 0., 0.
for l in arange(2*n):
    a_inf += (k * (d[0] * x2p(l*pi/n) - d[1] * x1p(l*pi/n)) * z[4*l+2] + 1j * c1 * xp(l*pi/n) * z[4*l]) * exp(-1j * k * (d[0] * x1(l*pi/n) + d[1] * x2(l*pi/n)))
 
    b_inf += (k * (d[0] * x2p(l*pi/n) - d[1] * x1p(l*pi/n)) * z[4*l+3] + 1j * c2 * xp(l*pi/n) * z[4*l+1]) * exp(-1j * k * (d[0] * x1(l*pi/n) + d[1] * x2(l*pi/n)))
a_inf = const * a_inf
b_inf = const * b_inf

a_inf_0 = l3 * sqrt(2/(pi*k)) * exp(-1j*pi/4) 
b_inf_0 = l4 * sqrt(2/(pi*k)) * exp(-1j*pi/4)

print a_inf, a_inf_0
print b_inf, b_inf_0

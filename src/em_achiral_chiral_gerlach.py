# -*- coding: utf-8 -*-

from op import *

c1, m1, c2, m2 = 1., 1., 2., 2.
l1, l2, l3, l4 = 1. + 1.j, 3., 1., 2.
omega = 1.
eps, mu, beta = 1., 1., 0.05
eps0, mu0, beta0 = 1.4, 1.2, 0.1
k = sqrt(omega**2 * eps * mu)
k0 = sqrt(omega**2 * eps0 * mu0)
gamma = sqrt(1 / (1 - k0**2 * beta0**2))
a_a = k0 * gamma**2 * (1 + k0 * beta0) 
a_b = k0 * gamma**2 * (1 - k0 * beta0)
xi = sqrt((eps0 * mu)/(eps * mu0))
xi_p = (xi + 1/xi) / 2
xi_m = (xi - 1/xi) / 2

b_direct = False 

n = 32    # sections 
n_o = 32  # observation points
count = 0
tic = dt_now() 

if b_direct:
    mode = 'TE'
    print 'Generating A now (n_o = %s & n = %s): started %s\n' % (n_o, n, tic)
    A = zeros((n_o, n_o, 2), dtype=complex)

    for ii in arange(n_o):
        d = [cos(2*pi*ii/n_o), sin(2*pi*ii/n_o)]      # observed direction    
        for jj in arange(n_o):
            d0 = [cos(2*pi*jj/n_o), sin(2*pi*jj/n_o)] # incident wave direction
            u = zeros((8*n,), dtype=complex)
            if mode == 'TE':
                for l in arange(2*n):
                    u[4*l] = -2 * 1j * k * a_a * sqrt(eps0 / eps) * exp(1j * k * (x1(l*pi/n) * d0[0] + x2(l*pi/n) * d0[1])) * (x2p(l*pi/n) * d0[0] - x1p(l*pi/n) * d0[1]) / xp(l*pi/n)
                    u[4*l+1] = 2 * 1j * k * a_b * sqrt(eps0 / eps) * exp(1j * k * (x1(l*pi/n) * d0[0] + x2(l*pi/n) * d0[1])) * (x2p(l*pi/n) * d0[0] - x1p(l*pi/n) * d0[1]) / xp(l*pi/n)
                    u[4*l+2] = -2 * k * sqrt(mu0 / mu) * exp(1j * k * (x1(l*pi/n) * d0[0] + x2(l*pi/n) * d0[1]))
                    u[4*l+3] = 2 * k * sqrt(mu0 / mu) * exp(1j * k * (x1(l*pi/n) * d0[0] + x2(l*pi/n) * d0[1]))
            
            elif mode == 'TM':
                for l in arange(2*n):
                    u[4*l] = -2 * k * a_a * sqrt(mu0 / mu) * exp(1j * k * (x1(l*pi/n) * d0[0] + x2(l*pi/n) * d0[1])) * (x2p(l*pi/n) * d0[0] - x1p(l*pi/n) * d0[1]) / xp(l*pi/n)
                    u[4*l+1] = -2 * k * a_b * sqrt(mu0 / mu) * exp(1j * k * (x1(l*pi/n) * d0[0] + x2(l*pi/n) * d0[1])) * (x2p(l*pi/n) * d0[0] - x1p(l*pi/n) * d0[1]) / xp(l*pi/n)
                    u[4*l+2] = 2 * 1j * k * sqrt(eps0 / eps) * exp(1j * k * (x1(l*pi/n) * d0[0] + x2(l*pi/n) * d0[1]))
                    u[4*l+3] = 2 * 1j * k * sqrt(eps0 / eps) * exp(1j * k * (x1(l*pi/n) * d0[0] + x2(l*pi/n) * d0[1]))

            def R(t):
                s = 0.
                for l in arange(1, n):
                    s += 1. / l * cos(l * t * pi/n)
                return -2 * pi/n * s - pi/(n**2) * ((-1)**t)
            
            K = zeros((8*n, 8*n), dtype=complex)
            for l in arange(2*n):
                for m in arange(2*n):
                    K[4*l, 4*m] = R(l - m) * (m1 * Ls1(pi*l/n, pi*m/n, a_a) - a_a/k * c1 * Ls1(pi*l/n, pi*m/n, k)) + pi/n * (m1 * Ls2(pi*l/n, pi*m/n, a_a)- a_a/k * c1 * Ls2(pi*l/n, pi*m/n, k))
                    K[4*l, 4*m+1] = 0
                    K[4*l, 4*m+2] = R(l - m) * (a_a/k * (N1(pi*l/n, pi*m/n, a_a) - N1(pi*l/n, pi*m/n, k))) + pi/n * (a_a/k * (N2(pi*l/n, pi*m/n, a_a) - N2(pi*l/n, pi*m/n, k)))
                    K[4*l, 4*m+3] = 0
                    K[4*l+1, 4*m] = 0 
                    K[4*l+1, 4*m+1] = R(l - m) * (m2 * Ls1(pi*l/n, pi*m/n, a_b) - a_b/k * c2 * Ls1(pi*l/n, pi*m/n, k)) + pi/n * (m2 * Ls2(pi*l/n, pi*m/n, a_b) - a_b/k * c2 * Ls2(pi*l/n, pi*m/n, k))
                    K[4*l+1, 4*m+2] = 0 
                    K[4*l+1, 4*m+3] = R(l - m) * (a_b/k * (N1(pi*l/n, pi*m/n, a_b) - N1(pi*l/n, pi*m/n, k))) + pi/n * (a_b/k * (N2(pi*l/n, pi*m/n, a_b) - N2(pi*l/n, pi*m/n, k)))
                    K[4*l+2, 4*m] = R(l - m) * (m1 * M1(pi*l/n, pi*m/n, a_a) - c1 * xi_p* M1(pi*l/n, pi*m/n, k)) + pi/n * (m1 * M2(pi*l/n, pi*m/n, a_a) - c1 * xi_p * M2(pi*l/n, pi*m/n, k))
                    K[4*l+2, 4*m+1] = R(l - m) * (-c2 * xi_m * M1(pi*l/n, pi*m/n, k)) + pi/n * (-c2 * xi_m * M2(pi*l/n, pi*m/n, k))
                    K[4*l+2, 4*m+2] = R(l - m) * (-xi_p * L1(pi*l/n, pi*m/n, k) + a_a/k * L1(pi*l/n, pi*m/n, a_a)) + pi/n * (-xi_p * L2(pi*l/n, pi*m/n, k) + a_a/k * L2(pi*l/n, pi*m/n, a_a))
                    K[4*l+2, 4*m+3] = R(l - m) * (-xi_m * L1(pi*l/n, pi*m/n, k)) + pi/n * (-xi_m * L2(pi*l/n, pi*m/n, k))
                    K[4*l+3, 4*m] = R(l - m) * (-c1 * xi_m * M1(pi*l/n, pi*m/n, k)) + pi/n * (-c1 * xi_m * M2(pi*l/n, pi*m/n, k))
                    K[4*l+3, 4*m+1] = R(l - m) * (m2 * M1(pi*l/n, pi*m/n, a_b) - c2 * xi_p * M1(pi*l/n, pi*m/n, k)) + pi/n * (m2 * M2(pi*l/n, pi*m/n, a_b) - c2 * xi_p * M2(pi*l/n, pi*m/n, k))
                    K[4*l+3, 4*m+2] = R(l - m) * (-xi_m * L1(pi*l/n, pi*m/n, k)) + pi/n * (-xi_m * L2(pi*l/n, pi*m/n, k))
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
            
            A[ii, jj, 0] = a_inf
            A[ii, jj, 1] = b_inf

            count += 1

            print u'processing %-5d of %5d: %3d %% ... time elapsed: %s' % (count, n_o**2, int(count * 100. / n_o**2),  lapse(dt_now() - tic))
        
    print u'\ncompleted! total time: %s' % lapse(dt_now() - tic)

    savez(os.path.join(os.getcwd(), 'A_achiral_chiral_%s' % mode), A=A)

else:
    mode = 'TM'
    npz = load(os.path.join(os.getcwd(), 'A_achiral_chiral_%s.npz' % mode))
    AA = npz['A']

    A1 = zeros((n_o, n_o), dtype=complex)
    A2 = zeros((n_o, n_o), dtype=complex)
    for ii in arange(n_o):
        for jj in arange(n_o):
            A1[ii, jj] = AA[ii, jj, 0]
            A2[ii, jj] = AA[ii, jj, 1]
    A = sqrt(mu0 / mu) ** (-1/2) * (A1 + A2) / 2 
    #A = sqrt(eps0 / eps) ** (-1/2) * (A1 - A2) / (2 * 1.j)

    U, La, V = linalg.svd(A.transpose())
    V = V.conj().transpose()

    # setup grids
    x_min, x_max = -5, 10
    y_min, y_max = -5, 10
    n_x, n_y = 100, 100 

    x = linspace(x_min, x_max, n_x)
    y = linspace(y_min, y_max, n_y)
    w = zeros((n_x, n_y), dtype=float64)

    count = 0
    tic = dt_now()
    print 'classify now: started %s\n' % tic

    for ii, zx in enumerate(x):
        for jj, zy in enumerate(y):
            
            wn = zeros((n_o,), dtype=complex)
            rz = zeros((n_o,), dtype=complex)
    
            for ll in arange(n_o):
                d = [cos(2*pi*ll/n_o), sin(2*pi*ll/n_o)]  # observed direction 
                rz[ll] = exp(-1j * k * (zx * d[0] + zy * d[1]))
            wn = dot(V.transpose(), rz)
            
            wt = 0
            for ll in arange(n_o): 
                wt += abs(wn[ll]) ** 2 / abs(La[ll])
            w[jj, ii] = 1. / wt
            
            count += 1
            print u'processing %-5d of %5d: %3d %% ... time elapsed: %s' % (count, n_x * n_y, int(count * 100. / (n_x * n_y)),  lapse(dt_now() - tic))
    
    savez(os.path.join(os.getcwd(), 'xyw_achiral_chiral_%s' % mode), x=x, y=y, w=w)
    
    print u'\ncompleted! total time: %s' % lapse(dt_now() - tic)

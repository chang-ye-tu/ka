# -*- coding: utf-8 -*-

from op import *

c_il, c_ir, c_el, c_er = 1. + 1.j, 3., 1., 2.
omega = 1.

eps_i, mu_i, beta_i = 1., 1., 0.
eps_e, mu_e, beta_e = 1.4, 1.2, 0.1

k_i = sqrt(omega**2 * eps_i * mu_i)
k_e = sqrt(omega**2 * eps_e * mu_e)
gamma_ir = k_i / (1 + k_i * beta_i) 
gamma_il = k_i / (1 - k_i * beta_i)
gamma_er = k_e / (1 + k_e * beta_e)
gamma_el = k_e / (1 - k_e * beta_e)
delta = sqrt(mu_i/mu_e)
rho = sqrt(eps_i/eps_e)

n = 32    # sections 
n_o = 32  # observation points
count = 0
b_direct = True
tic = dt_now() 

def R(t):
    s = 0.
    for l in arange(1, n):
        s += 1. / l * cos(l * t * pi/n)
    return -2 * pi/n * s - pi/(n**2) * ((-1)**t)

if b_direct:
    print 'Generating A_%s now (n_o = %s & n = %s): started %s\n' % (name(), n_o, n, tic)
    A = zeros((n_o, n_o, 2), dtype=complex)

    for ii in arange(n_o):
        d = [cos(2*pi*ii/n_o), sin(2*pi*ii/n_o)]      # observed direction    
        for jj in arange(n_o):
            d0 = [cos(2*pi*jj/n_o), sin(2*pi*jj/n_o)] # incident wave direction

            Q_el_inf, Q_er_inf = 0., 0.

            u = zeros((4*n,), dtype=complex)
            for l in arange(2*n):
                xdp = x1(l*pi/n) * d0[0] + x2(l*pi/n) * d0[1]
                d_nu = (x2p(l*pi/n) * d0[0] - x1p(l*pi/n) * d0[1]) / xp(l*pi/n)
                u[2*l + 0] = -2 * k * exp(1j * k * xdp) * d_nu 
                u[2*l + 1] = -2 * k * exp(1j * k * xdp) * d_nu

            E = zeros((4*n, 4*n), dtype=complex)
            for l in arange(2*n):
                E[2*l + 0, 2*l + 0] = (1.j * gamma_er / 2 + 1.j * gamma_el / 2) * 1
                E[2*l + 0, 2*l + 1] = 0
                E[2*l + 1, 2*l + 0] = 0
                E[2*l + 1, 2*l + 1] = (1.j / (2 * gamma_er) + 1.j / (2 * gamma_el)) * 1

            K = zeros((4*n, 4*n), dtype=complex)
            for l in arange(2*n):
                for m in arange(2*n):
                    K[2*l + 0, 2*m + 0] = 1.j * gamma_er * (R(l - m) * L1(pi*l/n, pi*m/n, gamma_er) + pi/n * L2(pi*l/n, pi*m/n, gamma_er)) / 2 + 1.j * gamma_el * (R(l - m) * L1(pi*l/n, pi*m/n, gamma_el) + pi/n * L2(pi*l/n, pi*m/n, gamma_el)) / 2
                    K[2*l + 0, 2*m + 1] = 1.j * (R(l - m) * M1(pi*l/n, pi*m/n, gamma_el) + pi/n * M2(pi*l/n, pi*m/n, gamma_el)) / 2 - 1.j * (R(l - m) * M1(pi*l/n, pi*m/n, gamma_er) + pi/n * M2(pi*l/n, pi*m/n, gamma_er)) / 2
                    K[2*l + 1, 2*m + 0] = 1.j * (R(l - m) * N1(pi*l/n, pi*m/n, gamma_er) + pi/n * N2(pi*l/n, pi*m/n, gamma_er)) / 2 - 1.j * (R(l - m) * N1(pi*l/n, pi*m/n, gamma_el) + pi/n * N2(pi*l/n, pi*m/n, gamma_el)) / 2
                    K[2*l + 1, 2*m + 1] =  - 1.j * (R(l - m) * Ls1(pi*l/n, pi*m/n, gamma_er) + pi/n * Ls2(pi*l/n, pi*m/n, gamma_er)) / (2 * gamma_er) - 1.j * (R(l - m) * Ls1(pi*l/n, pi*m/n, gamma_el) + pi/n * Ls2(pi*l/n, pi*m/n, gamma_el)) / (2 * gamma_el)

            z = linalg.solve(E + K, u)
            
            for l in arange(2*n):
                Q_er_inf += (z[2*l+1] * (1.j * xp(l*pi/n)) - gamma_er * (gamma_er * (d[0] * x2p(l*pi/n) - d[1] * x1p(l*pi/n))) * z[2*l]) * exp(-1.j * gamma_er * (d[0] * x1(l*pi/n) + d[1] * x2(l*pi/n)))
                Q_el_inf += (z[2*l+1] * (1.j * xp(l*pi/n)) + gamma_el * (gamma_el * (d[0] * x2p(l*pi/n) - d[1] * x1p(l*pi/n))) * z[2*l]) * exp(-1.j * gamma_el * (d[0] * x1(l*pi/n) + d[1] * x2(l*pi/n)))
            Q_er_inf = exp(-1j*pi/4) / sqrt(8*pi*gamma_er) * pi/n * Q_er_inf
            Q_el_inf = exp(-1j*pi/4) / sqrt(8*pi*gamma_el) * pi/n * Q_el_inf

            A[ii, jj, 0] = Q_er_inf
            A[ii, jj, 1] = Q_el_inf

            count += 1

            print u'processing %-5d of %5d: %3d %% ... time elapsed: %s' % (count, n_o**2, int(count * 100. / n_o**2),  lapse(dt_now() - tic))
        
    print u'\ncompleted! total time: %s' % lapse(dt_now() - tic)

    savez(os.path.join(os.getcwd(), 'data', 'A_%s' % name()), A=A)

else:
    npz = load(os.path.join(os.getcwd(), 'data', 'A_chiral_conductor.npz'))
    AA = npz['A']

    A1 = zeros((n_o, n_o), dtype=complex)
    A2 = zeros((n_o, n_o), dtype=complex)
    for ii in arange(n_o):
        for jj in arange(n_o):
            A1[ii, jj] = AA[ii, jj, 0]
            A2[ii, jj] = AA[ii, jj, 1]

    U1, La1, V1 = linalg.svd(A1.transpose())
    V1 = V1.conj().transpose()
    U2, La2, V2 = linalg.svd(A2.transpose())
    V2 = V2.conj().transpose()

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
    
    savez(os.path.join(os.getcwd(), 'data', 'xyw_chiral_conductor'), x=x, y=y, w=w)
    
    print u'\ncompleted! total time: %s' % lapse(dt_now() - tic)

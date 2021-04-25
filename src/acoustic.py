# -*- coding: utf-8 -*-

from op import *

k = 1. 
eta = k 

n_o = 16  # of observed direction
n = 16    # of precision

b_direct = 0     # direct problem: setup A 
if b_direct:
    tic = dt_now() 
    count = 0
    print 'Generating A now (using k = %s, n_o = %s & n = %s): started %s\n' % (k, n_o, n, tic)
    A = zeros((n_o, n_o), dtype=complex)
    for ii in arange(n_o):
        d = [cos(2*pi*ii/n_o), sin(2*pi*ii/n_o)]      # observed direction    
        for jj in arange(n_o):
            d0 = [cos(2*pi*jj/n_o), sin(2*pi*jj/n_o)] # incident wave direction
            def tmp_g(t):
                return -2 * exp(1j * k * (d0[0] * x1(t*pi/n) + d0[1] * x2(t*pi/n)))
            g = fromfunction(tmp_g, (2*n,))

            def R(x):
                s = 0
                for m in arange(1, n):
                    s += 1. / m * cos(m * x * pi/n)
                return -2 * pi/n * s - pi/(n**2) * ((-1)**x)
            
            K = zeros((2*n, 2*n), dtype=complex)
            for l in arange(2*n):
                for m in arange(2*n):
                    K[l, m] = R(l - m) * (L1(pi*l/n, pi*m/n, k) - 1j * eta * M1(pi*l/n, pi*m/n, k)) + pi/n * (L2(pi*l/n, pi*m/n, k) - 1j * eta * M2(pi*l/n, pi*m/n, k))
            
            x = linalg.solve(eye(2*n) + K, g)
            
            def u__(t):
                return (k * (d[0] * x2p(t*pi/n) - d[1] * x1p(t*pi/n)) + eta * xp(t*pi/n)) * exp(-1j * k * (d[0] * x1(t*pi/n) + d[1] * x2(t*pi/n)))
            u_ = fromfunction(u__, (2*n,))

            A[ii, jj] = exp(-1j*pi/4) / sqrt(8*pi*k) * pi/n * sum(dot(u_, x), dtype=complex)
            count += 1

            print u'processing %-5d of %5d: %3d %% ... time elapsed: %s' % (count, n_o**2, int(count * 100. / n_o**2),  lapse(dt_now() - tic))
                
    print u'\ncompleted! total time: %s' % lapse(dt_now() - tic)

    savez(os.path.join(os.getcwd(), 'A'), A=A)
    sys.exit()

npz = load(os.path.join(os.getcwd(), 'A.npz'))
A = npz['A']
U, La, V = linalg.svd(A.transpose())
V = V.conj().transpose()

x_min, x_max = -5, 10
y_min, y_max = -5, 10
n_x, n_y = 100, 100 

x = linspace(x_min, x_max, n_x)
y = linspace(y_min, y_max, n_y)
W = zeros((n_x, n_y), dtype=float64)

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
        W[jj, ii] = 1. / wt
        
        count += 1
        print u'processing %-5d of %5d: %3d %% ... time elapsed: %s' % (count, n_x * n_y, int(count * 100. / (n_x * n_y)),  lapse(dt_now() - tic))

savez(os.path.join(os.getcwd(), 'xyW'), x=x, y=y, W=W)

print u'\ncompleted! total time: %s' % lapse(dt_now() - tic)

# -*- coding: utf-8 -*-

import os, sys, datetime
os.chdir(os.path.dirname(__file__))

from numpy import * 
from scipy import *
from scipy.special import psi, j0, j1, hankel1

C = -psi(1)

# ===============================================
#  utilities 
# ===============================================

def name(f):
    return u'_'.join(os.path.splitext(os.path.split(f)[1])[0].split('_')[1:])

def dt_now():
    return datetime.datetime.now()

def lapse(dt):
    n = (1 if dt.microseconds > 5e5 else 0) + dt.seconds + dt.days * 24 * 3600 
    mm, ss = divmod(n, 60)
    hh, mm = divmod(mm, 60)
    return '%02d:%02d:%02d' % (hh, mm, ss)

# ===============================================
#  latex table output 
# ===============================================

def make_table(caption, content, real, imag, l_or_r='l'):
    s = 'el' if l_or_r == 'l' else 'er'
    tmp = ur"""
\begin{table*}
  \centering
  \renewcommand{\arraystretch}{1.1}
  \caption{$Q_\text{%s}^\infty$, %s}
  \begin{tabular}{@{}llll@{}}
    \toprule
    n & $\Re{Q_\text{%s}^\infty}$ & $\Im{Q_\text{%s}^\infty}$ & error \\
    \midrule
%s
    \bottomrule
  \end{tabular}
  \\ 
  $$\text{exact }\Re{Q_\text{%s}^\infty}=%s,\,\Im{Q_\text{%s}^\infty}=%s$$  
\end{table*}
"""
    return tmp % (s, caption, s, s, content.strip(), s, real, s, imag) 

# ===============================================
# kite-shaped domain
# ===============================================

def x1(t):
    return cos(t) + 0.65 * cos(2 * t) - 0.65

def x1p(t):
    return -sin(t) - 1.3 * sin(2 * t)

def x1pp(t):
    return -cos(t) - 2.6 * cos(2 * t)

def x2(t):
    return 1.5 * sin(t)

def x2p(t):
    return 1.5 * cos(t)

def x2pp(t):
    return -1.5 * sin(t)

def xp(t):
    return sqrt(x1p(t)**2 + x2p(t)**2)

def x(t):
    return sqrt(x1(t)**2 + x2(t)**2)

# ===============================================
#  boundary integral operators
# ===============================================

def r(t, y):
    return sqrt((x1(t) - x1(y))**2 + (x2(t) - x2(y))**2)

def M(t, y, k):
    return 1.j / 2 * hankel1(0, k * r(t, y)) * xp(y)

def L(t, y, k):
    return 1.j * k / 2 * (x2p(y) * (x1(t) - x1(y)) - x1p(y) * (x2(t) - x2(y))) * hankel1(1, k * r(t, y)) / r(t, y)

def Ls(t, y, k):
    return xp(y) / xp(t) * L(y, t, k)

def N(t, y, k):
    return (x2p(t) * (x1(y) - x1(t)) - x1p(t) * (x2(y) - x2(t))) * (x2p(y) * (x1(y) - x1(t)) - x1p(y) * (x2(y) - x2(t))) / (xp(t) * r(t, y)**4) * (1.j * k**2 / 2 * hankel1(0, k * r(t, y)) * r(t, y)**2 - 1.j * k * hankel1(1, k * r(t, y)) * r(t, y) + 2 / pi) + (x1p(t) * x1p(y) + x2p(t) * x2p(y)) / (xp(t) * r(t, y)**2) * (1.j * k / 2 * hankel1(1, k * r(t, y)) * r(t, y) - 1 / pi)

def M1(t, y, k):
    if t == y:
        return -1 / (2*pi) * xp(t)

    return -1 / (2*pi) * j0(k * r(t, y)) * xp(y)  

def M2(t, y, k):
    if t == y:
        return (1.j/2 - C/pi - 1/(2*pi) * log(k**2 / 4 * xp(t)**2)) * xp(t)

    return M(t, y, k) - M1(t, y, k) * log(4 * sin((t - y) / 2)**2)

def L1(t, y, k):
    if t == y:
        return 0.

    return -k / (2*pi) * (x2p(y) * (x1(t) - x1(y)) - x1p(y) * (x2(t) - x2(y))) * j1(k * r(t, y)) / r(t, y)

def L2(t, y, k):
    if t == y:
        return -1/(2*pi) * (x1p(t) * x2pp(t) - x2p(t) * x1pp(t)) / xp(t)**2

    return L(t, y, k) - L1(t, y, k) * log(4 * sin((t - y) / 2)**2)

def Ls1(t, y, k):
    if t == y:
        return 0.

    return xp(y) / xp(t) * L1(y, t, k)

def Ls2(t, y, k):
    if t == y:
        return L2(t, t, k)

    return xp(y) / xp(t) * L2(y, t, k)

def N1(t, y, k):
    if t == y:
        return -k**2 / (4*pi) * xp(t)

    return (x2p(t) * (x1(y) - x1(t)) - x1p(t) * (x2(y) - x2(t))) * (x2p(y) * (x1(y) - x1(t)) - x1p(y) * (x2(y) - x2(t))) / (xp(t) * r(t, y)**4) * (-k**2 / (2*pi) * j0(k * r(t, y)) * r(t, y)**2 + k / pi * j1(k * r(t, y)) * r(t, y)) - k / (2*pi) * (x1p(t) * x1p(y) + x2p(t) * x2p(y)) / (xp(t) * r(t, y)) * j1(k * r(t, y)) 

def N2(t, y, k):
    if t == y:
        return k**2 / 2 * (1.j/2 - C/pi - 1/(2*pi) * log(k**2 / 4 * xp(t)**2) + 1 / (2*pi)) * xp(t)

    return N(t, y, k) - N1(t, y, k) * log(4 * sin((t - y) / 2)**2)

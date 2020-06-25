from __future__ import division
import numpy as np
from sympy import *
from sympy.plotting import plot
from sympy.plotting import plot3d
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import axis as ax
import cplot
from itertools import count, islice
from scipy.special import binom
from math import pi
x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)

# function f(x) f und Ihre Ableitung f_

f = exp(-x**2)
f_ = diff(f, x, 1)

np_arange_test = np.arange(-2, 2, 2, dtype=float)
l = [*np_arange_test]
# print(l)

# print(len(l))
# #l = [1, 2, 0]
# print(l)

from sympy import Symbol, plot
import matplotlib.pyplot as plt

# transfer to


def move_sympyplot_to_axes(p, ax):
    backend = p.backend(p)
    backend.ax = ax
    backend._process_series(backend.parent._series, ax, backend.parent)
    # backend.ax.spines['right'].set_color('blue')
    # backend.ax.spines['bottom'].set_position('zero')
    # backend.ax.spines['top'].set_color('blue')
    # backend.ax.spines['left'].set_color('red')
    backend.ax.set_ylim((-1, 1))
    backend.ax.set_xlim((-2, 2))
    backend.ax.legend()
    backend.ax.grid(True)
    plt.close(backend.fig)


# x1y1 = p1.get_points()


# print(len(list))
# i = 0
# for i in l[1:]:  # x0 values as dict for evalf - here: only 1 entry
#     x0 = i
#     print(x0)
#     values = {x: x0}
#     print('x0:', x0)


# # evalf funtion
# # print(evalf(f, subs={x:0}))
#     y0_x0 = f.evalf(subs=values)
#     print('f(x0):', y0_x0)


# # tangente an x0
#     f_x0 = f_.evalf(subs=values)
#     print('f\'(x0):', f_x0)
#     g = f_.evalf(subs=values) * (x - x0) + f.evalf(subs=values)


# ############ plot #############
#     p1 = plot(f, (x, -3, 3), show=False)
#     p2 = plot(f_, (x, -3, 3), show=False)
#     p3 = plot(g, (x, -3, 3), show=False)
#     # p1.append(p2[0])
#     p1.append(p3[0])

#     fig, (ax, ax2) = plt.subplots(2, 1)
#     move_sympyplot_to_axes(p1, ax)
#     move_sympyplot_to_axes(p2, ax2)
#     # move_sympyplot_to_axes(p3, ax3)

# from sympy import symbols, Eq, I
# from sympy.plotting import plot_implicit

# x, y = symbols('x y', real=True)
# z = x + I * y
# expr = Eq(abs(z), 1)
# p4 = plot_implicit(expr, show=False)
# move_sympyplot_to_axes(p4, ax)
# plt.show()


# domain coloring
import numpy as np
import matplotlib.pyplot as plt


def f(z):
    return (z - 1) * (z + 1)**2 / ((z + 1j) * (z - 1j)**2)
    # return ((z))
    # return sum([1 / n**z                for n in range(1, 40)])


# xs, xe, rx, ys, ye, ry = -30, 30, 1000, -30, 30, 1000
# x, y = np.ogrid[xs:xe:1j * rx, ys:ye:1j * ry]
# print(x)
# print(y)
# plt.imshow(np.angle(f((x - 1j * y).T)))
# plt.legend()
# plt.show()

# p1.show()
print(count(0))


def zeta(s, t=5):
    # if s == 1:
    #    return float('inf')
    term = (1 / 2 ** (n + 1) * sum((-1)**k * binom(n, k) * (k + 1)**-s
                                   for k in range(n + 1))
            for n in count(0))
    return sum(islice(term, t)) / (1 - 2**(1 - s))


# plot3d
# p3 = plot(g, -5, 5)


print(abs(1 + 1j))
#plot3d(zeta, (x, -5, 5), (y, -5, 5))
# this intervall Re Axis, Intervall Im Axis, number of points to calculate in each axis
cplot.show(zeta, -20, 10, -100, 100, 1000, 1000)

#cplot.save_fig("out.png", np.tan, -5, +5, -5, +5, 200, 100)
#cplot.save_img("out.png", np.tan, -5, +5, -5, +5, 100, 100)

# There is a tripcolor function as well for triangulated 2D domains
# cplot.tripcolor(triang, z)

# The function get_srgb1 returns the SRGB1 triple for every complex input value.
# (Accepts arrays, too.)
#z = 2 + 5j
#val = cplot.get_srgb1(z)
# print(val)

from random import randint, uniform
import pygame
import matplotlib
from mpl_toolkits import mplot3d
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from vector_classes import (Vector, Vec1, Vec2, Vec3, Vec6, CarForSale, Matrix5_by_3, 
    Matrix, ImageVector, Function, Function2, LinearFunction, QuadraticFunction, Polynomial,
    PolygonModel, Ship, Asteroid)
from mfp_modules import colors, vector_drawing, vectors

def plot_function(f,tmin,tmax,tlabel=None,xlabel=None,axes=False, **kwargs):
    ts = np.linspace(tmin,tmax,1000)
    if tlabel:
        plt.xlabel(tlabel,fontsize=18)
    if xlabel:
        plt.ylabel(xlabel,fontsize=18)
    plt.plot(ts, [f(t) for t in ts], **kwargs)
    if axes:
        total_t = tmax-tmin
        plt.plot([tmin-total_t/10,tmax+total_t/10],[0,0],c='k',linewidth=1)
        plt.xlim(tmin-total_t/10,tmax+total_t/10)
        xmin, xmax = plt.ylim()
        plt.plot([0,0],[xmin,xmax],c='k',linewidth=1)
        plt.ylim(xmin,xmax)

def volume(t):
    return (t-4)**3 / 64 + 3.3

def average_flow_rate(v,t1,t2):
    return (v(t2) - v(t1)) / (t2 - t1)
# print(average_flow_rate(volume,4,9))

def flow_rate(t):
    return 3*(t-4)**2 / 64

def decreasing_volume(t):
    if t < 5:
        return 10 - (t**2)/5
    else:
        return 0.2*(10-t)**2

def secant_line(f,x1,x2):
    def line(x):
        return f(x1) + (x-x1) * (f(x2) - f(x1))/(x2-x1)
    return line

line = secant_line(flow_rate,4,9)
# print(line(3))

def plot_secant(f,x1,x2,color='k'):
    line = secant_line(f,x1,x2)
    plot_function(line,x1,x2,c=color)
    plt.scatter([x1,x2],[f(x1),f(x2)],c=color)

# print(np.arange(0,10,0.5))

def interval_flow_rates(v,t1,t2,dt):
    return [(t,average_flow_rate(v,t,t+dt))
                for t in np.arange(t1,t2,dt)]

def plot_interval_flow_rates(volume,t1,t2,dt):
    series = interval_flow_rates(volume,t1,t2,dt)
    times = [t for (t,_) in series]
    rates = [q for (_,q) in series]
    plt.scatter(times,rates)

##!# protip, if your function generates plots, they're held in memory >> draw in 1 chart...
# plot_secant(flow_rate,5,8)
# plot_function(flow_rate,0,10)
# plot_interval_flow_rates(volume,0,10,1)
# plot_interval_flow_rates(volume,0,10,.08)

## Exercise 8.4
# plot_interval_flow_rates(decreasing_volume,0,10,0.5)

# def plot_many(f,t1,t2,dt,function=True,intervals=False,secant=False):
#     if function == True:
#         plot_function(f,0,10)
#     if intervals == True:
#         plot_interval_flow_rates(f,t1,t2,dt)
#     if secant == True:
#         plot_secant(f,t1,t2)
# plot_many(flow_rate,0,10,1,True,False,True)

## Exercise 8.5
# def linear_volume_function(t):
#     return 5 * t + 3

# plot_interval_flow_rates(linear_volume_function,0,10,0.25)

## this is the derivative of the volume function
def instantaneous_flow_rate(v,t,digits=9):
    tolerance = 10 ** (-digits)
    h=1
    ## get average for increasingly smaller segments of time until desired resolution is reached
    approx = average_flow_rate(v,t-h,t+h)
    for i in range(0,2*digits):
        ## variables persist in the loop and sequentially change (i=1,h=0.1; i=2,h=0.01 ...)
        h = h / 10
        next_approx = average_flow_rate(v,t-h,t+h)
        if abs(next_approx - approx) < tolerance:
            return round(next_approx,digits)
        else:
            # print(f"{next_approx} :: {i} :: {h}")
            approx = next_approx
    raise Exception("Derivative did not converge")
for i in range(0,10):
    a = i * 0.5
    instantaneous_flow_rate(volume,a)

def get_flow_rate_function(v):
    def flow_rate_function(t):
        instantaneous_flow_rate(v,t)
    return flow_rate_function

# plot_function(flow_rate,0,20)
# plot_function(get_flow_rate_function(volume),0,10)

# plt.show()

## Exercise 8.6
# print(volume(1))
## passing secant_line() with parameters and then immediately calling it with parameter 1 for time
# print(secant_line(volume,0.9999, 1.001)(1))

## Exercise 8.7
# print(average_flow_rate(volume,7.999,8.0001))
# print(instantaneous_flow_rate(volume,8))

## Exercise 8.8
def sign(x):
    return x / abs(x)
# print(average_flow_rate(sign,-0.0000001,0.0000001))
# print(-.1 / abs(-.1))

def small_volume_change(q,t,dt):
    return q(t) * dt

# print(small_volume_change(flow_rate,2,1))
# print(volume(3) - volume(2))
# print(small_volume_change(flow_rate,2,0.0001))
# print(volume(2.0001) - volume(2))

def volume_change(q,t1,t2,dt):
    return sum(small_volume_change(q,t,dt)
        for t in np.arange(t1,t2,dt))

# print(volume_change(flow_rate,0,10,0.0001))
# print(volume(10) - volume(0))
# print(volume_change(flow_rate,0,10,0.001))

def approximate_volume(q,v0,dt,T):
    return v0 + volume_change(q,0,T,dt)

def approximate_volume_function(q,v0,dt):
    def volume_function(T):
        return approximate_volume(q,v0,dt,T)
    return volume_function

## approximate_volume_function called and passed:
#       1. flow_rate, volume at 0, interval scale
#       2. volume at 0
#       3. interval scale
#       4. Time
## when plot_function runs, it plots 1000 points (hard-coded). For each of these points, it
#  calls approximate_volume_function and calculates the point for that time sequence. 
#  approximate_volume could not be called directly by plot_function because plot_function provides
#  the time series so that variable needs to be curried in.
#   ts = np.linspace(tmin,tmax,1000)
#   plt.plot(ts, [f(t) for t in ts], **kwargs) <<< 'volume_function(t)' is what's being called here
## full stack:
### approximate_volume_function >> calls:
#### approximate_volume ( adds volume_change to original volume ) > calls:
##### volume_change ( sums up all the small_volume_changes calculated over intervals ) > calls:
###### small_volume_change ( generates volume amount for each interval (e.g. .1 hour) ) > calls:
####### flow_rate ( actual math {q(t)} )

# plot_function(approximate_volume_function(flow_rate,2.3,0.125),0,10)
# plot_function(volume,0,10)
# plt.show()

def get_volume_function(q,v0,digits=6):
    def volume_function(T):
        tolerance = 10 ** (-digits)
        dt = 1
        approx = v0 + volume_change(q,0,T,dt)
        for i in range(0,digits*2):
            dt = dt / 10
            next_approx = v0 + volume_change(q,0,T,dt)
            if abs(next_approx - approx) < tolerance:
                return round(next_approx,digits)
            else:
                approx = next_approx
        raise Exception("Did not converge!")
    return volume_function

# v = get_volume_function(flow_rate,2.3)
# print(v(1))
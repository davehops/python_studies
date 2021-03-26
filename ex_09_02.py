from math import pi,sin,cos
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
from mfp_modules import colors, vector_drawing, vectors, draw2d, draw3d

# for x,y in ranges:
#     t = 0
#     s = (0,0)
#     v = (1,0)
#     a = (0,0.2)
#     dt = y
#     print(x)
#     for _ in range(0,x):
#         t += 2
#         s = vectors.add(s, vectors.scale(dt,v))
#         v = vectors.add(v, vectors.scale(dt,a))
#         if x < 7:
#             positions_low.append(s)
#         elif x == 10:
#             positions_medium.append(s)
#         elif x > 10:
#             positions_medium_high.append(s)
#         else:
#             positions_high.append(s)

def pairs(lst):
    return list(zip(lst[:-1],lst[1:]))


## eulers_method params:
# 1. initial position
# 2. initial velocity
# 3. constant acceleration vector
# 4. time in seconds to plot
# 5. number of points to plot for interval
def eulers_method(s0,v0,a,total_time,step_count):
    positions = [s0]
    s = s0
    v = v0
    dt = total_time/step_count
    for _ in range(0,step_count):
        s = vectors.add(s,vectors.scale(dt,v))
        v = vectors.add(v,vectors.scale(dt,a))
        positions.append(s)
    return positions

def eulers_method_overapprox(s0,v0,a,total_time,step_count):
    positions = [s0]
    s = s0
    v = v0
    dt = total_time/step_count
    for _ in range(0,step_count):
        v = vectors.add(v,vectors.scale(dt,a))
        s = vectors.add(s,vectors.scale(dt,v))
        positions.append(s)
    return positions

# positions_low = eulers_method((0,0),(1,0),(0,0.2),10,5)
# positions_medium = eulers_method((0,0),(1,0),(0,0.2),10,9)
# positions_medium_high = eulers_method((0,0),(1,0),(0,0.2),10,25)
# positions_high = eulers_method((0,0),(1,0),(0,0.2),10,15)
# positions_high_over = eulers_method_overapprox((0,0),(1,0),(0,0.2),10,15)
# positions_high_exact = eulers_method((0,0),(1,0),(0,0.2),10,500)

# print(positions_low)
# print(pairs(positions_low))

# draw2d.draw2d(
#     # draw2d.Points2D(*positions_low, color=colors.gray),
#     # *[draw2d.Segment2D(t,h,color=colors.gray) for (h,t) in pairs(positions_low)],
#     # draw2d.Points2D(*positions_medium, color=colors.blue),
#     # *[draw2d.Segment2D(t,h,color=colors.blue) for (h,t) in pairs(positions_medium)],
#     # draw2d.Points2D(*positions_medium_high, color=colors.black),
#     # *[draw2d.Segment2D(t,h,color=colors.blue) for (h,t) in pairs(positions_medium_high)],
#     draw2d.Points2D(*positions_high, color=colors.orange),
#     draw2d.Points2D(*positions_high_over, color=colors.blue),
#     draw2d.Points2D(*positions_high_exact, color=colors.black)
#     # save_as='mfp_static_content/4eulers.svg'
# )

## Exercise 9.4

# angle = 20 * pi/180
# s0 = (0,1.5)
# v0 = (30*cos(angle),30*sin(angle))
# a = (0,-9.81)

# result = eulers_method(s0,v0,a,3,100)

## Exercise 9.5
# def baseball_trajectory():
#     longest_range = [(0,0)]
#     for i in range(640,720):
#         deg = i/16
#         angle = deg * pi/180
#         s0 = (0,1.5)
#         v0 = (30*cos(angle),30*sin(angle))
#         a = (0,-9.81)
#         new_range = [(x,y) for (x,y) in eulers_method(s0,v0,a,10,500000) if y>=0]
#         print(new_range[-1:][0][0])
#         if new_range[-1:][0][0] > longest_range[-1:][0][0]:
#             print(f"{deg} degrees is now the best throwing angle for distance")
#             longest_range = new_range
#     return longest_range

# longest_result = baseball_trajectory()

# draw2d.draw2d(
#     draw2d.Points2D(*longest_result, color=colors.black)
# )

# def baseball_trajectory(deg):
#     angle = deg * pi/180
#     s0 = (0,1.5)
#     v0 = (30*cos(angle),30*sin(angle))
#     a = (0,-9.81)
#     range = [(x,y) for (x,y) in eulers_method(s0,v0,a,10,800000) if y>=0]
#     return range

# # result43 = baseball_trajectory(43)
# # result442 = baseball_trajectory(44.2)
# result445 = baseball_trajectory(44.5)
# result443 = baseball_trajectory(44.3)

# draw2d.draw2d(
#     # draw2d.Points2D(*result43, color=colors.black),
#     # draw2d.Points2D(*result442, color=colors.red),
#     draw2d.Points2D(*result445, color=colors.orange),
#     draw2d.Points2D(*result443, color=colors.blue)
# )

## Exercise 9.6
traj3d = eulers_method((0,0,0),(1,2,0),(0,-1,1), 10, 1000)
draw3d.draw3d(
    draw3d.Points3D(*traj3d)
)
print(eulers_method((0,0,0),(1,2,0),(0,-1,1), 10, 1000)[-1])

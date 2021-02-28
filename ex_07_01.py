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

matrix = np.array(((1,-1),(1,2)))
output = np.array((0,8))
### linalg.solve takes a matrix and an output vector and finds the input vector that produced it
# print(np.linalg.solve(matrix,output))

pygame.init()
ship = Ship()
width, height = 400, 400
#### pygame tut begin
# size = (width, height)
# screen = pygame.display.set_mode(size)
# pygame.display.set_caption("My First Game")
#### pygame tut end
asteroid_count = 10
asteroids = [Asteroid() for _ in range(0,asteroid_count)]
laser = ship.laser_segment()
keys = pygame.key.get_pressed()
if keys[pygame.K_SPACE]:
    draw_segment(*laser)
    for asteroid in asteroids:
        if asteroid.does_interact(laser):
            asteroids.remove(asteroid)

for ast in asteroids:
    ast.x = randint(-9,9)
    ast.y = randint(-9,9)

GREEN = colors.green
def draw_poly(screen, polygon_model, color=GREEN):
    pixel_points = [to_pixels(x,y) for x,y in polygon_model.transformed()]
    pygame.draw.aalines(screen, color, True, pixel_points, 10)

def to_pixels(x,y):
    return (width/2 + width * x / 20, height/2 - height * y / 20)

## Exercise 7.13
asteroid = PolygonModel([(2,7), (1,5), (2,3), (4,2), (6,2), (7,4), (6,6), (4,6)])
# print(asteroid.does_intersect([(0,0),(0.99999,5.0000000000)]))
# print(asteroid.does_intersect([(0,0),(7,7)]))

## Exercise 7.14
square1 = PolygonModel([(0,0), (3,0), (3,3), (0,3)])
square2 = PolygonModel([(1,1), (4,1), (4,4), (1,4)])
square3 = PolygonModel([(-3,-3), (-2,-3), (-2,-2), (-3,-2)])
# print(square1.does_collide(square2))
# print(square1.does_collide(square3))
# print(square2.does_collide(square3))

## Simple visualization using vector drawings
# square1 = [(0,0), (3,0), (3,3), (0,3)]
# square2 = [(1,1), (4,1), (4,4), (1,4)]
# square3 = [(-3,-3), (-2,-3), (-2,-2), (-3,-2)]
# vector_drawing.draw(
#    vector_drawing.Polygon(*square1, color=colors.blue),
#    vector_drawing.Polygon(*square2, color=colors.red),
#    vector_drawing.Polygon(*square3, color=colors.green)
# )

def random_matrix(rows, columns):
    return tuple(
        tuple(uniform(-10,10) for j in range(0,columns))
        for i in range(0,rows)
    )
def random_round_scalar():
    return round(uniform(-10,10),0)
def random_vec2():
    return ((random_round_scalar(), random_round_scalar()))
def round_matrix(matrix):
    return tuple(
        tuple(
            round(x,0) for x in row
        ) for row in matrix
    )
def multiply_matrix_vector_by_dot_product(matr,vec):
    return tuple(
        vectors.dot(row,vec)
        for row in matr
    )

rand1 = ((6.0, 2.0), (4.0, -7.0))
exercise_test = ((2,1), (4,2))
ex_test_2 = ((1,-2))
### make yourself homework
# for _ in range(0,10):
#     a = round_matrix(random_matrix(3,2))
#     b = random_vec2()
#     print(a)
#     print(b)
#     print(multiply_matrix_vector_by_dot_product(a,b))
#     print("-------------------------------------------")

matrix = np.array(((1,1,-1),(0,2,-1),(1,0,1)))
vector = np.array((-1,3,2))
# print(np.linalg.solve(matrix,vector))

## Exercise 7.16
mtr716 = np.array(((5,4),(12,11)))
vec716 = np.array((-3,3))
# print(np.linalg.solve(mtr716,vec716))

## Exercise 7.19
def plane_equation(p1,p2,p3):
    parallel1 = vectors.subtract(p2,p1)
    parallel2 = vectors.subtract(p3,p1)
    a,b,c = vectors.cross(parallel1, parallel2)
    d = vectors.dot((a,b,c), p1)
    return a,b,c,d

# print(plane_equation((1,1,1),(3,0,0),(0,3,0)))

fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)

# # x + y - z = -1
# def z1(x,y):
#     return x + y + 1

# # 2y - z  = 3
# def z2(x,y):
#     return 2*y - 3

# # 2y - z  = 3
# def z3(x,y):
#     return -x + 2

# Z1 = X + Y + 1
# Z2 = 2*Y - 3
# Z3 = -X + 2

## Exercise 7.22
# def z1(x,y):
#     return x + y

# def z2(x,y):
#     return x - y

# def z3(x,y):
#     return x - 3
# Z1 = X + Y
# Z2 = X - Y
# Z3 = X - 3

## Exercise 7.24a
# # z - y = 0
# def z1(x,y):
#     return -y

# # z + y = 0
# def z2(x,y):
#     return y

# # z + x = 0
# def z3(x,y):
#     return x
# Z1 = -Y
# Z2 = Y
# Z3 = X

## plot lines for 7.24a
## plot groups describe starting and end values ([x-start,x-end],[y-start,y-end],[z-start,z-end])
# ax.scatter([-6],[-6],[-6],s=30,c='b')
# ax.scatter([6],[6],[6],s=30,c='b')
# ax.plot([-6,6],[-6,6],[-6,6],c='b')
# ax.plot([-6,6],[6,-6],[6,-6],c='m')
# ax.plot([6,-6],[6,-6],[-6,6],c='k')
# ax.scatter([0],[0],[0],s=50,c='k')


## plot lines for 7.24b
# z + x = 0
# def z1(x,y):
#     return -y

# # z - x = 0
# def z2(x,y):
#     return y

# # z = 3
# def z3(x,y):
#     return 0*y

# ax.plot([-6,6],[0,0],[0,0],c='k')


## plot lines for 7.24c
# z - y = 0
def z1(x,y):
    return y + 0.5

# 2z - 2y = 0
def z2(x,y):
    return y - 0.5

# 3z - 3y = 0
def z3(x,y):
    return y


## surface plots work for all equation pairs
# ax.plot_surface(X,Y,z1(X,Y), rcount=1, ccount=1, color='b', alpha=0.3, edgecolor='none')
# ax.plot_surface(X,Y,z2(X,Y), rcount=1, ccount=1, color='y', alpha=0.3, edgecolor='none')
# ax.plot_surface(X,Y,z3(X,Y), rcount=1, ccount=1, color='m', alpha=0.3, edgecolor='none')
# ax.zaxis.set_major_formatter('{x:.02f}')
# ax.xaxis.set_major_formatter('{x:.03f}')
# plt.show()

## Exercise 7.25
# matrix = np.array(((0,0,0,0,1),(0,1,0,0,0),(0,0,0,1,0),(1,0,0,0,0),(1,1,1,0,0)))
# vector = np.array((3,1,-1,0,-2))
# print(np.linalg.solve(matrix,vector))
def matrix_multiply(a,b):
    return tuple(
        tuple(vectors.dot(row,col) for col in zip(*b))
        for row in a
    )
## Exercise 7.26
matrix = np.array(((1,1,-1),(0,2,-1),(1,0,1)))
vector = np.array((-1,3,2))
inverse = np.linalg.inv(matrix)
# print(inverse)
# print(np.matmul(inverse,matrix))
# print(matrix_multiply(inverse,matrix))
# print(np.matmul(inverse, vector))

# w = np.array((1,3,-7))
# a = np.array(((1,-1,0),(0,-1,-1),(1,0,2)))
# print(np.linalg.solve(a,w))

w = np.array((5,5))
a = np.array(((10,1),(3,2)))
# print(np.linalg.solve(a,w))

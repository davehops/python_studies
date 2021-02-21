from mfp_modules import colors, vectors, vector_drawing, draw2d, draw3d, draw_model, transforms
from vector_arrays import dino_vectors
from random import uniform, randint, random
from math import sqrt, pi, cos, sin, isclose
import matplotlib
# from numpy import random
from hypothesis import given, example
import hypothesis.strategies as st
from abc import abstractproperty
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from vector_classes import (Vector, Vec1, Vec2, Vec3, Vec6, CarForSale, Matrix5_by_3, 
    Matrix, ImageVector, Function, Function2, LinearFunction, QuadraticFunction, Polynomial)
from json import loads, dumps
import os
from pathlib import Path
import numpy as np
import io
import PIL.Image as Image
from array import array

# print(Vec2(1,3) - Vec2(5,1))
# v = Vec2(3,4)
# w = v.add(Vec2(-2,6))
# print (w.x)
# v = Vec2(3,4)
# w = v.scale(50)
# print (w.x, w.y)
# print(Vec3(1,0,1).__eq__(Vec3(1,0,0)))

## Exercise 6.4,5
## Exercise 6.1,2,3,6,7 in vector_classes
def random_scalar():
    return uniform(-10,10)
def random_vec2():
    return Vec2(random_scalar(), random_scalar())
a = random_scalar()
u, v = random_vec2(), random_vec2()
# test below will fail
# assert a * (u + v) == a * v + a * u

def approx_equal_vec2(v,w):
    ## in print, ',' separates variables by a single space.
    # print("first assertion is", isclose(v.x,w.x),"and second assertion is", isclose(v.y,w.y) )
    return isclose(v.x,w.x) and isclose(v.y,w.y)
assert approx_equal_vec2(a * (u + v), a * v + a * u)

# for _ in range(0,100):
#     a = random_scalar()
#     u, v = random_vec2(), random_vec2()
#     assert approx_equal_vec2(a * (u + v), a * v + a * u)

def test(zero, eq, a, b, u, v, w):
    assert eq(u + v, v + u)
    assert eq(u + (v + w), (u + v) + w)
    assert eq(a * (b * v), (a * b) * v)
    assert eq(1 * v, v)
    assert eq(-1 * v + v, zero)
    assert eq((a + b) * v, a * v + b * v)
    assert eq(a * v + a * w, a * (v + w))
    assert eq(v + zero, v)
    assert eq(-v + v, zero)
    assert eq(0 * v, zero)

# for i in range(0,100):
#     a,b = random_scalar(), random_scalar()
#     u,v,w = random_vec2(), random_vec2(), random_vec2()
#     test(Vec2.zero(),approx_equal_vec2,a,b,u,v,w)

def random_vec3():
    return Vec3(random_scalar(), random_scalar(), random_scalar())

def approx_equal_vec3(v,w):
    if not isclose(v.x, w.x) or not isclose(v.y, w.y) or not isclose(v.z, w.z):
        print(v,w)
    return isclose(v.x, w.x) and isclose(v.y, w.y) and isclose(v.z, w.z) 

# for i in range(0,100):
#     a,b = random_scalar(), random_scalar()
#     u,v,w = random_vec3(), random_vec3(), random_vec3()
#     test(Vec3.zero(),approx_equal_vec3,a,b,u,v,w)

# print(Vec3(1,2,3) + Vec3(1,2,3))

# print(Vec6(1,2,3,4,5,6) + Vec6(1,2,3,4,5,6))
# print(Vec6(1,2,3,4,5,6) / 2)
# print(Vec1(1) + Vec1(2 ))
            
## Exercise 6.8
# for i in range(0,100):
#     a,b = random_scalar(), random_scalar()
#     u,v,w = random_scalar(), random_scalar(), random_scalar()
#     test(0, isclose, a,b,u,v,w)

# contents = Path('mfp_modules/cargraph.json').read_text()
# THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
# path = os.path.join(THIS_FOLDER, 'mfp_modules/cargraph.json')
# print(path)
###>>> example from online...
# Opening JSON file 
# f = open('data.json',) 
# returns JSON object as a dictionary 
# data = json.load(f) 
# Iterating through the json list 
# for i in data['emp_details']: 
#     print(i) 
# Closing file 
# f.close() 

base_path = Path(__file__).parent
contents = Path(base_path / 'mfp_static_content/cargraph.json').read_text()
cg = loads(contents)
cleaned = []
def parse_date(s):
    input_format="%m/%d - %H:%M"
    return datetime.strptime(s,input_format).replace(year=2018)
for car in cg['cars']:
    try:
        row = CarForSale(int(car[1]), float(car[3]), float(car[4]), parse_date(car[6]), car[2],  car[5],  car[7], car[8])
        cleaned.append(row)
    except: pass
cars = cleaned
# print((cars[0] + cars[1]).__dict__)
average_prius = sum(cars, CarForSale.zero()) / len(cars)
# print(average_prius.__dict__)

def plot(fs, xmin, xmax):
    xs = np.linspace(xmin,xmax,100)
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    for f in fs:
        ys = [f(x) for x in xs]
        plt.plot(xs,ys)
    plt.show()  
def f(x):
    return 0.5 * x + 3
def g(x):
    return sin(x)
def add_functions(f,g):
    def new_function(x):
        return f(x) + g(x)
    return new_function
def multiply_functions(f,g):
    def new_function(x):
        return f(x) * g(x)
    return new_function
h = add_functions(f,g)
# plot([f,g,h],-10,10)

def i(x):
    return 3 * sin(x)
# plot([g,i],-10,10)

##### image reading, manipulation and saving
## Exercise 6.19,20,21
inside_img = Path(base_path / 'mfp_static_content/inside.jpg')
outside_img = Path(base_path / 'mfp_static_content/outside.jpg')
def parse_date(s):
    input_format="%m/%d - %H:%M"
    return datetime.strptime(s,input_format).replace(year=2018)
now = datetime.now()
stamp = now.strftime("%m-%d_%H_%M_%S")
new_path = 'mfp_static_content/combine_' + stamp + '.jpg'
combined_image_path = Path(base_path / new_path )

def random_image():
    return ImageVector([(randint(0,255),randint(0,255),randint(0,255))
        for i in range(0,300 * 300)])
# fuzzz = random_image()
# staticcy = ImageVector(inside_img) + fuzzz
# ##### combined_img = 0.1 * ImageVector(inside_img) + 0.9 * ImageVector(outside_img)
# white = ImageVector([(255,255,255) for _ in range(0,300*300)])
# # combined_image_unchanged = ImageVector(inside_img) + ImageVector.zero()
# # combined_img = white - ImageVector(inside_img)
# image = Image.open(io.BytesIO(staticcy._repr_png_()))
# image.save(combined_image_path, 'JPEG')

def approx_equal_image(i1,i2):
    return all([
        isclose(c1,c2)
        for p1,p2 in zip(i1.pixels,i2.pixels)
        for c1,c2 in zip(p1,p2)
    ])

# for i in range(0,100):
#     a,b = random_scalar(), random_scalar()
#     ## use f-strings to concatenate variables in python 3.6+
#     print(f"{a} :*: {b}")
#     u,v,w = random_image(), random_image(), random_image()
#     test(ImageVector.zero(), approx_equal_image, a,b,u,v,w)


## Exercise 6.9
def random_time():
    return CarForSale.retrieved_date - timedelta(days=uniform(0,10))

def approx_equal_time(t1,t2):
    test = datetime.now()
    return isclose((test-t1).total_seconds(), (test-t2).total_seconds())

def random_car():
    return CarForSale(randint(1990,2019),randint(0,250000),
            27000. * random(), random_time())

def approx_equal_car(c1,c2):
    return (isclose(c1.model_year, c2.model_year)
            and isclose(c1.mileage, c2.mileage)
            and isclose(c1.price, c2.price)
            and approx_equal_time(c1.posted_datetime, c2.posted_datetime))

# for i in range(0,100):
#     a,b = random_scalar(), random_scalar()
#     ## use f-strings to concatenate variables in python 3.6+
#     print(f"{a} :: {b}")
#     u,v,w = random_car(), random_car(), random_car()
#     test(CarForSale.zero(), approx_equal_car, a,b,u,v,w)

## Exercise 6.10 - class structure for function in 'vector_classes'
f = Function(lambda x: 0.5 * x +3)
g = Function(sin)
# plot([f, g, f+g, 3*g], -10, 10)

## Exercise 6.11,12 - function equality
def approx_equal_function(f,g):
    results = []
    for _ in range(0,10):
        x = uniform(-10,10)
        results.append(isclose(f(x),g(x)))
    return all(results)

# print(approx_equal_function(lambda x: (x*x)/x, lambda x: x))

#####!!!!Need to sort out Polynomial class for this to work
def random_function():
    degree = randint(0,5)
    p = Polynomial(*[uniform(-10,10) for _ in range(0,degree)])
    return Function(lambda x: p(x))

# for i in range(0,100):
#     a,b = random_scalar(),random_scalar()
#     u,v,w = random_function(), random_function(), random_function()
#     test(Function.zero(), approx_equal_function, a,b,u,v,w)

## Exercise 6.13
# class Function2 in vector_classes doc
h = Function2(lambda x,y: x+y)
i = Function2(lambda x,y: x-y+1)
# print((h+i)(3,10))

## Exercise 6.15
class Matrix2_by_2(Matrix):
    def rows(self):
        return 2
    def columns(self):
        return 2
# print(2 * Matrix2_by_2(((1,2),(3,4))) + Matrix2_by_2(((1,2),(3,4))))

## Exercise 6.16
# defining properties of a vector space:
# A vector is an object equipped with a *suitable* way to add it to other vectors and multiply
# it by scalars (v+w = w+v), (u+(v+w) = (u+v)+w = u+v+w)
# if a & b are scalars and v is a vector, (a*(b*v) = (a*b)*v):
# (1 * v = v)
# (a * v + b * v) = ((a + b) * v)
# (a *(v + w)) = (a * v + a * w)
# A vector space is a collection of objects called vectors, equipped with suitable vector
# addition and scalar multiplication operations (obeying the rules above), such that every
# linear combination of vectors in the collection produces a vector that is also in the collection.

def random_matrix(rows, columns):
    return tuple(
        tuple(uniform(-10,10) for j in range(0,columns))
        for i in range(0,rows)
    )

def random_5_by_3():
    return Matrix5_by_3(random_matrix(5,3))

def approx_equal_matrix_5_by_3(m1,m2):
    return all([
        isclose(m1.matrix[i][j],m2.matrix[i][j])
        for j in range(0,3)
        for i in range(0,5)
    ])

# for i in range(0,100):
#     a,b = random_scalar(), random_scalar()
#     ## use f-strings to concatenate variables in python 3.6+
#     print(f"{a} :: {b}")
#     u,v,w = random_5_by_3(), random_5_by_3(), random_5_by_3()
#     test(Matrix5_by_3.zero(), approx_equal_matrix_5_by_3, a,b,u,v,w)
 
###### 6.3.5 finding subspaces of the vector space of functions
# lotzaLines = []
# for i in range(0,20):
#     lotzaLines.append(LinearFunction(-2-i,(2-i*2)))
# plot(lotzaLines,-20,20)

# Exercise 6.39
def solid_color(r,g,b):
    return ImageVector([(r,g,b) for _ in range(0,300*300)])
purple = solid_color(225,20,225)
# purple_img = Image.open(io.BytesIO(purple._repr_png_()))
####### stamp variable had colons in it originally. Colons appear to be illegal for WIN...
# save_path_str = 'mfp_static_content/combine_' + stamp + '.jpg'
# win_save_path = os.path.abspath(save_path_str)
# purple_img.save(win_save_path)

image_size = (300,300)
total_pixels = image_size[0] * image_size[1]
square_width = 10
square_count = 30

## i = image pixels (w or h) divided by image width,
#### purpose of i = gives you which square on x grid pixel will be averaged to
## j = image pixels (w or h) remainder of division by image width
#### purpose of j = gives you which square on y grid pixel will be averaged to
def ij(n):
    return (n // image_size[0], n % image_size[1])

def to_lowres_grayscale(img):
    ## create matrix of square_count width/height
    matrix = [
        [0 for i in range(0,square_count)]
        for j in range(0,square_count)
    ]
    ## loop over all image pixels
    for (n,p) in enumerate(img.pixels):
        i,j = ij(n)
        # if j == square_width or j == 0 or j == 15:
        #     print(f"{i} :*: {j} :*: {n}")
        ## r,g,b of each pixel for each grid coordinate is added together and 
        ## multiplied by a percentage to give weighted average of pixels
        ## in each square (e.g. 0.000005925252525...)
        weight = 1.0 / (3 * square_width * square_width)
        matrix[i // square_width][j // square_width] += (sum(p) * weight)
    return matrix

def from_lowres_grayscale(matrix):
    def lowres(pixels, ij):
        i,j = ij
        return pixels[i // square_width][j // square_width]
    def make_hires(limg):
        pixels = list(matrix)
        triple = lambda x: (x,x,x)
        ### taking the average values vector & expanding to r,g,b...
        return ImageVector([triple(lowres(matrix, ij(n))) for n in
            range(0,total_pixels)])
    return make_hires(matrix)

## make an image class object from raw image file
img_data = ImageVector(outside_img)
## run lowres function on image object
small_img_data = from_lowres_grayscale(to_lowres_grayscale(img_data))
## create new image from png representation of image object
small_img = Image.open(io.BytesIO(small_img_data._repr_png_()))
## save file to system
save_img_file = 'mfp_static_content/small_' + stamp + '.jpg'
small_save_path = os.path.abspath(save_img_file)
small_img.save(small_save_path)

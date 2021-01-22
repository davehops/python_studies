from mfp_modules import colors, vectors, vector_drawing, draw2d, draw3d, draw_model, transforms
from mfp_models import teapot
from vector_arrays import dino_vectors
from random import uniform
from math import sqrt, pi, cos, sin
import matplotlib
from numpy import random
from hypothesis import given, example
import hypothesis.strategies as st

def polygon_map(transformation, polygons):
    return [
        [transformation(vertex) for vertex in triangle]
        for triangle in polygons
    ]

# exercise 4.1, 4.2
def translate_by(translation):
   def new_function(v):
      return vectors.add(translation,v)
   return new_function

# draw_model.draw_model(polygon_map(translate_by((0,-1,-1)), teapot.load_triangles()))
# exercise 4.3
# draw_model.draw_model(polygon_map(transforms.scale_by(-1), teapot.load_triangles()))
# draw_model.draw_model(polygon_map(transforms.rotate_y_by(-15), teapot.load_triangles()))
# exercise 4.4
# draw_model.draw_model(polygon_map(transforms.compose(transforms.rotate_y_by(-15), transforms.rotate_x_by(-15)), teapot.load_triangles()))
# exercise 4.5
# draw_model.draw_model(polygon_map(compose(
#     transforms.scale_by(0.6), 
#     transforms.scale_by(1.5),
#     transforms.rotate_y_by(-15), 
#     transforms.rotate_x_by(-15)
#     ), teapot.load_triangles()))

# exercise 4.6
def compose(*args):
    def new_function(input):
        result = input
        for f in reversed(args):
            result = f(result)
        return result
    return new_function

def prepend(str):
    def new_function(input):
        return str + input
    return new_function
# f = compose(prepend('p'), prepend('y'), prepend('t'), prepend('h'), prepend('h'))
# print(f('hon'))

# exercise 4.7
def curry_two(f):
    def g(x):
        def new_function(y):
            return f(x,y)
        return new_function
    return g
scale_curry = curry_two(vectors.scale)
# print(scale_curry(4)((4,6,8,10)))

def stretch_x(scalar,vector):
    x,y,z = vector
    return (scalar*x,y,z)

def curry_stretch_x(f):
    def new_function(vector):
        x,y,z = vector
        return (f*x,y,z)
    return new_function
# print(stretch_x(2,(4,6,8)))
# print(curry_stretch_x(2)((4,6,8)))

# exercise 4.14
linear = []
zero_change = []
def square_vector(xy):
    x,y = xy
    return (x**2,y**2)
def nothing(xy):
    return xy
def plot(array,func):
    for x in range(0,6):
        for y in range(0,6):
            array.append(func((x,y)))

def curry_plot(arr):
    def new_function(func):
        return plot(arr,func)
    return new_function

curry_plot(zero_change)(nothing)
curry_plot(linear)(square_vector)
# plot(zero_change,nothing)
# plot(nonlinear,square_vector)
# vector_drawing.draw(
#    vector_drawing.Points(*zero_change, color=colors.gray),
#    vector_drawing.Points(*linear, color=colors.blue)
# )

# mini-project 4.15
# rand_vectors = []
# for i in range(0,20):
#     x = random.randint(20)
#     y = random.randint(20)
#     rand_vectors.append((x,y))
@given(st.integers(), st.integers())
def test_linearity_fail(x,y):
    print(str((x + y)**2) + ' does not always equal ' + str(y**2 + x**2))
    assert (x + y)**2 != y**2 + x**2
@given(st.integers(), st.integers())
def test_transform_linearity(x,y):
    vector = (2,2)
    func = transforms.translate_by
    print( str(func(vector)((x,y))) + ' === (' + str(x+vector[0]) + ',' + str(y+vector[1]) + ')' )
    assert func(vector)((x,y)) == (x+vector[0], y+vector[1])
@given(st.integers(), st.integers())
def test_linearity_success(x,y):
    print(str((x + y)*2) + ' equals ' + str(y*2 + x*2))
    assert (x + y)*2 == y*2 + x*2
# test_linearity_fail()
# test_linearity_success()
test_transform_linearity()


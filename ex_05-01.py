from mfp_modules import colors, vectors, vector_drawing, draw2d, draw3d, draw_model, transforms
from mfp_models import teapot
from vector_arrays import dino_vectors
from random import uniform, randint
from math import sqrt, pi, cos, sin
import matplotlib
from numpy import random
from hypothesis import given, example
import hypothesis.strategies as st
from ex_04_01 import curry_two

def infer_matrix(n,transformation):
    def standard_basis_vector(i):
        return tuple(1 if i==j else 0 for j in range(1,n+1))
    standard_basis = [standard_basis_vector(i) for i in range(1,n+1)]
    # print(standard_basis)
    # ((1,0,0), (0,1,0), (-0,0,1)) >> standard basis vectors
    cols = [transformation(v) for v in standard_basis]
    return tuple(zip(*cols))
matrix = infer_matrix(3,transforms.rotate_y_by(pi/2))
# print(matrix)

def random_matrix(rows,cols,min=-2,max=2):
    return tuple(
        tuple(
            randint(min,max) for j in range(0,cols)
        ) for i in range(0,rows)
    )

def matrix_multiply(a,b):
    return tuple(
        tuple(vectors.dot(row,col) for col in zip(*b))
        for row in a
    )

vec_a = random_matrix(3,3,2,9)
vec_b = random_matrix(3,3,2,9)
print(vec_a)
print(vec_b)
print(matrix_multiply(vec_b,vec_a))
print(matrix_multiply(vec_a,vec_b))
vec_zip = [(6, 5, 3), (8, 3, 5), (4, 6, 7)]
# print(list(zip(*vec_zip)))
vec_c = [(6,8,4),(6,9,6),(7,8,6)]
# print(matrix_multiply(vec_zip,vec_c))
vec_d = [(6,8),(6,9)]
vec_e = [(6, 5), (8, 3)]
# print(matrix_multiply(vec_d,vec_e))
# print(matrix_multiply(vec_e,vec_d))

## exercise 5.6
def transform(v):
    m = ((2,1,1),(1,2,1),(1,1,2))
    return transforms.multiply_matrix_vector(m,v)
# draw_model.draw_model(transforms.polygon_map(transform, teapot.load_triangles()))

## exercise 5.7,8
vec_d = [(6,8),(6,9)]
vec_f = (6, 5)
def multiply_matrix_vector_by_sum(matrix,vector):
    return tuple(
        sum(vector_entry * matrix_entry
        for vector_entry, matrix_entry in zip(row,vector))
        for row in matrix
    )
def multiply_matrix_vector_by_dot_product(matr,vec):
    return tuple(
        vectors.dot(row,vec)
        for row in matr
    )
# print(multiply_matrix_vector_by_dot_product(vec_d,vec_f))
#dot_product produces(76,81)
# print(multiply_matrix_vector_by_sum(vec_d,vec_f))
#sum produces(76,81)

## exercise 5.10
a = ((1,1,0),(1,0,1),(1,-1,1))
b = ((0,2,1),(0,1,0),(1,0,-1))
def transform_a(v):
    return multiply_matrix_vector_by_sum(a,v)
def transform_b(v):
    return multiply_matrix_vector_by_sum(b,v)
compose_a_b = transforms.compose(transform_a,transform_b)
# print(infer_matrix(3,compose_a_b))

## exercise 5.11
def round_matrix(matrix):
    return tuple(
        tuple(
            round(x,0) for x in row
        ) for row in matrix
    )
# print(round_matrix(matrix))
matrix2 = [transforms.rotate_y_by(3 * (pi/2))(row) for row in matrix]
matrix3 = [transforms.rotate_y_by(pi/2)(row) for row in matrix]
# print(round_matrix(matrix))
# ((0.0, 0.0, -1.0), (0, 1, 0), (1.0, 0.0, 0.0))  >> start from identity matrix 90deg
# print(round_matrix(matrix2))
# ((-1.0, 0.0, 0.0), (-0.0, 1, -0.0), (-0.0, 0.0, -1.0)) >> start from identity 270deg
# print(round_matrix(matrix3))
# ((1.0, 0.0, 0.0), (0.0, 1, 0.0), (-0.0, 0.0, 1.0)) >> start from 270, +90 back to 360 / 0
# print(infer_matrix(2,transforms.rotate_y_by(pi/2)))

## exercise 5.12
def matrix_power(power,matrix):
    result = matrix
    for _ in range(1,power):
        result = matrix_multiply(result,matrix)
    return result
power_mtrx = ((2,1,1),(1,2,1),(1,1,2))
std_bs = ((1,0,0), (0,1,0), (-0,0,1))
# print(matrix_power(3,power_mtrx))

## exercise 5.17
mtrx_5_3 = ((1,6,3),(1,2,1),(3,2,7),(4,3,0),(1,3,1))
mtrx_2_3 = ((2,1),(3,2),(1,1))
# print(list(zip(*mtrx_2_3)))
# print(matrix_multiply(mtrx_5_3,mtrx_2_3))

## exercise 5.18
def transpose(matrix):
    return tuple(zip(*matrix))
# print(transpose(mtrx_2_3))

## exercise 5.20

mtrx_3_5 = ((1,6,3),(1,2,1),(3,2,7),(4,3,0),(1,3,1))
mtrx_2_3 = ((2, 3, 1), (1, 2, 1))
def matrix_multiply_without_zip(a,b):
    return tuple(
        tuple(vectors.dot(row,col) for col in b)
        for row in a
    )
# print(matrix_multiply_without_zip(mtrx_2_3,mtrx_3_5))

## exercise 5.24
lemons = ((1,),(2,),(3,),(4,),(5,),(6,))
std_basis_6d = (
    (1,0,0,0,0,0),
    (0,1,0,0,0,0),
    (0,0,1,0,0,0),
    (0,0,0,1,0,0),
    (0,0,0,0,1,0),
    (0,0,0,0,0,1)
)
solemn_transform = (
    (0,0,0,0,0,1),
    (0,0,0,1,0,0),
    (1,0,0,0,0,0),
    (0,1,0,0,0,0),
    (0,0,1,0,0,0),
    (0,0,0,0,1,0)
)
# print(matrix_multiply(solemn_transform,lemons))
### prints ((6,), (4,), (1,), (2,), (3,), (5,))
# print(matrix_multiply(std_basis_6d,lemons))

## exercise 5.26 >> combining vector transformations
dino_magic_transform = (
    (1,0,2),
    (0,1,2),
    (0,0,1)
)
#> map vector array in 3D space
dino_3d = [(x,y,1) for x,y in dino_vectors.trex]
#> translate vector
translated_dino = [multiply_matrix_vector_by_dot_product(dino_magic_transform, v) for v in dino_3d]
#> map vector back into 2D space
translated_2d = [(x,y) for (x,y,z) in translated_dino]
# draw2d.draw2d(
#     draw2d.Points2D(*translated_2d),
#     draw2d.Polygon2D(*translated_2d),
#     draw2d.Points2D(*dino_vectors.trex),
#     draw2d.Polygon2D(*dino_vectors.trex)
# )

## exercise 5.29 >> transform and translate with matrix
rotate_45_degrees = curry_two(vectors.rotate2d)(pi/4)
rotation_matrix = infer_matrix(2,rotate_45_degrees)
scale_50_perc = curry_two(vectors.scale)(0.5)
scale_matrix = infer_matrix(2,scale_50_perc)
# print(scale_matrix)
rotate_and_scale = matrix_multiply(scale_matrix,rotation_matrix)
# print(rotate_and_scale)
scale_and_rotate = matrix_multiply(rotation_matrix,scale_matrix)
# print(scale_and_rotate)
((a,b),(c,d)) = rotate_and_scale
final_matrix = ((a,b,-1),(c,d,3),(0,0,1))
# print(final_matrix)

final_dino = [multiply_matrix_vector_by_dot_product(final_matrix, v) for v in dino_3d]

final_to_2d = [(x,y) for (x,y,z) in final_dino for v in final_dino]
# draw2d.draw2d(
#     draw2d.Points2D(*final_to_2d),
#     draw2d.Polygon2D(*final_to_2d),
#     draw2d.Points2D(*dino_vectors.trex),
#     draw2d.Polygon2D(*dino_vectors.trex)
# )

 ## exercise 5.30 >> do transformations in opposite order
final_matrix_no_translation = ((a,b,0),(c,d,0),(0,0,1))
final_dino_reverse = [multiply_matrix_vector_by_dot_product(final_matrix_no_translation,v) for v in dino_3d]
std_basis_3d = infer_matrix(3,transforms.rotate_y_by(0))
translate_matrix = ((1,0,-1),(0,1,3),(0,0,1))
transform_matrix = matrix_multiply(std_basis_3d,translate_matrix)
# print(transform_matrix)
final_dino_reverse_translate = [multiply_matrix_vector_by_dot_product(transform_matrix,v) for v in final_dino_reverse]
final_dino_reverse_translate_to_2d = [(x,y) for (x,y,z) in final_dino_reverse_translate for v in final_dino_reverse_translate]
# draw2d.draw2d(
#     draw2d.Points2D(*final_dino_reverse_translate_to_2d),
#     draw2d.Polygon2D(*final_dino_reverse_translate_to_2d),
#     draw2d.Points2D(*final_to_2d),
#     draw2d.Polygon2D(*final_to_2d)
# )

## exercise 5.31
def translate_4d(translation):
    def new_function(target):
        a,b,c,d = translation
        x,y,z,w = target
        matrix = (
            (1,0,0,0,a),
            (0,1,0,0,b),
            (0,0,1,0,c),
            (0,0,0,1,d),
            (0,0,0,0,1),
        )
        vector = (x,y,z,w,1)
        x_out,y_out,z_out,w_out,_ = multiply_matrix_vector_by_dot_product(matrix,vector)
        return (x_out,y_out,z_out,w_out)
    return new_function
# print(translate_4d((1,2,3,4))((10,20,30,40)))

from mfp_modules import colors, vectors, vector_drawing, draw3d, draw2d
from vector_arrays import dino_vectors
from random import uniform
from math import sqrt, pi, cos, sin
import matplotlib

# exercise 3.2
# eight 3d vectors with coordinates all either -1 or +1
manual_plot = [
   (1,1,1),(-1,1,1),(1,-1,1),(1,1,-1),
   (-1,-1,1),(1,-1,-1),(-1,1,-1),(-1,-1,-1)
]
rng = [60,-15]
def coord_options(set):
   vertices = [(x,y,z) for x in set for y in set for z in set]
   edges = [ ((set[1],y,z),(set[0],y,z)) for y in set for z in set ] +\
      [ ((x,y,set[1]),(x,y,set[0])) for x in set for y in set ] +\
      [ ((x,set[1],z),(x,set[0],z)) for x in set for z in set ]
   return [vertices, edges]
one_one_box = coord_options(rng)
# print(one_one_box[0])
# draw3d.draw3d(
#    draw3d.Points3D(*one_one_box[0], color=colors.orange),
#    *[draw3d.Segment3D(*edge) for edge in one_one_box[1]]
# )

## long way for adding vectors from chapter 3 text example
## zip extracts each dimension from the vector
## tuple reassembles the vectors from zip
def add_vectors(*vectors):
    by_coordinate = zip(*vectors)
    coordinate_sums = [sum(coords) for coords in by_coordinate]
    return tuple(coordinate_sums)
two_vectors = [
   (1,1,9),(-1,9,1)
]
# print(add_vectors(*two_vectors))

# exercise 3.3
ex3_3_vectors = [
   (8,0,6),(-2,0,2)
]
ex3_3_sum = add_vectors(*ex3_3_vectors)
# draw3d.draw3d(
#    draw3d.Arrow3D(ex3_3_vectors[0], color=colors.red),
#    draw3d.Arrow3D(ex3_3_vectors[1], color=colors.blue),
#    draw3d.Arrow3D(ex3_3_sum, ex3_3_vectors[0], color=colors.blue),
#    draw3d.Arrow3D(ex3_3_vectors[1], ex3_3_sum, color=colors.red),
#    draw3d.Arrow3D(ex3_3_sum, color=colors.black),
# )
vs = [(sin(pi*t/6), cos(pi*t/6), 3.0/3) for t in range(0,120)]
running_sum = (0,0,0)
arrows = []
for v in vs:
       next_sum = add_vectors(running_sum, v)
       arrows.append(draw3d.Arrow3D(next_sum, running_sum))
       running_sum = next_sum
# print(running_sum)
# draw3d.draw3d(*arrows)

## exercise 3.6
## normal return is in array form, tuple formats
## as an ordered pair
def vector_scale(scalar,vector):
   return [ scalar * v for v in vector ]
# print( tuple(vector_scale(5,(5,5,5))) )

## exercise 3.9
perf_squares = []
def find_wn_lengths(max_coord=100):
   for x in range(1,max_coord): 
      for y in range(1,x+1):
         for z in range(1,y+1):
            if vectors.length((x,y,z)).is_integer():
                   perf_squares.append((x,y,z))
# find_wn_lengths(250)
# print(len(perf_squares))
# print(perf_squares[len(perf_squares)-1])

## exercise 3.10
## find a vector in the same direction as (-1,-1,2) but which has length 1
## scale vector multiplies original vector by percentage we want it to change
# print(vectors.length((-1,-1,2)))
# print(vectors.scale(1/vectors.length((-1,-1,2)),(-1,-1,2)))

## exercise 3.22 - 25
# print(vectors.cross((1,-2,1),(-6,12,-6)))
# print(vectors.cross((0,0,3),(0,-2,0)))
# print(vectors.cross((1,0,1),(-1,0,0)))
# print((1 * 0)-(1 * -1))

## exercise 3.27
# xy_plane = [(1,0,0),(0,1,0),(-1,0,0),(0,-1,0)]
# edges = [draw3d.Segment3D(top,p) for p in xy_plane] +\
#    [draw3d.Segment3D(bottom,p) for p in xy_plane] +\
#       [draw3d.Segment3D(xy_plane[i],xy_plane[(i+1)%4]) for i in range(0,4)]
# draw3d.draw3d(*edges)

## 2d render example section 3.5.2,
top = (1,0,0)
bottom = (-1,0,0)
xy_plane = [(0,0,-1),(0,1,0),(0,0,1),(0,-1,0)]
face_0 = []
for p in range(0,len(xy_plane)):
   if p == len(xy_plane)-1:
      face_0.append([(bottom),(xy_plane[p]),(xy_plane[0])])
      face_0.append([(top),(xy_plane[p]),(xy_plane[0])])
   else:
      face_0.append([(bottom),(xy_plane[p]),(xy_plane[p+1])])
      face_0.append([(top),(xy_plane[p]),(xy_plane[p+1])])
# print(face_0)
octahedron = [
   [(1,0,0),(0,1,0),(0,0,1)],
   [(1,0,0),(0,0,-1),(0,1,0)],
   [(1,0,0),(0,0,1),(0,-1,0)],
   [(1,0,0),(0,-1,0),(0,0,-1)],
   [(-1,0,0),(0,0,1),(0,1,0)],
   [(-1,0,0),(0,1,0),(0,0,-1)],
   [(-1,0,0),(0,-1,0),(0,0,1)],
   [(-1,0,0),(0,0,-1),(0,-1,0)],
]
def vector_to_2d(v):
   return (vectors.component(v,(0,0,1)), vectors.component(v,(0,1,0)))
def face_to_2d(face):
   return [vector_to_2d(vertex) for vertex in face]
def unit(v):
   return vectors.scale(1./vectors.length(v),v)
def normal(face):
   return(vectors.cross(vectors.subtract(face[1],face[0]),vectors.subtract(face[2],face[0])))
blues = matplotlib.cm.get_cmap('Blues')
def render(faces, light=(1,2,3), color_map=blues, lines=None):
   polygons = []
   for face in faces:
      unit_normal = unit(normal(face))
      if unit_normal[2] > 0:
         c = color_map(1 - vectors.dot(unit(normal(face)), 
            unit(light)))
         p = draw2d.Polygon2D(*face_to_2d(face),
            fill=c, color=lines)
         polygons.append(p)
   draw2d.draw2d(*polygons,axes=False,origin=False, grid=None)
# render(face_0, color_map=blues, lines=colors.gray)
from mfp_modules import colors, vectors, vector_drawing
from vector_arrays import dino_vectors
from random import uniform
from math import sqrt, pi, cos, sin

# exercise 2.19
# u = (-1,1)
# v = (1,1)
# def random_r():
#        return uniform(-3,3)
# def random_s():
#        return uniform(-1,1)
# possibilities = [vectors.add(vectors.scale(random_r(), u), vectors.scale(random_s(), v))
#        for i in range(0,100)]
# vector_drawing.draw(
#    vector_drawing.Points(*possibilities)
# )

# exercise 2.24
# def vector_subtract(v1,v2):
#    return (sum([v1[0], (v2[0] * -1)]), sum([v1[1], (v2[1] * -1)]))
# print(vector_subtract((-2,0), (1.5,1.5)))

def vector_subtract(v1,v2):
   return(v1[0] - v2[0], v1[1] - v2[1])  
   # return (sum([v1[0], (v2[0] * -1)]), sum([v1[1], (v2[1] * -1)]))
u = (-2,0)
v = (1.5,1.5)
w = (4,1)
x = (0,0)
y = (3,4)
assignment = [(v,w),(u,v),(w,v)]
# homework = [print(vector_subtract(i[0],i[1])) for i in assignment]

# exercise 2.25
def vector_distance_pyth(v1,v2):
   disp = vector_subtract(v1,v2)
   return(sqrt(disp[0]**2 + disp[1]**2))
def vector_distance_len(v1,v2):
   return(vectors.length(vector_subtract(v1,v2)))
# print(vector_distance_pyth(x,y))
# print(vector_distance_len(x,y))

def perimeter(vectors):
   distances = [vector_distance_pyth(vectors[i], vectors[(i+1)%len(vectors)])
               for i in range(0,len(vectors))]
   return sum(distances)
# print(perimeter([(1,0),(1,1),(0,1),(0,0)]))
# print(perimeter(dino_vectors.trex))

answers = [(1,-1)]
for n in range(-40,20):
       for m in range(-20,20):
              if vector_distance_pyth((n,m),answers[0]) == 13 and n > m:
                     answers.append((n,m))
# vector_drawing.draw(
#    vector_drawing.Points(*answers)
# )

# exercise 2.35
def to_rad(angle):
       return angle * (1 / 57.296)
# print(to_rad(116.57))

# exercise 2.36
# print('sine of 10pi/6 = ' + str(sin(10*pi/6)))
# print('cosine of 10pi/6 = ' + str(cos(10*pi/6)))

# exercise 2.37
polar_coords = [(cos(5*x*pi/500.0), 2*pi*x/1000.0) for x in range (0,1000)]
# vector_drawing.draw(
#    vector_drawing.Points(*polar_coords)
# )
# vector_coords = [vectors.to_cartesian(p) for p in polar_coords]
# vector_drawing.draw(
#    vector_drawing.Polygon(*vector_coords, color=colors.gray)
# )
# vector_drawing.Polygon(*polar_coords, color=colors.orange)

# exercise 2.42
def rotate(angle, vect):
   polars = [vectors.to_polar(v) for v in vect]
   return [vectors.to_cartesian((l, a+angle)) for l,a in polars]

new_dino = rotate(to_rad(-71.6), dino_vectors.trex)
# vector_drawing.draw(
#    vector_drawing.Polygon(*new_dino, color=colors.orange)
# )   

# exercise 2.43
def regular_polygon(n,r):
   return [vectors.to_cartesian((r, 2*pi*k/n)) for k in range(0,n)]
   # return [rotate(360/n, [(1,1])) for i in range(0,n)]
poly = regular_polygon(36,2)
poly2 = regular_polygon(18,1)
poly3 = regular_polygon(72,3)
# print(poly)
vector_drawing.draw(
   vector_drawing.Polygon(*poly, color=colors.blue),
   vector_drawing.Points(*poly, color=colors.red),
   vector_drawing.Polygon(*poly2, color=colors.blue),
   vector_drawing.Points(*poly2, color=colors.red),
   vector_drawing.Polygon(*poly3, color=colors.blue),
   vector_drawing.Points(*poly3, color=colors.red)
)
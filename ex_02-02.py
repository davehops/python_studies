from mfp_modules import vectors, vector_drawing
from vector_arrays import dino_vectors
from random import uniform
from math import sqrt

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
vector_drawing.draw(
   vector_drawing.Points(*answers)
)

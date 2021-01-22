from mfp_modules import vectors, vector_drawing
from vector_arrays import dino_vectors

dino_vectors2 = [vectors.add((-1.5,-2.5),v) for v in dino_vectors.trex]

# exercise 2.4
vector_drawing.draw(
   vector_drawing.Points(*dino_vectors.trex),
   vector_drawing.Polygon(*dino_vectors.trex, color=vector_drawing.orange)
)

# exercise 2.5 render here:
# draw(
#    Points(*[(x,x**2) for x in range(-10,11)]),
#    grid=(1,10),
#    nice_aspect_ratio=False
# )

# exercise 2.7
# def addy(*vectors):
#     return (sum([v[0] for v in vectors]), sum(v[1] for v in vectors))
# print(addy(*dino_vectors.trex))
# print(addy(*dino_vectors2))
# sm_arr = [(1,2),(2,4),(3,6),(4,8)]
# print(addy(*sm_arr)) # returns (10,20)

# exercise 2.8
# def translate(translation, vect):
#     return [vectors.add(translation, v) for v in vect]
# print(dino_vectors.trex
# print(translate((-12,-5), dino_vectors.trex)    

# exercise 2.11
# def hundred_dinos(rg,colr,vects):
#     translations=[(13*x,11*y)
#          for x in range(-rg,rg)
#          for y in range(-rg,rg)]
#     dinos = [Polygon(*translate(t,vects),color=colr)
#          for t in translations]
#     draw(*dinos, grid=None, axes=None, origin=None)
# hundred_dinos(6,'red',dino_vectors.trex)
# hundred_dinos(3,'blue',dino_vectors2)

# exercise 2.15
# max function >> param 1 = array, key = function for calculation
# print(max(dino_vectors.trex, key=vectors.length))

# figure 2.13
# draw(
#     Points(*dino_vectors.trex, color=blue),
#     Polygon(*dino_vectors.trex, color=blue),
#     Points(*dino_vectors2, color=red),
#     Polygon(*dino_vectors2, color=red)
# )

# figure 2.18
# scalar = (6.5 * 1.2, 6.5 * -3.1)
# print(scalar)

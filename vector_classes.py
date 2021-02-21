from abc import ABCMeta, abstractmethod, abstractproperty
from mfp_modules import vectors
from datetime import datetime

class Vector(metaclass=ABCMeta):
    @abstractmethod
    def scale(self,scalar):
        pass
    @abstractmethod
    def add(self,other):
        pass
    def subtract(self,other):
        return self.add(-1 * other)
    @classmethod
    @abstractproperty
    def zero():
        pass
    def __add__(self,other):
        return self.add(other)
    def __sub__(self,other):
        return self.subtract(other)
    def __mult__(self,scalar):
        return self.scale(scalar)
    def __rmul__(self,scalar):
        return self.scale(scalar)
    def __truediv__(self,scalar):
        return self.scale(1.0/scalar)
    def __neg__(self):
        return self.scale(-1)

class Vec0(Vector):
    def __init__(self):
        pass
    def add(self, other):
        return Vec0()
    def scale(self, scalar):
        return Vec0()
    @classmethod
    def zero(cls):
        return Vec0()
    def __eq__(self,other):
        return self.__class__ == other.__class__ == Vec0
    def __repr__(self):
        return "Vec0"

class Vec2(Vector):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def zero():
        return Vec2(0,0)
    def add(self, other):
        assert self.__class__ == other.__class__
        return Vec2(self.x + other.x, self.y + other.y)
    def scale(self, scalar):
        return Vec2(scalar * self.x, scalar * self.y)
    def __eq__(self,other):
        return (self.__class__ == other.__class__ 
            and self.x == other.x and self.y == other.y)
    def __repr__(self):
        return "Vec2({},{})".format(self.x,self.y)

class Vec3(Vector):
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
    def zero():
        return Vec3(0,0,0)
    def add(self, other):
        assert self.__class__ == other.__class__
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def scale(self, scalar):
        return Vec3(scalar * self.x, scalar * self.y, scalar * self.z)
    def __eq__(self, other):
        return (self.__class__ == other.__class__
            and self.x == other.x and self.y == other.y and self.z == other.z)
    def __repr__(self):
        return "Vec3({},{},{})".format(self.x, self.y, self.z)

class CoordinateVector(Vector):
    @abstractproperty
    def dimension(self):
        pass
    def __init__(self,*coordinates):
        self.coordinates = tuple(x for x in coordinates)
    def add(self,other):
        return self.__class__(*vectors.add(self.coordinates, other.coordinates))
    def scale(self, scalar):
        return self.__class__(*vectors.scale(scalar, self.coordinates))
    def zero():
        pass
    def __repr__(self):
        return "{}{}".format(self.__class__.__qualname__, self.coordinates)

class Vec6(CoordinateVector):
    def dimension(self):
        return 6
    def zero():
        return Vec6(0,0,0,0,0,0)
# print(Vec6.zero())
# print(Vec6(1,2,3,4,5,6) + Vec6(1,2,3,4,5,6))

class Vec1(CoordinateVector):
    def dimension(self):
        return 1
    def zero():
        return Vec1(0)

class Matrix5_by_3(Vector):
    rows = 5
    columns = 3
    def __init__(self, matrix):
        self.matrix = matrix
    def add(self, other):
        return Matrix5_by_3(tuple(
            tuple(a + b for a,b in zip(row1, row2))
            for (row1, row2) in zip(self.matrix, other.matrix)
        ))
    def scale(self,scalar):
        return Matrix5_by_3(tuple(
            tuple(scalar * x for x in row)
            for row in self.matrix
        ))
    @classmethod
    def zero(cls):
        return Matrix5_by_3(tuple(
            tuple(0 for j in range(0, cls.columns))
            for i in range(0, cls.rows)
        ))

class Matrix(Vector):
    @abstractproperty
    def rows(self):
        pass
    @abstractproperty
    def columns(self):
        pass
    def __init__(self, entries):
        self.entries = entries
    def add(self, other):
        return self.__class__(
            tuple(
                tuple(self.entries[i][j] + other.entries[i][j]
                    for j in range(0,self.columns()))
                for i in range(0,self.rows())))
    def scale(self,scalar):
        return self.__class__(
            tuple(
                tuple(scalar * x for x in row)
            for row in self.entries))
    def __repr__(self):
        return "%s%r" % (self.__class__.__qualname__, self.entries)
    @classmethod
    def zero(self):
        return self.__class__(
            tuple(
                tuple(0 for j in range(0, self.columns()))
            for i in range(0, self.rows())))



from PIL import Image
class ImageVector(Vector):
    size = (300,300)
    def __init__(self,input):
        try:
            img = Image.open(input).resize(ImageVector.size)
            self.pixels = img.getdata()
        except:
            self.pixels = input
    def image(self):
        img = Image.new('RGB', ImageVector.size)
        img.putdata([(int(r), int(g), int(b)) 
                     for (r,g,b) in self.pixels])
        return img
    def add(self,img2):
        return ImageVector([(r1+r2,g1+g2,b1+b2) 
                            for ((r1,g1,b1),(r2,g2,b2)) 
                            in zip(self.pixels,img2.pixels)])
    def scale(self,scalar):
        return ImageVector([(scalar*r,scalar*g,scalar*b) 
                      for (r,g,b) in self.pixels])
    @classmethod
    def zero(cls):
        total_pixels = cls.size[0] * cls.size[1]
        return ImageVector([(0,0,0) for _ in range(0,total_pixels)])
    def _repr_png_(self):
        return self.image()._repr_png_() 

# class CarForSale():
#     def __init__(self, model_year, mileage, price, posted_datetime,
#                 model, source, location, description):
#         self.model_year = model_year
#         self.mileage = mileage
#         self.price = price
#         self.posted_datetime = posted_datetime
#         self.model = model
#         self.source = source
#         self.location = location
#         self.description = description

class CarForSale(Vector):
    retrieved_date = datetime(2018,11,30,12)
    def __init__(self, model_year, mileage, price, posted_datetime, 
                 model="(virtual)", source="(virtual)",
                 location="(virtual)", description="(virtual)"):
        self.model_year = model_year
        self.mileage = mileage
        self.price = price
        self.posted_datetime = posted_datetime
        self.model = model
        self.source = source
        self.location = location
        self.description = description
    def add(self, other):
        def add_dates(d1, d2):
            age1 = CarForSale.retrieved_date - d1
            age2 = CarForSale.retrieved_date - d2
            sum_age = age1 + age2
            return CarForSale.retrieved_date - sum_age
        return CarForSale(
            self.model_year + other.model_year,
            self.mileage + other.mileage,
            self.price + other.price,
            add_dates(self.posted_datetime, other.posted_datetime)
        )
    def scale(self,scalar):
        def scale_date(d): #5
            age = CarForSale.retrieved_date - d
            return CarForSale.retrieved_date - (scalar * age)
        return CarForSale(
            scalar * self.model_year,
            scalar * self.mileage,
            scalar * self.price,
            scale_date(self.posted_datetime)
        )
    @classmethod
    def zero(cls):
        return CarForSale(0, 0, 0, CarForSale.retrieved_date)

class Function(Vector):
    def __init__(self,f):
        self.function = f
    def add(self, other):
        return Function(lambda x: self.function(x) + other.function(x))
    def scale(self,scalar):
        return Function(lambda x: scalar * self.function(x))
    @classmethod
    def zero(cls):
        return Function(lambda x: 0)
    def __call__(self, arg):
        return self.function(arg)
    # def __repr__(self):
    #     return "{}".format(self.__str__())

class Function2(Vector):
    def __init__(self,f):
        self.function = f
    def add(self, other):
        return Function2(lambda x,y: self.function(x,y) + other.function(x,y))
    def scale(self,scalar):
        return Function2(lambda x,y: scalar * self.function(x,y))
    @classmethod
    def zero(cls):
        return Function2(lambda x,y: 0)
    def __call__(self, *args):
        return self.function(*args)

class LinearFunction(Vector):
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def add(self, v):
        return LinearFunction(self.a + v.a, self.b + v.b)
    def scale(self,scalar):
        return LinearFunction(scalar * self.a, scalar * self.b)
    def __call__(self, x):
        return self.a * x + self.b
    @classmethod
    def zero(cls):
        return LinearFunction(0,0,0)

class QuadraticFunction(Vector):
    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c
    def add(self, v):
        return QuadraticFunction(self.a + v.a, 
                                 self.b + v.b,
                                 self.c + v.c)
    def scale(self,scalar):
        return QuadraticFunction(scalar * self.a, 
                                 scalar * self.b,
                                 scalar * self.c)
    def __call__(self, x):
        return self.a * x * x + self.b * x + self.c
    @classmethod
    def zero(cls):
        return LinearFunction(0,0,0)

class Polynomial(Vector):
    def __init__(self, *coefficients):
        self.coefficients = coefficients
    def __call__(self,x):
        return sum(coefficient * x ** power
                   for (power,coefficient)
                   in enumerate(self.coefficients))
    def add(self,p):
        return Polynomial([a + b
                        for a,b
                        in zip(self.coefficients,
                                p.coefficients)])
    def scale(self,scalar):
        return Polynomial([scalar * a
                           for a in self.coefficients])
        return "$ %s $" % (" + ".join(monomials))
    @classmethod
    def zero(cls):
        return Polynomial(0)

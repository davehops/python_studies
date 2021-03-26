from abc import ABC, abstractmethod
import math

def paren_if_instance(exp,*args):
    for typ in args:
        if isinstance(exp,typ):
            return "\\left( {} \\right)".format(exp.latex())
    return exp.latex()

def package(maybe_expression):
    if isinstance(maybe_expression,Expression):
        return maybe_expression
    elif isinstance(maybe_expression,int) or isinstance(maybe_expression,float):
        return Number(maybe_expression)
    else:
        raise ValueError("can't convert {} to expression.".format(maybe_expression))
def dot_if_necessary(latex):
    if latex[0] in '-1234567890':
        return '\\cdot {}'.format(latex)
    else:
        return latex

def distinct_variables(exp):
    if isinstance(exp, Variable):
        return set(exp.symbol)
    elif isinstance(exp, Number):
        return set()
    elif isinstance(exp, Sum):
        return set().union(*[distinct_variables(exp) for exp in exp.exps])
    elif isinstance(exp, Product):
        return distinct_variables(exp.exp1).union(distinct_variables(exp.exp2))
    elif isinstance(exp, Power):
        return distinct_variables(exp.base).union(distinct_variables(exp.exponent))
    elif isinstance(exp, Apply):
        return distinct_variables(exp.argument)
    else:
        raise TypeError("Not a valid expression.")

class Expression(ABC):
    # @abstractmethod
    # def latex(self):
    #     pass
    def _repr_latex_(self):
        return "$$" + self.latex() + "$$"
    @abstractmethod
    def evaluate(self, **bindings):
        pass
    # @abstractmethod
    # def substitute(self, var, expression):
    #     pass
    @abstractmethod
    def expand(self):
        pass
    # @abstractmethod
    def display(self):
        pass
    def __repr__(self):
        return self.display()
    # @abstractmethod
    # def derivative(self,var):
    #     pass
    def __call__(self, *inputs):
        var_list = sorted(distinct_variables(self))
        return self.evaluate(**dict(zip(var_list, inputs)))
    def __add__(self, other):
        return Sum(self,package(other))
    def __sub__(self,other):
        return Difference(self,package(other))
    def __mul__(self,other):
        return Product(self,package(other))
    def __rmul__(self,other):
        return Product(package(other),self)
    def __truediv__(self,other):
        return Quotient(self,package(other))
    def __pow__(self,other):
        return Power(self,package(other))
    # @abstractmethod
    # def _python_expr(self):
    #     pass
    def python_function(self,**bindings):
#         code = "lambda {}:{}".format(
#             ", ".join(sorted(distinct_variables(self))),
#             self._python_expr())
#         print(code)
        global_vars = {"math":math}
        return eval(self._python_expr(),global_vars,bindings)

class Power(Expression):
    def __init__(self,base,exponent):
        self.base = base
        self.exponent = exponent
    ## must write expand properly before using
    def expand(self):
        return self
    def display(self):
        return "Power({},{})".format(self.base.display(),self.exponent.display())
    def evaluate(self, **bindings):
        return self.base.evaluate(**bindings) ** self.exponent.evaluate(**bindings)
class Number(Expression):
    def __init__(self,number):
        self.number = number
    def expand(self):
        return self
    def display(self):
        return "Number({})".format(self.number)
    def evaluate(self, **bindings):
        return self.number
class Variable(Expression):
    def __init__(self,symbol):
        self.symbol = symbol
    ## must write expand properly before using
    def expand(self):
        return self
    def evaluate(self, **bindings):
        try:
            return bindings[self.symbol]
        except:
            raise KeyError("Variable '{}' is not bound.".format(self.symbol))
    def display(self):
        return "Variable(\"{}\")".format(self.symbol)
class Sum(Expression):
    def __init__(self,*exps):
        self.exps = exps
    def expand(self):
        return Sum(*[exp.expand() for exp in self.exps])
    def display(self):
        return "Sum({})".format(",".join([e.display() for e in self.exps]))
    def evaluate(self, **bindings):
        return sum([exp.evaluate(**bindings) for exp in self.exps])
class Difference(Expression):
    def __init__(self,exp1,exp2):
        self.exp1 = exp1
        self.exp2 = exp2
    ## must write expand properly before using
    def expand(self):
        expanded1 = self.exp1.expand()
        expanded2 = self.exp2.expand()
        if isinstance(expanded1, Sum):
            return Sum(*[Difference(e,expanded2).expand()
                        for e in expanded1.exps])
        elif isinstance(expanded2, Sum):
            return Sum(*[Difference(expanded1,e).expand()
                        for e in expanded2.exps])
        else:
            return Difference(expanded1,expanded2)
    def display(self):
        return "Difference({},{})".format(self.exp1.display(), self.exp2.display())
class Product(Expression):
    def __init__(self,exp1,exp2):
        self.exp1 = exp1
        self.exp2 = exp2
    def expand(self):
        expanded1 = self.exp1.expand()
        expanded2 = self.exp2.expand()
        if isinstance(expanded1, Sum):
            return Sum(*[Product(e,expanded2).expand()
                        for e in expanded1.exps])
        elif isinstance(expanded2, Sum):
            return Sum(*[Product(expanded1,e).expand()
                        for e in expanded2.exps])
        else:
            return Product(expanded1,expanded2)
    def display(self):
        return "Product({},{})".format(self.exp1.display(),self.exp2.display())
    def evaluate(self, **bindings):
        return self.exp1.evaluate(**bindings) * self.exp2.evaluate(**bindings)
class Quotient(Expression):
    ## must write expand properly before using
    def expand(self):
        return self
    def __init__(self,numerator,denominator):
        self.numerator = numerator
        self.denominator = denominator
class Function():
    def __init__(self,name):
        self.name = name
    def display(self):
        return "Function({})".format(self.name.display())
class Negative(Expression):
    def __init__(self,exp):
        self.exp = exp
    ## must write expand properly before using
    def expand(self):
        return self
    def evaluate(self, **bindings):
        return - self.exp.evaluate(**bindings)
class Apply(Expression):
    def __init__(self,function,argument):
        self.function = function
        self.argument = argument
    def expand(self):
        return Apply(self.function, self.argument.expand())
    def evaluate(self, **bindings):
        return _function_bindings[self.function.name](self.argument.evaluate(**bindings))
    def display(self):
        return "Apply(Function(\"{}\"),{})".format(self.function.name, self.argument.display())
def f_log(y,z):
    return log(y**z)
def f_sqrt(x):
    return sqrt(x)

def contains(exp, var):
    if isinstance(exp, Variable):
        return exp.symbol == var.symbol
    elif isinstance(exp, Number):
        return False
    elif isinstance(exp, Sum):
        return any([contains(e,var) for e in exp.exps])
    elif isinstance(exp, Product):
        return contains(exp.exp1,var) or contains(exp.exp2,var)
    elif isinstance(exp, Power):
        return contains(exp.base,var) or contains(exp.exponent,var)
    elif isinstance(exp, Apply):
        return contains(exp.argument,var)
    else:
        raise TypeError("Not a valid expression")

def distinct_functions(exp):
    if isinstance(exp, Variable):
        return set()
    elif isinstance(exp, Number):
        return set()
    elif isinstance(exp, Sum):
        return set().union(*[distinct_functions(exp) for exp in exp.exps])
    elif isinstance(exp, Product):
        return distinct_functions(exp.exp1).union(distinct_functions(exp.exp2))
    elif isinstance(exp, Power):
        return distinct_functions(exp.base).union(distinct_functions(exp.exponent))
    elif isinstance(exp, Apply):
        return set([exp.function.name]).union(distinct_functions(exp.argument))
    else:
        raise TypeError("Not a valid expression.")

def contains_sum(exp):
    if isinstance(exp, Variable):
        return False
    elif isinstance(exp, Number):
        return False
    elif isinstance(exp, Sum):
        return True
    elif isinstance(exp, Product):
        return contains_sum(exp.exp1) or contains_sum(exp.exp2)
    elif isinstance(exp, Power):
        return contains_sum(exp.base) or contains_sum(exp.exponent)
    elif isinstance(exp, Apply):
        return contains_sum(exp.argument)
    else:
        raise TypeError("Not a valid expression.")


_function_bindings = {
    "sin": math.sin,
    "cos": math.cos,
    "ln": math.log,
    "sqrt": math.sqrt
}

f_expression = Product(
                Sum(
                    Product(
                        Number(3),
                        Power(
                            Variable("x"),
                            Number(2))),
                    Variable("x")),
                Apply(
                    Function("sin"),
                    Variable("x")))

var_x = Variable("y")
# print(contains(f_expression,var_x))

# print(f_expression.evaluate(x=5))

# Apply(Function("f_log"), Power(Variable("y"), Variable("z")))
# Difference(
#     Power(Variable('b'),Number(2)),
#     Product(Number(4),Product(Variable('a'),Variable('c')))
# )
### -(x^2 + y)
# Negative(Sum(Power(Variable('x'),Number(2)),Variable('y')))
### quadratic formula abstracted
# A = Variable('a')
# B = Variable('b')
# C = Variable('c')
# Sqrt = Function('f_sqrt')

# Quotient(
#     Sum(
#         Negative(B),
#         Apply(
#             Sqrt,
#             Difference(
#                 Power(B,Number(2)),
#                 Product(Number(4),Product(A,C))
#             )
#         )
#     ),
#     Product(Number(2), A)
# )

# def f(x):
#     return (3*x**2 + x) * sin(x)

# print(f(2))

# print(Product(Variable("x"), Variable("y")).evaluate(x=2,y=5))

Y = Variable('y')
Z = Variable('z')
A = Variable('a')
B = Variable('b')

# print(Product(Sum(A,B),Sum(Y,Z)))
# print(Product(Sum(A,B),Sum(Y,Z)).expand())
# print(f_expression.expand())

from sympy import *
from sympy.core.core import *

x = Symbol('x')
y = Symbol('y')
# print(y*(3+x).subs(x,5))
# print((3*x**2).integrate(x))
print(Integer(0).integrate(x))
print((x*cos(x)).integrate(x))
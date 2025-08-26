import numpy as np
import matplotlib.pyplot as plt
import math as m 

class Polynomials:


  """ The Polynomials class takes n inputs of type int that correspond to the coefficients of a polynomial of nth degree.
  The class has attributes:
  - coeffs: returning a list of the polynomials coefficients
  The class has methods:
  - degree: returning the degree of the polynomial
  - eval: evaluating the polynomial at a value and returning a scalar
  - horner: evaluates the polynomial using the Horner Method
  - __str__: returns the polynomial as a string
  - __neg__: negates the coefficients of the polynomial
  - plot: plots the polynomial over a range of x values (default is (-10, 10))
  - __add__: adds two polynomials or a polynomial and a scalar
  - __radd__: allows the argumets to be input in the opposite order as __add__
  - __sub__: subtracts two polynomials or a polynomial and a scalar
  - __rsub__: allows the arguments to be input in the opposite order as __sub__
  - __mul__: multiplies two polynomials or a polynomial and a scalar
  - __rmul__: allows the arguments to be input in the opposite order as __mul__
  - deriv: computes the derivative of the polynomial
  - integral: computes the indefinite integral of a polynomial
  - __getitem__: calls the coefficient of the polynomial at a specific index
  - __setitem__: assigns the value of the polynomials coefficient at a specific index"""

  def __init__(self, *args):
        # Initializaion method
        self.coeffs = [arg for arg in args]

  def degree(self):
        # Returns degree of polynomial
        return len(self.coeffs)-1

  def eval(self,x):
        # Evaluates the polynomial at a specific value x
        res = 0
        for n, a_n in enumerate(self.coeffs):

            res += a_n * x**n

        return res

  def horner(self,x):
        # Uses Horner's method to evaluate the polynomial in a more efficient way

        n = self.degree()

        res = self.coeffs[n]

        for i in range(n, 0,-1):

            res = x*res + self.coeffs[i-1]

        return res

  def __str__(self):
        # String method to print the polynomial

        str = fr'{self.coeffs[0]}'

        hashmap = {True: '+', False: '-'} # Hashmap to determine the sign

        for n, coeff in enumerate(self.coeffs):

            if n == 0:
                continue

            if abs(coeff) < 1e-5:
                coeff = 0

            str += fr" {hashmap[coeff >= 0]} {abs(coeff)}x^{{{n}}}" if coeff != 0 else ''

        return str

  def __neg__(self):

        # Usage of List Comprehension, Unpacking and the initialization method to obtain the negated polynomial

        return Polynomials(*[-coeff for coeff in self.coeffs])

  def plot(self,x_min = -10, x_max = 10):
        # Plots the polynomial at a range of x values. By default plots from -10 to 10

        xs = np.linspace(x_min,x_max)
        ys = np.array([self.eval(x) for x in xs])

        plt.plot(xs,ys, c = 'b', label = str(self))
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc = 'best')

        plt.show()

  def __add__(self, poly):
        
        # Add a polynomial or a scalar

        # For a scalar
        if isinstance(poly, (int, float)):
            coeffs_new = self.coeffs[:]
            coeffs_new[0] += poly
            return Polynomials(*coeffs_new)
        
        # For a polynomial
        elif isinstance(poly, type(self)):
            max_deg = max(self.degree(), poly.degree())
            coeffs_new = [0] * (max_deg + 1)

            for i in range(len(coeffs_new)):
                coeff1 = self.coeffs[i] if i < len(self.coeffs) else 0
                coeff2 = poly.coeffs[i] if i < len(poly.coeffs) else 0
                coeffs_new[i] = coeff1 + coeff2

            return Polynomials(*coeffs_new)

    # Raise an error for unsupported types
        else:
            raise TypeError("Unsupported type for addition")


  def __radd__(self, poly):

      # Allows the argumets to be input in the opposite order as __add__

        sum = self.__add__(poly)
        return Polynomials(*[sum])

  def __sub__(self, poly):
    # Subtracts a polynomial or a scalar

    # For a scalar
    if isinstance(poly, (int, float)):  
        coeffs_new = self.coeffs[:]  
        coeffs_new[0] -= poly  
    
    # For a polynomial
    elif isinstance(poly, type(self)):  
        min_deg = min(self.degree(), poly.degree())
        coeffs_new = []

        for i in range(min_deg + 1):
            coeffs_new.append(self.coeffs[i] - poly.coeffs[i])

        # In case a polynomial is of larger degree
        if self.degree() > poly.degree():
            coeffs_new.extend(self.coeffs[min_deg + 1 :]) 
        else:
            coeffs_new.extend(-c for c in poly.coeffs[min_deg + 1 :]) 
    else:
        raise TypeError("Unsupported type for subtraction. Must be a scalar or a Polynomials instance.")

    return Polynomials(*coeffs_new)

  def __rsub__(self, poly):

      # Allows the arguments to be input in the opposite order as __sub__

        diff = self.__sub__(poly)
        return Polynomials(*[diff])

  def __mul__(self, poly):
    # Multiplies by a scalar or a polynomial

        # For a scalar
        if isinstance(poly, (int, float)):
            new_coeffs = [c * poly for c in self.coeffs]
            return Polynomials(*new_coeffs)

        # For a polynomial
        elif isinstance(poly, type(self)):
            max_deg = self.degree() + poly.degree()  
            new_coeffs = [0] * (max_deg + 1) 

            for i, a in enumerate(self.coeffs):
                for j, b in enumerate(poly.coeffs):
                    new_coeffs[i + j] += a * b  

            return Polynomials(*new_coeffs)

    # Raise an error for unsupported types
        else:
            raise TypeError("Unsupported type for multiplication")

  def __rmul__(self, poly):

      # Multiples when inputs are in the oppoite order as __mul__

        mul = self.__mul__(poly)
        return mul

  def deriv(self):
    # Derives the polynomial by applying the power rule

    deriv = [0] * self.degree()
    for i in range(1, self.degree() + 1):  
        deriv[i - 1] = self.coeffs[i] * i  
    return Polynomials(*deriv)

  def integral(self, c=0):
    # Integrates the polynomial by applying the power rule

    integral = [0] * (self.degree() + 2) 
    integral[0] = c  # Set the constant of integration

    for i in range(self.degree() + 1):
        integral[i + 1] = self.coeffs[i] / (i + 1)

    return Polynomials(*integral)


  def __getitem__(self, key):
        # Raise index error for key outside of index
        try:
            value = self.coeffs[key]
            return Polynomials([value])
        except:
            if key > self.degree() or key < 0: # If key is negetive or larger than the max degree of the object
                raise ValueError('Index out of range')

  def __setitem__(self, key, value):
      # assigns a value to the polynomial at a specific index
        if key >= self.degree():
            for i in range (self.degree() + 1, key + 1):
                self.coeffs.append(0)
        self.coeffs[key] = value
        return Polynomials(self.coeffs[key])

  def Laguerre_Polynomials(n, alpha):
        # Obtains the generalized Laguerre Polynomial of degree n, alpha
        coeffs = []
        for i in range(n+1):
          val = (-1) ** i * m.comb(n + alpha, n - i) * (1/m.factorial(i))
          coeffs.append(val)
        
        return Polynomials(*coeffs)
  
# First method - Degree

p = Polynomials(1, 2, 3, 4)
print(p.degree())
print('Expected degree is 3')
print(p.degree() == 3)

# Second method - Evaluation

p = Polynomials(1, 2, 3, 4)
print(p.eval(2))
print(p.horner(2))

# Third method - String

p = Polynomials(1, 2, 3, 4)
print(str(p))

# Fourth method - Negative

p = Polynomials(1, 2, 3, 4)
print(-p)

# Fifth method - Plotting  

p = Polynomials(1, 2, 3, 4)
p.plot()    
p.plot(-1,1)   

# Fifth method - Addition

p = Polynomials(1, 2, 3, 4)
q = Polynomials(2, 3, 4, 5)
print(p + q)
print(p + 1)
print(1 + p)

# Sixth method - Subtraction

p = Polynomials(1, 2, 3, 4)
q = Polynomials(2, 3, 4, 5)
print(p - q)
print(p - 1)
print(1 - p) 

# Seventh method - Multiplication

p = Polynomials(1, 2, 3, 4)
q = Polynomials(2, 3, 4, 5)
print(p * q)
print(p * 2)
print(2 * p)

# Eighth method - Derivative

p = Polynomials(1, 2, 3, 4)
print(p.deriv())

# Ninth method - Integration

p = Polynomials(1, 2, 3, 4)
print(p.integral())

#  Tenth method - Get Item

p = Polynomials(1, 2, 3, 4)
print(p[2])

# Eleventh method - Set Item

p = Polynomials(1, 2, 3, 4)
p[2] = 5
print(p)

# Twelfth method - Laguerre_Polynomials

p = Polynomials.Laguerre_Polynomials(3,5)
print(p)

# Proof 1 

p=(5*Polynomials.Laguerre_Polynomials(5,3+1)) - (5*Polynomials.Laguerre_Polynomials(5-1,3+1)) + Polynomials.Laguerre_Polynomials(5-1,3+1)*Polynomials(0,1) - ((5+3)*Polynomials.Laguerre_Polynomials(5-1,3))
print(p)

# Proof 2

p= (Polynomials(0,1) * Polynomials.Laguerre_Polynomials(5,3).deriv().deriv() ) + ( 4 * Polynomials.Laguerre_Polynomials(5,3).deriv() ) - ((Polynomials(0,1) * Polynomials.Laguerre_Polynomials(5,3).deriv())) + (5*Polynomials.Laguerre_Polynomials(5,3))
print(p)





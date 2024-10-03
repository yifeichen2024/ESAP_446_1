import numpy as np
import math


class Polynomial:

    def __init__(self, coefficients):
        '''
        init the Attributes:
        coefficients
        order
        '''
        self.coefficients = np.array(coefficients, dtype=int)
        self._remove_trailing_zeros()
        self.order = len(self.coefficients) - 1

    def _remove_trailing_zeros(self):
        """
        get rid off the high order zero term 
        """
        while len(self.coefficients) > 1 and self.coefficients[-1] == 0:
            self.coefficients = self.coefficients[:-1]

    @staticmethod 
    def from_string(poly_string):
        '''
        Define a polynomial from a strings
        '''
        poly_string = poly_string.replace(" ", "")
        terms = poly_string.replace("-", "+-").split("+")
        coefficients = {}

        for term in terms:
            if not term:
                continue
            
            term = term.replace("*", "")
            
            if "x^" in term:
                coef, power = term.split("x^")
                power = int(power)
                if coef in ("", "+"):
                    coef = 1
                elif coef == "-":
                    coef = -1
                else:
                    coef = int(coef)

            elif "x" in term:
                coef = term.replace("x", "")
                # coef = term.split("*x")[0]
                power = 1
                if coef in ("", "+"):
                    coef = 1
                elif coef in "-":
                    coef = -1
                else:
                    coef = int(coef)
             
            elif term:
                coef = int(term)
                power = 0
            
            # coef = int(coef) if coef else 1
            coefficients[power] = coef

        if coefficients:
            max_order = max(coefficients.keys())
            result = np.zeros(max_order + 1, dtype=int)
        
            for power, coef in coefficients.items():
                result[power] = coef
        
            return Polynomial(result)
        else:
            return Polynomial([0])

    def __repr__(self):
 
        terms = []
        for power, coef in enumerate(self.coefficients):
            if coef == 0:
                continue
            if power == 0:
                terms.append(f"{coef}")
            elif power == 1:
                if coef == 1:
                    terms.append("x")
                elif coef == -1:
                    terms.append("-x")
                else:
                    terms.append(f"{coef}*x")
            else:
                if coef == 1:
                    terms.append(f"x^{power}")
                elif coef == -1:
                    terms.append(f"-x^{power}")
                else:
                    terms.append(f"{coef}*x^{power}")
    
        # correct the plus and minos 
        result = " + ".join(terms)
        result = result.replace("+ -", "- ")  
    
        return result

    def __add__(self, other):

        max_order = max(self.order, other.order)
        new_coefficients = np.zeros(max_order + 1, dtype=int)
    
        # coeff of self
        for i in range(self.order + 1):
            new_coefficients[i] += self.coefficients[i]
    
        # coeff of self
        for i in range(other.order + 1):
            new_coefficients[i] += other.coefficients[i]
    
        # return for the polynomial type
        return Polynomial(new_coefficients)  
    
    def __sub__(self, other):

        max_order = max(self.order, other.order)
        new_coefficients = np.zeros(max_order + 1, dtype=int)

        for i in range(self.order + 1):
            new_coefficients[i] += self.coefficients[i]
    
        for i in range(other.order + 1):
            new_coefficients[i] -= other.coefficients[i]

        return Polynomial(new_coefficients)
    
    # Add for testing 
    def __neg__(self):
        return Polynomial([-coef for coef in self.coefficients])

    def __mul__(self, other):
        '''
        multiplication operation
        '''
        new_coefficients = np.zeros(self.order + other.order + 1, dtype=int)
        for i in range(self.order + 1):
            for j in range(other.order + 1):
                new_coefficients[i + j] += self.coefficients[i] * other.coefficients[j]
        
        return Polynomial(new_coefficients)

    def __eq__(self, other):
        '''
        equality for polynomial 
        Return bool 
        '''
        return np.array_equal(self.coefficients, other.coefficients)

    def __truediv__(self, other):
        '''
        Dividing two polynomials and return a RationalPolynomial
        '''
        if isinstance(other, Polynomial):
            return RationalPolynomial(self, other)
        else:
            raise ValueError("ERROR: ")



class RationalPolynomial:
    def __init__(self, numerator: Polynomial, denominator: Polynomial):
        '''
        Store a numerator and denominator,
        each of which are Polynomials,
        Both integers in Fraction 
        '''
        self.numerator = numerator
        self.denominator = denominator
        self._reduce()

    @staticmethod
    def from_string(string):
        '''
        The string used to specify a RationalPolynomial,
        numerator and denominator separated by a / character  
        '''
        string = string.replace(" ","")
        numerator_str, denominator_str = string.split("/")
        # move /
        numerator = Polynomial.from_string(numerator_str[1:-1])
        denominator = Polynomial.from_string(denominator_str[1:-1])

        return RationalPolynomial(numerator, denominator)
    
    def _reduce(self):
        """
        尝试化简分子和分母，找到分子和分母的最大公因式。
        """
        # 获取分子和分母的系数
        num_coeff = self.numerator.coefficients
        denom_coeff = self.denominator.coefficients
    
        # 找到分子和分母中最短的系数数组
        min_len = min(len(num_coeff), len(denom_coeff))
    
        # 初始化最大公因数为1
        common_gcd = 1

        # 找到分子和分母相应项的最大公因数
        for i in range(min_len): 
            common_gcd = math.gcd(num_coeff[i], denom_coeff[i])
            if common_gcd > 1:
                # 如果找到最大公因数，除以公因数
                self.numerator.coefficients[i] //= common_gcd
                self.denominator.coefficients[i] //= common_gcd
    
        # 确保分母不为负数
        if self.denominator.coefficients[-1] < 0:
            self.denominator.coefficients = [-c for c in self.denominator.coefficients]
            self.numerator.coefficients = [-c for c in self.numerator.coefficients]
            # self.numerator = -self.numerator
            # self.denominator = -self.denominator

    def __repr__(self):
        '''
        Useful visual representation 
        '''
        return f"({self.numerator}) / ({self.denominator})"

    def __add__(self, other):
        '''
        addition operation
        '''
        numerator = (self.numerator * other.denominator) + (self.denominator * other.numerator)
        denominator = self.denominator * other.denominator

        return RationalPolynomial(numerator, denominator)

    def __sub__(self, other):
        '''
        subtraction operation 
        '''
        return self + (-other)
        # numerator = (self.numerator * other.denominator) - (self.denominator * other.numerator)
        # denominator = self.denominator * other.denominator

        # return RationalPolynomial(numerator, denominator)

    def __neg__(self):

        return RationalPolynomial(-self.numerator, self.denominator)
    
    def __mul__(self, other):
        '''
        multiplication operation
        '''
        numerator = self.numerator * other.numerator
        denominator = self.denominator * other.denominator

        return RationalPolynomial(numerator, denominator)
    
    def __truediv__(self, other):
        '''
        RationalPolynomial divide 
        '''
        numerator = self.numerator * other.denominator
        denominator = self.denominator * other.numerator

        return RationalPolynomial(numerator, denominator)
    
    def __eq__(self, other):
        '''
        equality 
        '''
        if isinstance(other, RationalPolynomial):
            return (self.numerator * other.denominator) == (self.denominator * other.numerator)
        return False


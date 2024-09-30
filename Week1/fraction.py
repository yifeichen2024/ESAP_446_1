import math

class Fraction:
    
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
        self._reduce()
    
    @staticmethod
    def from_string(string):
        numerator, denominator = string.split("/")
        numerator = int(numerator)
        denominator = int(denominator)
        return Fraction(numerator, denominator)
    
    def __repr__(self):
        string = str(self.numerator) + " / " + str(self.denominator)
        return string
    
    def _reduce(self):
        gcd = math.gcd(self.numerator, self.denominator)
        self.numerator = self.numerator // gcd
        self.denominator = self.denominator // gcd
        if self.denominator < 0:
            self.numerator *= -1
            self.denominator *= -1
            
    def __add__(self, other):
        denominator = self.denominator*other.denominator
        numerator = self.numerator*other.denominator + self.denominator*other.numerator
        return Fraction(numerator, denominator)
    
    def __neg__(self):
        numerator = -1*self.numerator
        return Fraction(numerator, self.denominator)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        numerator = self.numerator*other.numerator
        denominator = self.denominator*other.denominator
        return Fraction(numerator, denominator)
    
    def __truediv__(self, other):
        numerator = self.numerator*other.denominator
        denominator = self.denominator*other.numerator
        return Fraction(numerator, denominator)
    
    def __eq__(self, other):
        if self.numerator == other.numerator:
            if self.denominator == other.denominator:
                return True
        return False

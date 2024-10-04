from polynomial import Polynomial, RationalPolynomial

a = RationalPolynomial.from_string("(1 + 7*x^2)/(3)")
b = RationalPolynomial.from_string("(-3 - x + 2*x^2)/(3)")
c = RationalPolynomial.__add__(a, b)
print("Parsed A: ", a)
print("Parsed B: ", b)
print("Result C: ", c)

# a2 = RationalPolynomial.from_string("(2 + x)/1")
# b2 = RationalPolynomial.from_string("(4 + 2*x)/2")
# c2 = RationalPolynomial.from_string("(x + 2)/1")
# print("Parsed A2 ", a2)
# print("Parsed B2: ", b2)
# print("Result C2: ", c2)
# print(a2 == b2)
# print(a2 == b2)

# a1 = RationalPolynomial.from_string("(x^2 - 4)/(x - 2)") 
# b1 = RationalPolynomial.from_string("(x + 2)/(1)")
# c1 = RationalPolynomial.from_string("(1)/(1)")
# print("Parsed A1: ", a1)
# print("Parsed B1: ", b1)
# print("ComputeC1: ", a1/b1)
# print("Result C1: ", c1)
# print(a1/b1 == c1)

rp1 = RationalPolynomial.from_string("(2 + x)/(-1 + x + 2*x^3)")
rp2 = RationalPolynomial.from_string("(-4 - 2*x)/(2 - 2*x - 4*x^3)")
assert rp1 == rp2
print(rp1)
print(rp2)
print(rp1 == rp2)
from polynomial import Polynomial, RationalPolynomial

def test_rational_polynomial_from_string():
    a = RationalPolynomial.from_string("(1 + 2*x)/(-3 + 4*x + 5*x^6)")
    assert repr(a) == "(1 + 2*x) / (-3 + 4*x + 5*x^6)"

def test_rational_polynomial_reduce():
    a = RationalPolynomial.from_string("(1 + 2*x - 5*x^2)/(-1 + x + 3*x^4)")
    b = RationalPolynomial.from_string("(-3 - 6*x + 15*x^2)/(3 - 3*x - 9*x^4)")
    assert a == b
# Addition
def test_rational_polynomial_addition():
    a = RationalPolynomial.from_string("(1 + x)/2")
    b = RationalPolynomial.from_string("(2 - x)/2")
    c = RationalPolynomial.from_string("(3)/2")
    assert a + b == c

def test_rational_polynomial_add1():
    a = RationalPolynomial.from_string("(1 + 7*x^2)/(3*x^2)")
    b = RationalPolynomial.from_string("(-3 - x + 2*x^2)/(3*x^2)")
    c = RationalPolynomial.from_string("(-2 - x + 9*x^2)/(3*x^2)")
    assert a + b == c

def test_rational_polynomial_add2():
    a = RationalPolynomial.from_string("(1 + 7*x^2)/(2 + 1*x)")
    b = RationalPolynomial.from_string("(-3 - x^2 + 2*x^3)/(2 + x)")
    c = RationalPolynomial.from_string("(-2 + 6*x^2 + 2*x^3)/(2 + x)")
    assert a + b == c

def test_rational_polynomial_add3():
    a = RationalPolynomial.from_string("(-3 - x^2 + 2*x^3)/(5 + x+ 2*x^3)")
    b = RationalPolynomial.from_string("(1 + 7*x^2)/(5 + x+ 2*x^3)")
    c = RationalPolynomial.from_string("(-2 + 6*x^2 + 2*x^3)/(5 + x+ 2*x^3)")
    assert a + b == c

def test_rational_polynomial_subtraction():
    a = RationalPolynomial.from_string("(3*x)/x")
    b = RationalPolynomial.from_string("(1)/1")
    c = RationalPolynomial.from_string("(2)/1")
    assert a - b == c

def test_rational_polynomial_subtraction1():
    a = RationalPolynomial.from_string("(-2 + 6*x^2 + 2*x^3)/(2)")
    b = RationalPolynomial.from_string("(-3 - x^2 + 2*x^3)/(2)")
    c = RationalPolynomial.from_string("(1 + 7*x^2)/(2)")
    assert a - b == c

def test_rational_polynomial_subtraction2():
    a = RationalPolynomial.from_string("(-2 - x)/(1)")
    b = RationalPolynomial.from_string("(1 + 7*x^2)/(1)")
    c = RationalPolynomial.from_string("(-3 - x - 7*x^2)/(1)")
    assert a - b == c

def test_rational_polynomial_multiplication():
    a = RationalPolynomial.from_string("(x^2)/3")
    b = RationalPolynomial.from_string("(x)/3")
    c = RationalPolynomial.from_string("(x^3)/9")
    assert a * b == c

def test_rational_polynomial_multiplication1():
    a = RationalPolynomial.from_string("(4)/(1)")
    b = RationalPolynomial.from_string("(2 - x + 3*x^2)/(2)")
    c = RationalPolynomial.from_string("(8 - 4*x + 12*x^2)/(2)")
    assert a * b == c

def test_rational_polynomial_multiplication2():
    a = RationalPolynomial.from_string("(3 - 2*x^2 + x^3)/(3)")
    b = RationalPolynomial.from_string("(2 - x + 3*x^2)/(4)")
    c = RationalPolynomial.from_string("(6 - 3*x + 5*x^2 + 4*x^3 - 7*x^4 + 3*x^5)/(12)")
    assert a * b == c

def test_rational_polynomial_division():
    a = RationalPolynomial.from_string("(x^2)/8")
    b = RationalPolynomial.from_string("(x)/2")
    c = RationalPolynomial.from_string("(x)/4")
    assert a / b == c


def test_rational_polynomial_division1():
    a = RationalPolynomial.from_string("(x^2 - 4)/(x - 2)")
    b = RationalPolynomial.from_string("(x + 2)/(1)")
    c = RationalPolynomial.from_string("(1)/(1)")
    assert a / b == c


def test_rational_polynomial_equality():
    a = RationalPolynomial.from_string("(2 + x)/1")
    b = RationalPolynomial.from_string("(4 + 2*x)/2")
    c = RationalPolynomial.from_string("(x + 2)/1")
    assert a == b
    assert a == c

def test_rational_polynomial_eq1():
    a = RationalPolynomial.from_string("(x^2 - 4)/(x - 2)")
    b = RationalPolynomial.from_string("(x + 2)/(1)")
    assert a == b

def test_rational_polynomial_eq2():
    a = RationalPolynomial.from_string("(x^2 - 4)/(x - 2)")
    b = RationalPolynomial.from_string("(x - 2)/(x^2 - 4)")
    assert a != b

def test_rational_polynomial_zero_numerator():
    a = RationalPolynomial.from_string("(0)/1")
    b = RationalPolynomial.from_string("(0)/x")
    assert a == b
    assert repr(a) == "(0) / (1)"











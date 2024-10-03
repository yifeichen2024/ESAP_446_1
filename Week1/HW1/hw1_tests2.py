import pytest
from polynomial import Polynomial, RationalPolynomial

def test_rational_polynomial_from_string():
    rp = RationalPolynomial.from_string("(2 + x)/(-1 + x + 2*x^3)")
    assert repr(rp) == "(2 + x) / (-1 + x + 2*x^3)"

def test_rational_polynomial_addition():
    a = RationalPolynomial.from_string("(1 + x)/1")
    b = RationalPolynomial.from_string("(2 - x)/1")
    c = RationalPolynomial.from_string("(3)/1")
    assert a + b == c

def test_rational_polynomial_subtraction():
    a = RationalPolynomial.from_string("(3*x)/x")
    b = RationalPolynomial.from_string("(1)/1")
    c = RationalPolynomial.from_string("(2)/1")
    assert a - b == c

def test_rational_polynomial_multiplication():
    a = RationalPolynomial.from_string("(x)/1")
    b = RationalPolynomial.from_string("(x)/1")
    c = RationalPolynomial.from_string("(x^2)/1")
    assert a * b == c

def test_rational_polynomial_division():
    a = RationalPolynomial.from_string("(x^2)/1")
    b = RationalPolynomial.from_string("(x)/1")
    c = RationalPolynomial.from_string("(x)/1")
    assert a / b == c

def test_rational_polynomial_equality():
    a = RationalPolynomial.from_string("(2 + x)/1")
    b = RationalPolynomial.from_string("(2 + x)/1")
    c = RationalPolynomial.from_string("(x + 2)/1")
    assert a == b
    assert a == c

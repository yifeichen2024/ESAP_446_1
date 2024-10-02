from polynomial import Polynomial

def test_eq():
    a = Polynomial.from_string("-4 + x^2")
    b = Polynomial.from_string("x^2 - 4")
    assert a == b
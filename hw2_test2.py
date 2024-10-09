import pytest
import numpy as np
import finite

order_range = [2, 4, 6, 8]

error_bound_1_nonuniform = [0.02, 0.005, 2e-05, 2e-07]

@pytest.mark.parametrize('order_index', np.arange(4))
def test_NUFD(order_index):
    order = order_range[order_index]
    values = np.sort(np.random.uniform(0, 2 * np.pi, 100))
    grid = finite.NonUniformPeriodicGrid(values, 2 * np.pi)
    x = grid.values

    f = 1 / (2 + np.sin(x) ** 2)

    d = finite.DifferenceNonUniformGrid(1, order, grid)

    df = d @ f
    df0 = -((2 * np.cos(x) * np.sin(x)) / (2 + np.sin(x) ** 2) ** 2)

    error = np.max(np.abs(df - df0))
    error_est = error_bound_1_nonuniform[order_index]

    assert error < error_est

error_bound_2_nonuniform = [0.01, 0.0005, 5e-06, 1e-06]
# error_bound_2_nonuniform = [0.002460084948765484, 2.8916785483446006e-05, 7.619883177767406e-07, 3.4954780403495533e-08]
@pytest.mark.parametrize('order_index', np.arange(4))
def test_NUFD_2(order_index):
    order = order_range[order_index]
    values = np.sort(np.random.uniform(0, 2 * np.pi, 100))
    grid = finite.NonUniformPeriodicGrid(values, 2 * np.pi)
    x = grid.values

    f = 1 / (2 + np.sin(x) ** 2)

    d2 = finite.DifferenceNonUniformGrid(2, order, grid)

    d2f = d2 @ f
    d2f0 = (8 * np.cos(x) ** 2 * np.sin(x) ** 2) / (2 + np.sin(x) ** 2) ** 3 - (2 * np.cos(x) ** 2) / (2 + np.sin(x) ** 2) ** 2 + (2 * np.sin(x) ** 2) / (2 + np.sin(x) ** 2) ** 2

    error = np.max(np.abs(d2f - d2f0))
    error_est = error_bound_2_nonuniform[order_index]

    assert error < error_est

error_bound_3_nonuniform = [0.1, 0.002, 1e-03, 5e-05]
# error_bound_3_nonuniform = [0.026372498378231357, 0.0007845734749516997, 3.823640687972141e-05, 2.7946235658937724e-06]
@pytest.mark.parametrize('order_index', np.arange(4))
def test_NUFD_3(order_index):
    order = order_range[order_index]
    values = np.sort(np.random.uniform(0, 2 * np.pi, 100))
    grid = finite.NonUniformPeriodicGrid(values, 2 * np.pi)
    x = grid.values

    f = 1 / (2 + np.sin(x) ** 2)

    d3 = finite.DifferenceNonUniformGrid(3, order, grid)

    d3f = d3 @ f
    d3f0 = (2 * np.cos(x) ** 3 * (37 * np.sin(x) + np.sin(3 * x))) / (2 + np.sin(x) ** 2) ** 4

    error = np.max(np.abs(d3f - d3f0))
    error_est = error_bound_3_nonuniform[order_index]

    assert error < error_est

# Extreme non-uniform grid test
error_bound_extreme_nonuniform = [0.1, 0.005, 1e-03, 5e-05]

@pytest.mark.parametrize('order_index', np.arange(4))
def test_NUFD_extreme(order_index):
    order = order_range[order_index]
    # Create an extreme non-uniform grid with large variations in spacing
    values = np.concatenate((np.linspace(0, np.pi, 50), np.linspace(np.pi + 0.1, 2 * np.pi, 50)))
    grid = finite.NonUniformPeriodicGrid(values, 2 * np.pi)
    x = grid.values

    f = 1 / (2 + np.sin(x) ** 2)

    d = finite.DifferenceNonUniformGrid(1, order, grid)

    df = d @ f
    df0 = -((2 * np.cos(x) * np.sin(x)) / (2 + np.sin(x) ** 2) ** 2)

    error = np.max(np.abs(df - df0))
    error_est = error_bound_extreme_nonuniform[order_index]

    assert error < error_est
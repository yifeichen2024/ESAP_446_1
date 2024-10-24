import pytest
import numpy as np
import finite
import timesteppers

resolution_list = [100, 200, 400]

error_BDF_variable_wave = {
    (100, 2): 0.08, (200, 2): 0.02, (400, 2): 0.005,
    (100, 4): 0.04, (200, 4): 0.008, (400, 4): 0.002
}

@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_BDF_variable_wave(resolution, spatial_order):
    grid = finite.UniformPeriodicGrid(resolution, 2 * np.pi)
    x = grid.values

    IC = np.exp(-(x - np.pi) ** 2 * 8)
    target = np.exp(-(x - np.pi - 2 * np.pi * 0.2) ** 2 * 8)

    d = finite.DifferenceUniformGrid(1, spatial_order, grid)
    ts = timesteppers.BackwardDifferentiationFormula(IC, d, 3)

    alpha = 0.1
    num_periods = 1.8
    current_time = 0
    final_time = 2 * np.pi * num_periods

    # Variable timestep evolution with alternating timestep sizes
    while current_time < final_time:
        if current_time < final_time / 3:
            dt = alpha * grid.dx
        elif current_time < 2 * final_time / 3:
            dt = 1.5 * alpha * grid.dx
        else:
            dt = 2 * alpha * grid.dx

        if current_time + dt > final_time:
            dt = final_time - current_time
        ts.step(dt)
        current_time += dt

    error = np.max(np.abs(ts.u - target))
    error_est = error_BDF_variable_wave[(resolution, spatial_order)]

    assert error < error_est

error_BDF_variable_diff = {
    (100, 2): 2e-3, (200, 2): 5e-4, (400, 2): 1.5e-4,
    (100, 4): 4e-4, (200, 4): 1e-4, (400, 4): 2e-5
}

@pytest.mark.parametrize('resolution', resolution_list)
@pytest.mark.parametrize('spatial_order', [2, 4])
def test_BDF_variable_diff(resolution, spatial_order):
    grid = finite.UniformPeriodicGrid(resolution, 50)
    x = grid.values

    IC = np.exp(-(x - 20) ** 2 / 4)
    target = 1 / np.sqrt(5) * np.exp(-(x - 20) ** 2 / 20)

    d = finite.DifferenceUniformGrid(2, spatial_order, grid)
    ts = timesteppers.BackwardDifferentiationFormula(IC, d, 3)

    alpha = 0.5
    current_time = 0
    final_time = 4

    # Variable timestep evolution with increasing timestep size
    dt_list = [alpha * grid.dx] * (resolution // 3) + [1.5 * alpha * grid.dx] * (resolution // 3) + [2 * alpha * grid.dx] * (resolution // 3)
    for dt in dt_list:
        if current_time + dt > final_time:
            dt = final_time - current_time
        ts.step(dt)
        current_time += dt

    error = np.max(np.abs(ts.u - target))
    error_est = error_BDF_variable_diff[(resolution, spatial_order)]

    assert error < error_est

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def create_simulation(x1, x2, t, p, skip_frame=8, from_top=True, writeVids=False, equal_axis=True):
    """
    Create a simulation from the solved advection-diffusion data.
    
    Parameters:
    x1, x2 (array-like): Spatial grid points.
    t (array-like): Time steps.
    p (array-like): Solution data for the advection-diffusion equation.
    skip_frame (int): Number of frames to skip for each plotted frame.
    from_top (bool): Whether to view the plot from the top.
    writeVids (bool): Whether to write the output to a video file.
    equal_axis (bool): Whether to keep axis scales equal.
    """
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x1[0], x1[-1])
    ax.set_ylim(x2[0], x2[-1])
    ax.set_zlim(-0.05, 0.2)
    
    # Clip extreme values in `p` to avoid numerical instability in plotting
    p_clipped = np.clip(p[:, :, 0], -0.05, 0.2)
    surf = ax.plot_surface(x2, x1, p_clipped, edgecolor='none')
    
    if equal_axis:
        ax.set_box_aspect([1, 1, 0.5])
    
    if writeVids:
        myVideo = animation.FFMpegWriter(fps=50, bitrate=1800)
        ani = animation.FuncAnimation(fig, update_frame, frames=range(1, len(t), skip_frame), 
                                      fargs=(ax, x1, x2, p, surf, from_top), blit=False)
        ani.save('FPE_movie.mp4', writer=myVideo)
    else:
        for j in range(1, len(t), skip_frame):
            update_frame(j, ax, x1, x2, p, surf, from_top)
            plt.pause(0.1)
    
    plt.show()

def update_frame(frame, ax, x1, x2, p, surf, from_top):
    """
    Update the frame for the animation.
    
    Parameters:
    frame (int): Current frame number.
    ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes to update.
    x1, x2 (array-like): Spatial grid points.
    p (array-like): Solution data for the advection-diffusion equation.
    surf (Poly3DCollection): The surface plot to update.
    from_top (bool): Whether to view the plot from the top.
    """
    ax.clear()
    ax.set_xlim(x1[0], x1[-1])
    ax.set_ylim(x2[0], x2[-1])
    ax.set_zlim(-0.05, 0.2)
    
    # Clip extreme values in `p` to avoid numerical instability in plotting
    p_clipped = np.clip(p[:, :, frame], -0.05, 0.2)
    surf = ax.plot_surface(x2, x1, p_clipped, edgecolor='none')
    
    if from_top:
        ax.view_init(90, 0)

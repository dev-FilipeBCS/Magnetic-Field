import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np

from matplotlib.ticker import MultipleLocator

def plot_sources(ax, sources, linewidth=1, close_loop=True):
    for src in sources:
        pos = src.pos
        current = src.current

        if current > 0:
            color = ('red', 'blue')
        elif current < 0:
            color = ('blue', 'red')
        else:
            color = ('grey', 'grey')

        n = len(pos)
        half_point = n // 2

        pos1 = pos[:half_point]
        pos2 = pos[half_point:]

        if close_loop:
            pos1 = np.vstack((pos1, pos2[0]))
            # pos2 = np.vstack((pos2, pos1[0]))

        ax.plot3D(pos1[:, 0], pos1[:, 1], pos1[:, 2],
                  color=color[0], linewidth=linewidth)

        ax.plot3D(pos2[:, 0], pos2[:, 1], pos2[:, 2],
                  color=color[1], linewidth=linewidth)

def plot_check(
        sources, 
        field_points=None, 
        view=(30,-60,0), 
        ticks=None, 
        show_axis=(True, True, True), 
        title=None,
        save=False
        ):

    plt.rcParams.update({'font.size': 8})
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=view[0], azim=view[1], roll=view[2])

    plot_sources(ax=ax, sources=sources)

    # --- Plot observation points ---
    if field_points is not None:
        ax.scatter(field_points[:, 0], field_points[:, 1], field_points[:, 2],
                   color='black', s=4, label='Observation Points')

    # --- Force equal axis sizes ---
    ax.set_aspect('equal')

    # --- Tick spacing ---
    if ticks is not None:
        ax.xaxis.set_major_locator(MultipleLocator(ticks[0]))
        ax.yaxis.set_major_locator(MultipleLocator(ticks[1]))
        ax.zaxis.set_major_locator(MultipleLocator(ticks[2]))

    # --- Labels ---
    ax.set_xlabel('X (m)', labelpad=15)
    ax.set_ylabel('Y (m)', labelpad=15)
    ax.set_zlabel('Z (m)', labelpad=10)

    # --- Axis visibility ---
    if not show_axis[0]:
        ax.set_xticks([])
        ax.set_xlabel('')
    if not show_axis[1]:
        ax.set_yticks([])
        ax.set_ylabel('')
    if not show_axis[2]:
        ax.set_zticks([])
        ax.set_zlabel('')
    
    if title:
        ax.set_title(title)

    if save==True:
        plt.savefig(fname='plot_check.png', dpi=400)

    plt.show()

def plot_solution(
        sources,
        field_points,
        h,
        mode='vector',          # 'vector' | 'scatter'
        cmap='viridis',
        log_scale=False,
        arrow_size=0.01,
        view=(30,-60,0),
        ticks=None,
        show_axis=(True, True, True),
        save=False
        ):
    
    plt.rcParams.update({'font.size': 12})
    
    fig = plt.figure(figsize=(8, 5))    
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=view[0], azim=view[1], roll=view[2])

    # --- Plot Sources ---
    plot_sources(ax=ax, sources=sources)

    # --- Plot Field ---
    if h is not None:
        h_mag = np.linalg.norm(h, axis=1)

        if log_scale:
            # norm = mcolors.LogNorm(vmin=h_mag[h_mag > 0].min(), vmax=h_mag.max())
            norm = mcolors.LogNorm(vmin=90, vmax=300000)

        else:
            norm = mcolors.Normalize(vmin=h_mag.min(), vmax=h_mag.max())

        if mode == 'vector':
            mappable = ax.quiver(
                field_points[:, 0], field_points[:, 1], field_points[:, 2],
                h[:, 0], h[:, 1], h[:, 2],
                cmap=cmap,
                array=h_mag,
                norm=norm,
                length=arrow_size,
                normalize=True
            )

        elif mode == 'scatter':
            mappable = ax.scatter(
                field_points[:, 0], field_points[:, 1], field_points[:, 2],
                c=h_mag, cmap=cmap, norm=norm, s=20
            )

        else:
            raise ValueError(f"mode must be 'vector' or 'scatter', got '{mode}'")

        cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Magnetic Field |H| (A/m)')

    # --- Labels ---
    ax.set_xlabel('X (m)', labelpad=15)
    ax.set_ylabel('Y (m)', labelpad=15)
    ax.set_zlabel('Z (m)', labelpad=10)

    # --- Tick spacing ---
    if ticks is not None:
        ax.xaxis.set_major_locator(MultipleLocator(ticks[0]))
        ax.yaxis.set_major_locator(MultipleLocator(ticks[1]))
        ax.zaxis.set_major_locator(MultipleLocator(ticks[2]))

    # --- Equal scale (no distortion, no forced square) ---
    ax.set_aspect('equal')

    # --- Axis visibility ---
    if not show_axis[0]:
        ax.set_xticks([])
        ax.set_xlabel('')
    if not show_axis[1]:
        ax.set_yticks([])
        ax.set_ylabel('')
    if not show_axis[2]:
        ax.set_zticks([])
        ax.set_zlabel('')

    plt.tight_layout()

    if save:
        plt.savefig(fname='plot_solution.png', dpi=600)

    plt.show()
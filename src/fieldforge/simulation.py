import numpy as np

#### VECTORIZED SIMULATION FUNCTIONs ####
def sim_h(sources, field_points):
    n_obs = len(field_points)
    h_total = np.zeros((n_obs, 3))
    
    for src in sources:
        pos = src.pos
        dl = src.dL
        current = src.current

        r_vec = field_points[:, None, :] - pos[None, :, :]
        r = np.linalg.norm(r_vec, axis=2)
        r_safe = np.where(r < 1e-12, 1e-12, r)
        r_hat = r_vec / r_safe[:, :, None]

        cross_x = dl[:, 1] * r_hat[:, :, 2] - dl[:, 2] * r_hat[:, :, 1]
        cross_y = dl[:, 2] * r_hat[:, :, 0] - dl[:, 0] * r_hat[:, :, 2]
        cross_z = dl[:, 0] * r_hat[:, :, 1] - dl[:, 1] * r_hat[:, :, 0]

        scale = current / (4 * np.pi * r_safe**2)
        h_total[:, 0] += np.sum(cross_x * scale, axis=1)
        h_total[:, 1] += np.sum(cross_y * scale, axis=1)
        h_total[:, 2] += np.sum(cross_z * scale, axis=1)
    
    return h_total

def sim_b(sources, field_points):
    """
    Compute magnetic field B = μ₀ * H
    Uses vectorized version by default.
    """
    mu = np.pi * 4e-7
    h = sim_h(sources, field_points)
    b = h * mu
    return b


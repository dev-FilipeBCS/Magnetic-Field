import numpy as np

#### VECTORIZED SIMULATION FUNCTIONS ####
def sim_h_vectorized(sources, field_points):
    n_obs = len(field_points)
    h_total = np.zeros((n_obs, 3))
    
    for src in sources:
        pos = src.pos
        dl = src.dL
        current = src.current

        r_vec = field_points[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (M, N, 3)
        r = np.linalg.norm(r_vec, axis=2)
        r = np.where(r < 1e-12, 1e-12, r)
        r_hat = r_vec / r[:, :, np.newaxis]

        dl_expanded = dl[np.newaxis, :, :]
        cross_prod = np.cross(dl_expanded, r_hat, axis=2)

        factor = current / (4 * np.pi * (r ** 2)[:, :, np.newaxis])
        dh = factor * cross_prod
        h_total += np.sum(dh, axis=1)
    
    return h_total

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

def sim_h_old(sources, field_points):
    n_obs = len(field_points)
    h = np.zeros((n_obs, 3))

    for j in range(n_obs):
        obs = field_points[j]
        for src in sources:
            pos = src.pos
            dl = src.dL
            current = src.current

            dh = np.zeros_like(pos)
            for i in range(len(pos)):
                rv = obs - pos[i]
                rs = np.linalg.norm(rv)
                rs = max(rs, 1e-12)
                ruv = rv / rs
                dh[i] = current * np.cross(dl[i], ruv) / (4 * np.pi * rs**2)
            h[j, :] += np.sum(dh, axis=0)
    return h

def sim_b(sources, field_points, vectorized=True):
    """
    Compute magnetic field B = μ₀ * H
    Uses vectorized version by default.
    """
    mu = np.pi * 4e-7 
    if vectorized:
        h = sim_h(sources, field_points)
    else:
        h = sim_h_old(sources, field_points)
    b = h * mu
    return b


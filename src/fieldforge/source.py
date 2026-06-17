import numpy as np


# Creates Function to aide with rotation
def rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ])

class Source:
    def __init__(self, pos, dL, current, source_type=None):
        self._pos = np.array(pos, dtype=float)
        self._dL = np.array(dL, dtype=float)
        self._current = current
        self._type = source_type

        # Compute center automatically
        self._center = np.mean(self._pos, axis=0)

        # Make parameters immutable
        self._pos.setflags(write=False)
        self._dL.setflags(write=False)

    @property
    def pos(self):
        return self._pos

    @property
    def dL(self):
        return self._dL

    @property
    def current(self):
        return self._current

    @property
    def center(self):
        return self._center
    
    #######################
    # Translate Functions #
    #######################
    def move(self, t):
        t = np.asarray(t)

        return Source(
            pos=self._pos + t,
            dL=self._dL,
            current=self._current,
            source_type=self._type
        )
    
    def move_to(self, target):
        target = np.asarray(target)
        shift = target - self._center
        return self.move(shift)

    ######################
    # Rotation Functions #
    ######################
    def rotate(self, axis, angle, deg=False):
        if deg:
            angle = np.deg2rad(angle)

        R = rotation_matrix(axis, angle)

        return Source(
            pos=self._pos @ R.T,
            dL=self._dL @ R.T,
            current=self._current,
            source_type=self._type
        )
    
    def rotate_about(self, axis, angle, deg=False, point=None):
        if deg:
            angle = np.deg2rad(angle)
        
        # Use provided point as pivot, otherwise default to object's center
        pivot = np.asarray(point) if point is not None else self._center
        
        R = rotation_matrix(axis, angle)
        centered = self._pos - pivot
        return Source(
            pos=centered @ R.T + pivot,
            dL=self._dL @ R.T,
            current=self._current,
            source_type=self._type
        )

#### SOURCE CREATION ####
def solenoid(radius, height, current, turns, n_elements):
    """
    Returns a single Source (all rings combined).
    """
    z_positions = np.linspace(-height/2, height/2, turns) if turns > 1 else [0.0]

    phi = np.linspace(0, 2*np.pi, n_elements, endpoint=False)

    all_pos = []
    all_dl = []

    for z in z_positions:
        pos = np.zeros((n_elements, 3))
        pos[:, 0] = radius * np.cos(phi)
        pos[:, 1] = radius * np.sin(phi)
        pos[:, 2] = z

        dL = np.zeros((n_elements, 3))
        dL[:, 0] = -radius * np.sin(phi) * (2*np.pi/n_elements)
        dL[:, 1] =  radius * np.cos(phi) * (2*np.pi/n_elements)

        all_pos.append(pos)
        all_dl.append(dL)

    pos = np.vstack(all_pos)
    dL = np.vstack(all_dl)

    return Source(pos, dL, current, source_type="solenoid")

def solenoid_spiral(radius, height, current, turns, n_elements):
    """
    Creates a helical (spiral) solenoid.
    """

    # Total angle spans all turns
    phi = np.linspace(0, 2*np.pi * turns, n_elements)

    # Positions
    pos = np.zeros((n_elements, 3))
    pos[:, 0] = radius * np.cos(phi)
    pos[:, 1] = radius * np.sin(phi)
    pos[:, 2] = height * (phi / (2*np.pi * turns)) - height/2

    # Differential elements (tangent to helix)
    dphi = phi[1] - phi[0]

    dL = np.zeros((n_elements, 3))
    dL[:, 0] = -radius * np.sin(phi) * dphi
    dL[:, 1] =  radius * np.cos(phi) * dphi
    dL[:, 2] =  height / (2*np.pi * turns) * dphi

    return Source(pos, dL, current, source_type="solenoid_spiral")
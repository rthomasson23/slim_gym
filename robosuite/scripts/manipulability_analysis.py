import numpy as np

def calculateManipulabilityEllipsoid(Jv, Jw):
    """
    Calculates the manipulability ellipsoids of a robot given its Jacobian.

    Jv: Linear velocity Jacobian
    Jw: Angular velocity Jacobian

    Returns: Two lists of eigenvalues and eigenvectors for the manipulability
    ellipsoids (linear and rotational) of the robot.
    """

    # Calculate the manipulability ellipsoid for the linear velocity
    A = np.dot(Jv, Jv.T)
    w, v = np.linalg.eig(A)
    w = np.sqrt(w)

    # Calculate the manipulability ellipsoid for the angular velocity
    A = np.dot(Jw, Jw.T)
    w2, v2 = np.linalg.eig(A)
    w2 = np.sqrt(w2)

    return w, v, w2, v2






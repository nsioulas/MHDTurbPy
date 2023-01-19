import numpy as np


def minvar(data):
    """
    This program computes the principal variance directions and variances of a
    vector quantity as well as the associated eigenvalues.

    Parameters
    -----------
    data:
        Vxyz, an (npoints, ndim) array of data(ie Nx3)

    Returns
    -------
    vrot:
        an array of (npoints, ndim) containing the rotated data in the new coordinate system, ijk.
        Vi(maximum direction)=vrot[0,:]
        Vj(intermediate direction)=vrot[1,:]
        Vk(minimum variance direction)=Vrot[2,:]
    v:
        an (ndim,ndim) array containing the principal axes vectors
        Maximum variance direction eigenvector, Vi=v[*,0]
        Intermediate variance direction, Vj=v[*,1] (descending order)
    w:
        the eigenvalues of the computation
    """

    #  Min var starts here
    # data must be Nx3
    vecavg = np.nanmean(data, axis=0)

    mvamat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mvamat[i, j] = np.nanmean(data[:, i] * data[:, j]) - vecavg[i] * vecavg[j]

    # Calculate eigenvalues and eigenvectors
    w, v = np.linalg.eigh(mvamat, UPLO='U')

    # Sorting to ensure descending order
    w = np.abs(w)
    idx = np.flip(np.argsort(w))
    w = w[idx]
    v = v[:, idx]

    # Rotate intermediate var direction if system is not Right Handed
    YcrossZdotX = v[0, 0] * (v[1, 1] * v[2, 2] - v[2, 1] * v[1, 2])
    if YcrossZdotX < 0:
        v[:, 1] = -v[:, 1]

    # Ensure minvar direction is along +Z (for FAC system)
    if v[2, 2] < 0:
        v[:, 2] = -v[:, 2]
        v[:, 1] = -v[:, 1]

    s = data.shape
    vrot = np.array([np.dot(row, v) for row in data])

    return vrot, v, w
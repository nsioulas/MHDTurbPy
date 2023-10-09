from joblib import Parallel, delayed
from scipy.integrate import quad


def calculate_moment(funcptr, low_lim, up_lim):
    return quad(funcptr, low_lim, up_lim)[0]


def parallel_integration_loop(funcptr, moment, low_lim, up_lim):
    return quad(funcptr, low_lim, up_lim)[0] * moment


def parallel_integration(funcptr_list, moments, low_lim, up_lim, n_jobs):
    keep_moments = Parallel(n_jobs=n_jobs)(
        delayed(parallel_integration_loop)(funcptr_list[i], moments[i], low_lim, up_lim)
        for i in range(len(moments))
    )
    return keep_moments

"""
Utility module for miscellaneous stuff
"""
import fractions
import math
import sys

def isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0):
    """
    Python 3.4 does not have math.isclose, so we need to steal it and add it here.
    """
    try:
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    except AttributeError:
        # Running on older version of python, fall back to hand-rolled implementation
        if (rel_tol < 0.0) or (abs_tol < 0.0):
            raise ValueError("Tolerances must be non-negative, but are rel_tol: {} and abs_tol: {}".format(rel_tol, abs_tol))
        if math.isnan(a) or math.isnan(b):
            return False  # NaNs are never close to anything, even other NaNs
        if (a == b):
            return True
        if math.isinf(a) or math.isinf(b):
            return False  # Infinity is only close to itself, and we already handled that case
        diff = abs(a - b)
        return (diff <= rel_tol * abs(b)) or (diff <= rel_tol * abs(a)) or (diff <= abs_tol)

def lcm(a, b):
    """
    Python 3.4 and others differ on how to get at the least common multiple.
    """
    major, minor, _micro, _level, _serial = sys.version_info

    if major > 3 or minor > 4:
        return a * b // math.gcd(a, b)
    else:
        return a * b // fractions.gcd(a, b)

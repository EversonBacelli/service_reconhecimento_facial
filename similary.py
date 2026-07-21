
from numpy.linalg import norm
import numpy as np

def cosine_similarity(v1,v2):

    return np.dot(v1,v2) / (
        norm(v1)*norm(v2)
    )
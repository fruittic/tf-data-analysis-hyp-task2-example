import pandas as pd
import numpy as np
from hyppo.ksample import MMD

chat_id = 224851402 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    alpha = 0.06
    return MMD(compute_kernel="rbf", gamma=1).test(x, y)[1] < alpha

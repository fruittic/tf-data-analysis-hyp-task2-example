import pandas as pd
import numpy as np
from scipy import stats

chat_id = 224851402 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    alpha = 0.06
    return stats.ks_2samp(x, y).pvalue < alpha

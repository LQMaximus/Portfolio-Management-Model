# -*- coding: utf-8 -*-
"""
Created on 2024/7/27

@author: LiuQuan (Maximus)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from joblib import Parallel, delayed
from typing import Optional

class MVO:
    '''
    Discription:
    使用均值方差模型得到资产配置模型
    
    Methods:
    --------
    '''

    def __init__(self,returns,cov_matrix,target_vol: Optional[float] = None, target_return: Optional [float] = None):
        '''
        Parameters:

        returns: np.ndarray
            各类资产的预期年化收益率
        cov_matrix: np.array
            资产的年化协方差矩阵
        target_vol: float
            资产组合的综合年化波动率std
        target_return:
            资产组合的

        '''

    
    def setParams(self,cov_matrix=None,returns,vol)
        
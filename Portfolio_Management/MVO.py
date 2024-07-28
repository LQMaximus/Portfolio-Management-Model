# -*- coding: utf-8 -*-
"""
Created on 2024/7/27

@author: LiuQuan (Maximus)
"""
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize


class MVO:
    '''
    Discription:
    Asset allocation using Mean Variance model

    Attributes
    --------
    returns : np.ndarray
        The expected annualized returns of various assets.
    cov_matrix : np.ndarray
        The covariance matrix of asset returns.
    individual_vol : np.ndarray
        The individual volatilities of each asset.
    num_assets : int
        The number of asset
    
    Methods:
    --------
    portfolio_performance(weights, returns, cov_matrix):
        Calculates the expected return and volatility of a portfolio.
    __maximize_return(weights, returns, cov_matrix):
        Returns the negative of the portfolio's expected return.
    optimize_return_numerical(target_volatility, initial_weights, bounds, options, constraints, disp):
        Optimizes the portfolio to achieve a given target volatility.
    plot_efficient_frontier(vol):
        Plots the efficient frontier for a range of target volatilities.
    '''

    def __init__(self,returns: np.ndarray, cov_matrix: np.ndarray):
        '''
        Description:
        -----------
        Constructs all the necessary attributes for the MVO object

        Args:
        -----------
        returns: np.ndarray float
            Expected Annualized Returns of Various Assets
        cov_matrix: np.ndarray float
            The covariance matrix of Assets
        target_vol: float
            target volatility(standard deviation)
        target_return:
            target return (annualized)

        '''
        self._returns = returns
        self._cov_matrix = cov_matrix
        self.individual_vol = np.sqrt(np.diag(self._cov_matrix))
        self.num_assets = len(self._returns)
        
    # Getter and Setter for returns
    @property
    def returns(self) -> np.ndarray:
        return self._returns
    
    @returns.setter
    def returns(self, returns: np.ndarray):
        self._returns = returns
        self.num_assets = len(self._returns)

    # Getter and Setter for cov_matrix
    @property
    def cov_matrix(self) -> np.ndarray:
        return self._cov_matrix
    
    @cov_matrix.setter
    def cov_matrix(self, cov_matrix: np.ndarray):
        self._cov_matrix = cov_matrix


    def portfolio_performance(self, weights: np.ndarray, returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[float, float]:
        '''
        Description:
        ------------
        Calculate the portfolio's volatility (std) and returns.

        Parameters:
        -----------
        weights : np.ndarray
            The weights of the assets in the portfolio.
        returns : np.ndarray
            The expected returns for each asset.
        cov_matrix : np.ndarray
            The covariance matrix of asset returns.

        Returns:
        --------
        Tuple[float, float]
            The expected return and volatility of the portfolio.
        '''
        portfolio_return = np.sum(weights * returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_return, portfolio_volatility
    
    def __maximize_return(self,weights, returns, cov_matrix):
        '''
        Description:
            calculate the return and maximize it given volatility using scipy.optimize.minimize
            Return the negative value of returns since we need to minimize 
        '''
        # 返回收益的负值，因为 scipy.optimize.minimize 是最小化目标函数
        return -self.portfolio_performance(weights, returns, cov_matrix)[0]
    
    def optimize_return_numerical(self,target_volatility,initial_weights=None,bounds=None ,options=None, constraints= None, disp = False):
        '''
        Description:
        ------------
        Optimize the portfolio to achieve a given target volatility.

        Parameters:
        -----------
        target_volatility : float
            The target volatility (standard deviation).
        initial_weights : Optional[np.ndarray]
            The initial weights for the optimization.Default:initial_weights = np.array(self.num_assets * [1. / self.num_assets])
        bounds : Optional[Tuple[Tuple[float, float], ...]]
            The bounds for the weights of the assets. Default: bounds = tuple((0, 1) for _ in range(self.num_assets))
        options : Optional[Dict[str, float]] 
            The options for the optimizer.Default:options = {'maxiter': 1000, 'ftol': 1e-9, 'disp': disp, 'eps': 1e-8}
        constraints : Optional[Dict[str, float]]
            The constraints for the optimization.
        disp : bool
            Whether to display optimization output.

        Returns:
        --------
        minimize.OptimizeResult
            The result of the optimization.
        '''
        
        if initial_weights is None:
            initial_weights = np.array(self.num_assets * [1. / self.num_assets])
        
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        if options is None:
            options = {'maxiter': 1000, 'ftol': 1e-9, 'disp': disp, 'eps': 1e-8}
        
        if constraints is None:
            # 定义约束条件
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重之和等于 1
                    {'type': 'ineq', 'fun': lambda x:  target_volatility - self.portfolio_performance(weights=x, returns=self._returns, cov_matrix=self._cov_matrix)[1]})  # 波动率小于目标波动率

        # 优化过程，最大化收益
        result = minimize(self.__maximize_return,x0 = initial_weights, args=(self.returns, self.cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints, options=options)
        # maximize_return 需要最小化的函数 
        # initial_weights 初始搜索值
        # args 传递给
        return result
    
    def plot_efficient_frontier(self,vol:np.ndarray=None):
        '''   
        Description:
        ------------
        Plots the efficient frontier for a range of target volatilities.

        Parameters:
        -----------
        vol : np.ndarray, optional
            An array of target volatilities to plot. Default: volatility = np.linspace(0, 1, 500)
        '''
        volatility = np.linspace(0, 1, 500)
        efficient_returns = np.array([])
        successful_volatilities = np.array([])
        # 并行计算，能够提升10倍速度
        res = Parallel(n_jobs=-1)(delayed(self.optimize_return_numerical)(target_volatility=v) for v in volatility)

        for res, vol  in zip(res,volatility):
            if res.success:
                efficient_returns = np.append(efficient_returns,-res.fun)
                successful_volatilities = np.append(successful_volatilities,vol)
            else:
                pass    
                # 优化失败
                # print(f"Optimization failed for target volatility {vol}")

        # 绘制有效前沿
        plt.figure(figsize=(10, 6))
        plt.plot(successful_volatilities, efficient_returns, 'b-', marker='o', markersize=3)

        # 绘制单个资产点
        for i in range(self.num_assets):
            plt.plot(self.individual_vol[i], self.returns[i], 'ro', markersize=10)
            plt.text(self.individual_vol[i], self.returns[i], f'Asset {i+1}', fontsize=12, verticalalignment='bottom')

        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.title('Efficient Frontier')
        plt.grid(True)
        plt.show()
            
        
        
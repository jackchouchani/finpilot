import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy.optimize import minimize

class PortfolioOptimizationAgent:
    def optimize(self, portfolio):
        if not portfolio:
            return {"erreur": "Aucun portefeuille trouvé pour cet utilisateur"}
        print(portfolio)

        tickers = [stock['symbol'] for stock in portfolio]
        weights = [float(stock['weight']) / 100 for stock in portfolio]  # Convertir en float et en pourcentage

        data = yf.download(tickers, period="5y")['Adj Close']
        returns = data.pct_change().dropna()

        def portfolio_performance(weights, mean_returns, cov_matrix):
            returns = np.sum(mean_returns * weights) * 252
            std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return std, returns

        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
            p_std, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
            return -(p_ret - risk_free_rate) / p_std

        def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
            num_assets = len(mean_returns)
            args = (mean_returns, cov_matrix, risk_free_rate)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bound = (0.0, 1.0)
            bounds = tuple(bound for asset in range(num_assets))
            result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)
            return result

        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        risk_free_rate = 0.01

        opt = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
        
        current_std, current_ret = portfolio_performance(weights, mean_returns, cov_matrix)
        current_sharpe = (current_ret - risk_free_rate) / current_std

        opt_std, opt_ret = portfolio_performance(opt.x, mean_returns, cov_matrix)
        opt_sharpe = (opt_ret - risk_free_rate) / opt_std

        rapport = f"""
Analyse de l'optimisation du portefeuille

Portefeuille actuel:
- Poids: {dict(zip(tickers, weights))}
- Rendement attendu: {current_ret:.2f}
- Volatilité: {current_std:.2f}
- Ratio de Sharpe: {current_sharpe:.2f}

Portefeuille optimisé:
- Poids: {dict(zip(tickers, opt.x))}
- Rendement attendu: {opt_ret:.2f}
- Volatilité: {opt_std:.2f}
- Ratio de Sharpe: {opt_sharpe:.2f}

Conclusion:
Cette analyse fournit un aperçu de l'optimisation du portefeuille. Les investisseurs peuvent utiliser ces informations pour ajuster leur stratégie d'investissement.
"""
        return rapport

portfolio_optimization_agent = PortfolioOptimizationAgent()
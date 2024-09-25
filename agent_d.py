import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

class PortfolioOptimizationAgent:
    def optimize(self, portfolio):
        if not portfolio:
            return "Désolé, je n'ai pas trouvé de portefeuille à analyser. Pouvez-vous vérifier et me fournir les détails de votre portefeuille ?"

        tickers = [stock['symbol'] for stock in portfolio]
        weights = [float(stock['weight']) / 100 for stock in portfolio]
        
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
        risk_free_rate = 0.03

        opt = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
        current_std, current_ret = portfolio_performance(weights, mean_returns, cov_matrix)
        current_sharpe = (current_ret - risk_free_rate) / current_std
        opt_std, opt_ret = portfolio_performance(opt.x, mean_returns, cov_matrix)
        opt_sharpe = (opt_ret - risk_free_rate) / opt_std

        current_weights = {ticker: f"{weight:.2%}" for ticker, weight in zip(tickers, weights)}
        optimized_weights = {ticker: f"{weight:.2%}" for ticker, weight in zip(tickers, opt.x)}

        rapport = f"""Bonjour ! J'ai analysé votre portefeuille et j'ai quelques recommandations intéressantes à vous partager. Voici un résumé de mon analyse :

📊 Votre portefeuille actuel :
{self._format_portfolio(current_weights)}

Avec cette répartition, voici vos indicateurs actuels :
📈 Rendement attendu : {current_ret:.2%}
📉 Volatilité : {current_std:.2%}
💹 Ratio de Sharpe : {current_sharpe:.2f}

Après optimisation, voici ce que je suggère :

🔄 Portefeuille optimisé :
{self._format_portfolio(optimized_weights)}

Cette nouvelle répartition pourrait vous offrir :
📈 Rendement attendu : {opt_ret:.2%} ({opt_ret - current_ret:+.2%})
📉 Volatilité : {opt_std:.2%} ({opt_std - current_std:+.2%})
💹 Ratio de Sharpe : {opt_sharpe:.2f} ({opt_sharpe - current_sharpe:+.2f})

💡 Ce que cela signifie pour vous :
1. Le rendement attendu est {'amélioré' if opt_ret > current_ret else 'réduit'}, passant de {current_ret:.2%} à {opt_ret:.2%}.
2. La volatilité {'augmente' if opt_std > current_std else 'diminue'}, ce qui implique {'plus' if opt_std > current_std else 'moins'} de risque, mais aussi {'plus' if opt_std > current_std else 'moins'} de potentiel de gain.
3. Le ratio de Sharpe {'s\'améliore' if opt_sharpe > current_sharpe else 'se dégrade'}, indiquant un {'meilleur' if opt_sharpe > current_sharpe else 'moins bon'} équilibre rendement/risque.

🔑 Points clés à considérer :
{self._generate_key_points(current_weights, optimized_weights)}

N'oubliez pas que cette analyse est basée sur des données historiques et des modèles mathématiques. Elle ne garantit pas les performances futures. Il est toujours recommandé de diversifier et d'ajuster votre stratégie en fonction de votre situation personnelle et de vos objectifs à long terme.

Que pensez-vous de ces suggestions ? Souhaitez-vous que nous discutions plus en détail de certains aspects spécifiques de cette analyse ?"""

        return rapport

    def _format_portfolio(self, weights):
        return "\n".join([f"- {ticker}: {weight}" for ticker, weight in weights.items()])

    def _generate_key_points(self, current, optimized):
        points = []
        for ticker in current.keys():
            current_weight = float(current[ticker].strip('%')) / 100
            optimized_weight = float(optimized[ticker].strip('%')) / 100
            diff = optimized_weight - current_weight
            if abs(diff) > 0.05:  # Seuil arbitraire pour les changements significatifs
                if diff > 0:
                    points.append(f"- L'optimisation suggère d'augmenter significativement la part de {ticker} (de {current[ticker]} à {optimized[ticker]}).")
                else:
                    points.append(f"- L'optimisation suggère de réduire significativement la part de {ticker} (de {current[ticker]} à {optimized[ticker]}).")
        
        if not points:
            points.append("- Les changements suggérés sont relativement mineurs pour tous les titres.")
        
        return "\n".join(points)

portfolio_optimization_agent = PortfolioOptimizationAgent()

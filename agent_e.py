import numpy as np
import yfinance as yf
from scipy.stats import norm

class RiskManagementAgent:
    def analyze(self, portfolio, confidence_level=0.95):
        if not portfolio:
            return {"erreur": "Aucun portefeuille trouvé pour cet utilisateur"}

        tickers = [stock['symbol'] for stock in portfolio]
        weights = np.array([stock['weight'] / 100 for stock in portfolio])

        data = yf.download(tickers, period="1y")['Adj Close']
        returns = data.pct_change().dropna()

        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * np.sqrt(252)

        VaR = norm.ppf(1-confidence_level) * portfolio_std
        CVaR = -1 * (norm.pdf(norm.ppf(1-confidence_level)) / (1-confidence_level)) * portfolio_std

        rapport = f"""
Analyse de gestion des risques

Rendement du portefeuille: {portfolio_return:.2f}
Volatilité du portefeuille: {portfolio_std:.2f}
Valeur à Risque (VaR) à {confidence_level * 100:.0f}% de confiance: {-VaR:.2f}
VaR Conditionnelle (CVaR) à {confidence_level * 100:.0f}% de confiance: {-CVaR:.2f}

Conclusion:
Cette analyse fournit un aperçu des risques associés au portefeuille. Les investisseurs devraient utiliser ces informations pour ajuster leur stratégie d'investissement en fonction de leur tolérance au risque.
"""
        return rapport

risk_management_agent = RiskManagementAgent()
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

        return {
            "rendement_portefeuille": float(portfolio_return),
            "volatilité_portefeuille": float(portfolio_std),
            "Valeur_à_Risque": float(-VaR),
            "VaR_Conditionnelle": float(-CVaR),
            "niveau_confiance": confidence_level
        }

risk_management_agent = RiskManagementAgent()
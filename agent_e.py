import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
from sklearn.covariance import LedoitWolf

class RiskManagementAgent:
    def analyze(self, portfolio, confidence_level=0.95, risk_free_rate=0.02):
        if not portfolio:
            return {"erreur": "Aucun portefeuille trouvé pour cet utilisateur"}

        tickers = [stock['symbol'] for stock in portfolio]
        weights = np.array([stock['weight'] / 100 for stock in portfolio])

        data = yf.download(tickers, period="2y")['Adj Close']
        returns = data.pct_change().dropna()

        # Utilisation de l'estimateur de covariance Ledoit-Wolf pour une meilleure stabilité
        lw = LedoitWolf().fit(returns)
        cov_matrix = lw.covariance_

        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

        # Calcul du VaR et CVaR
        VaR = norm.ppf(1-confidence_level) * portfolio_std
        CVaR = -1 * (norm.pdf(norm.ppf(1-confidence_level)) / (1-confidence_level)) * portfolio_std

        # Calcul du ratio de Sharpe
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std

        # Calcul des contributions au risque
        marginal_risk = np.dot(cov_matrix, weights) / portfolio_std
        risk_contribution = weights * marginal_risk
        percentage_risk_contribution = risk_contribution / np.sum(risk_contribution)

        # Calcul des drawdowns
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min().min()

        rapport = f"""
Analyse de gestion des risques avancée

1. Mesures de performance:
   Rendement annualisé du portefeuille: {portfolio_return:.2%}
   Volatilité annualisée du portefeuille: {portfolio_std:.2%}
   Ratio de Sharpe: {sharpe_ratio:.2f}

2. Mesures de risque:
   Valeur à Risque (VaR) à {confidence_level*100:.0f}% de confiance: {-VaR:.2%}
   VaR Conditionnelle (CVaR) à {confidence_level*100:.0f}% de confiance: {-CVaR:.2%}
   Drawdown maximal: {max_drawdown:.2%}

3. Contribution au risque par action:
"""
        for ticker, weight, contrib in zip(tickers, weights, percentage_risk_contribution):
            rapport += f"   {ticker}: Poids {weight:.2%}, Contribution au risque {contrib:.2%}\n"

        rapport += f"""
4. Analyse de corrélation:
   Les 3 paires d'actions les plus corrélées:
"""
        corr_matrix = returns.corr()
        corr_pairs = []
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                corr_pairs.append((tickers[i], tickers[j], corr_matrix.iloc[i, j]))
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for pair in corr_pairs[:3]:
            rapport += f"   {pair[0]} - {pair[1]}: {pair[2]:.2f}\n"

        rapport += f"""
5. Analyse de scénarios:
   Impact estimé d'une baisse de marché de 10%: {-10 * portfolio_std / np.sqrt(252):.2%}
   Impact estimé d'une hausse des taux d'intérêt de 1%: Nécessite une analyse plus approfondie des sensibilités aux taux

Conclusion:
Cette analyse avancée fournit un aperçu détaillé des risques et de la performance du portefeuille. 
Points clés à retenir:
1. Le portefeuille a un ratio de Sharpe de {sharpe_ratio:.2f}, indiquant {'une bonne' if sharpe_ratio > 1 else 'une faible'} performance ajustée au risque.
2. La VaR suggère une perte potentielle maximale de {-VaR:.2%} dans 95% des cas sur une journée.
3. {tickers[np.argmax(percentage_risk_contribution)]} contribue le plus au risque global du portefeuille.
4. Le drawdown maximal de {max_drawdown:.2%} indique la perte maximale historique du portefeuille.

Recommandations:
1. {'Envisager de réduire l'exposition à ' + tickers[np.argmax(percentage_risk_contribution)] if max(percentage_risk_contribution) > 0.3 else 'La diversification semble adéquate'}
2. {'Le ratio de Sharpe pourrait être amélioré en ajustant l'allocation des actifs' if sharpe_ratio < 1 else 'Maintenir la stratégie actuelle qui offre un bon équilibre rendement/risque'}
3. Surveiller de près les corrélations élevées entre certains actifs pour éviter une concentration excessive du risque
4. Envisager des stratégies de couverture pour atténuer l'impact potentiel des scénarios de baisse identifiés

Les investisseurs devraient utiliser ces informations pour ajuster leur stratégie d'investissement en fonction de leur tolérance au risque et de leurs objectifs financiers.
"""
        return rapport

risk_management_agent = AdvancedRiskManagementAgent()
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

class ComplianceAgent:
    def __init__(self):
        self.regulations = {
            "max_single_stock_exposure": 0.30,
            "min_stocks_count": 5,
            "restricted_sectors": ["Tobacco", "Gambling", "Weapons"],
            "esg_score_threshold": 50,
            "max_sector_exposure": 0.40,
            "min_developed_markets_exposure": 0.60,
            "max_volatility": 0.25
        }

    def get_stock_data(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'sector': info.get('sector', 'Non disponible'),
                'esg_score': info.get('esgScore', 'Non disponible'),
                'country': info.get('country', 'Non disponible'),
                'beta': info.get('beta', 'Non disponible')
            }
        except:
            return {
                'sector': 'Non disponible',
                'esg_score': 'Non disponible',
                'country': 'Non disponible',
                'beta': 'Non disponible'
            }

    def calculate_volatility(self, tickers, weights):
        try:
            data = yf.download(tickers, period="1y")['Adj Close']
            returns = data.pct_change().dropna()
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * np.sqrt(252)
            return portfolio_std, returns
        except:
            return 'Non disponible', None

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        if returns is not None:
            excess_returns = returns.mean() * 252 - risk_free_rate
            return excess_returns / (returns.std() * np.sqrt(252))
        return 'Non disponible'

    def calculate_correlation(self, returns):
        if returns is not None:
            corr_matrix = returns.corr()
            return corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)]
        return []

    def check_compliance(self, portfolio):
        violations = []
        warnings = []
        portfolio_dict = {stock['symbol']: stock['weight'] / 100 for stock in portfolio}
        
        sector_exposure = {}
        market_exposure = {'Développé': 0, 'Émergent': 0, 'Non disponible': 0}
        stock_data = {}
        
        for stock in portfolio:
            data = self.get_stock_data(stock['symbol'])
            stock_data[stock['symbol']] = data
            
            sector = data['sector']
            sector_exposure[sector] = sector_exposure.get(sector, 0) + stock['weight'] / 100
            
            country = data['country']
            market_type = 'Développé' if country in ['United States', 'Canada', 'United Kingdom', 'Germany', 'France', 'Japan', 'Australia'] else 'Émergent'
            market_exposure[market_type] += stock['weight'] / 100

        tickers = [stock['symbol'] for stock in portfolio]
        weights = np.array([stock['weight'] / 100 for stock in portfolio])
        volatility, returns = self.calculate_volatility(tickers, weights)
        
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        correlations = self.calculate_correlation(returns)

        # Vérifications de conformité (comme avant)
        # ...

        rapport = self.generate_report(violations, warnings, sector_exposure, market_exposure, volatility, portfolio_dict, stock_data, sharpe_ratio, correlations, portfolio)
        return rapport

    def generate_report(self, violations, warnings, sector_exposure, market_exposure, volatility, portfolio_dict, stock_data, sharpe_ratio, correlations, portfolio):
        rapport = f"""
Analyse de conformité avancée

Statut de conformité: {"Non conforme" if violations else "Conforme avec avertissements" if warnings else "Totalement conforme"}

1. Violations majeures:
"""
        if violations:
            for violation in violations:
                rapport += f"   - {violation}\n"
        else:
            rapport += "   Aucune violation majeure détectée.\n"

        rapport += f"""
2. Avertissements et recommandations:
"""
        if warnings:
            for warning in warnings:
                rapport += f"   - {warning}\n"
        else:
            rapport += "   Aucun avertissement à signaler.\n"

        rapport += f"""
3. Exposition sectorielle:
"""
        for sector, exposure in sector_exposure.items():
            rapport += f"   - {sector}: {exposure:.2%}\n"

        rapport += f"""
4. Exposition géographique:
   - Marchés développés: {market_exposure['Développé']:.2%}
   - Marchés émergents: {market_exposure['Émergent']:.2%}
   - Non disponible: {market_exposure['Non disponible']:.2%}

5. Mesures de risque:
   - Volatilité du portefeuille: {volatility:.2%}
   - Ratio de Sharpe: {sharpe_ratio:.2f}

6. Analyse de concentration:
   - Action la plus importante: {max(portfolio_dict, key=portfolio_dict.get)} ({max(portfolio_dict.values()):.2%})
   - Actions représentant plus de 10% du portefeuille: {', '.join([f"{symbol} ({weight:.2%})" for symbol, weight in portfolio_dict.items() if weight > 0.1])}

7. Analyse de performance:
"""
        for stock in portfolio:
            symbol = stock['symbol']
            entry_price = stock['entry_price']
            current_price = stock['current_price']
            performance = (current_price - entry_price) / entry_price
            rapport += f"   - {symbol}: {performance:.2%} (Prix d'entrée: {entry_price:.2f}, Prix actuel: {current_price:.2f})\n"

        rapport += f"""
8. Analyse de corrélation:
   - Corrélation moyenne: {np.mean(correlations):.2f}
   - Corrélation maximale: {np.max(correlations):.2f}
   - Corrélation minimale: {np.min(correlations):.2f}

Conclusion:
Cette analyse de conformité avancée fournit un aperçu détaillé des potentielles violations réglementaires et des zones de risque dans le portefeuille.

Points clés à retenir:
1. {f"Le portefeuille présente {len(violations)} violation(s) majeure(s) qui doivent être adressées immédiatement." if violations else "Aucune violation majeure n'a été détectée."}
2. {f"Il y a {len(warnings)} avertissement(s) qui méritent une attention particulière pour améliorer la conformité et réduire les risques." if warnings else "Aucun avertissement n'a été émis."}
3. L'exposition sectorielle la plus élevée est dans {max(sector_exposure, key=sector_exposure.get)} à {max(sector_exposure.values()):.2%}.
4. L'exposition aux marchés développés est de {market_exposure['Développé']:.2%}, {"ce qui est conforme" if market_exposure['Développé'] >= self.regulations["min_developed_markets_exposure"] else "ce qui est inférieur"} aux recommandations.
5. La volatilité du portefeuille est {"conforme" if volatility != 'Non disponible' and volatility <= self.regulations["max_volatility"] else "supérieure"} aux limites recommandées.

"""
        recommandations = []

        if violations:
            recommandations.append(f"1. Adresser immédiatement les violations en ajustant les positions dans {', '.join([v.split()[3] for v in violations if 'exposition' in v])}.")
        else:
            recommandations.append("1. Maintenir la conformité actuelle du portefeuille.")

        if any(sector in self.regulations["restricted_sectors"] for sector in sector_exposure):
            recommandations.append("2. Réévaluer l'exposition aux secteurs à haut risque et envisager une diversification accrue.")
        else:
            recommandations.append("2. Envisager une diversification dans d'autres secteurs pour réduire le risque sectoriel.")

        if any("ESG" in w for w in warnings):
            recommandations.append("3. Examiner les positions ayant des scores ESG faibles et envisager des alternatives plus durables.")
        else:
            recommandations.append("3. Continuer à surveiller les scores ESG des positions actuelles.")

        if market_exposure['Développé'] < self.regulations["min_developed_markets_exposure"]:
            recommandations.append("4. Considérer une augmentation de l'exposition aux marchés développés pour améliorer la stabilité du portefeuille.")
        else:
            recommandations.append("4. Maintenir l'exposition actuelle aux marchés développés.")

        if volatility != 'Non disponible' and volatility > self.regulations["max_volatility"]:
            recommandations.append("5. Envisager des stratégies de réduction de la volatilité, comme l'ajout d'actifs à faible corrélation.")
        else:
            recommandations.append("5. Continuer à surveiller la volatilité du portefeuille.")

        rapport += "Recommandations:\n"
        for recommendation in recommandations:
            rapport += f"{recommendation}\n"
            return rapport

compliance_agent = ComplianceAgent()
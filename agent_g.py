import yfinance as yf
import pandas as pd
import numpy as np

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

    def get_sector_info(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('sector', 'Non disponible')
        except:
            return 'Non disponible'

    def get_esg_score(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('esgScore', 'Non disponible')
        except:
            return 'Non disponible'

    def get_market_type(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            country = stock.info.get('country', 'Non disponible')
            developed_markets = ['United States', 'Canada', 'United Kingdom', 'Germany', 'France', 'Japan', 'Australia']
            return 'Développé' if country in developed_markets else 'Émergent'
        except:
            return 'Non disponible'

    def calculate_volatility(self, tickers, weights):
        try:
            data = yf.download(tickers, period="1y")['Adj Close']
            returns = data.pct_change().dropna()
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * np.sqrt(252)
            return portfolio_std
        except:
            return 'Non disponible'

    def check_compliance(self, portfolio):
        violations = []
        warnings = []
        portfolio_dict = {stock['symbol']: stock['weight'] / 100 for stock in portfolio}
        
        # Vérification de l'exposition maximale à un seul stock
        for symbol, weight in portfolio_dict.items():
            if weight > self.regulations["max_single_stock_exposure"]:
                violations.append(f"L'exposition à {symbol} ({weight:.2%}) dépasse le maximum autorisé ({self.regulations['max_single_stock_exposure']:.2%})")

        # Vérification du nombre minimum de stocks
        if len(portfolio) < self.regulations["min_stocks_count"]:
            violations.append(f"Le portefeuille ne contient que {len(portfolio)} actions. Le minimum requis est de {self.regulations['min_stocks_count']}")

        # Vérification des secteurs et de l'exposition aux secteurs
        sector_exposure = {}
        for stock in portfolio:
            sector = self.get_sector_info(stock['symbol'])
            if sector in self.regulations["restricted_sectors"]:
                violations.append(f"{stock['symbol']} appartient au secteur restreint: {sector}")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + stock['weight'] / 100

        for sector, exposure in sector_exposure.items():
            if exposure > self.regulations["max_sector_exposure"]:
                violations.append(f"L'exposition au secteur {sector} ({exposure:.2%}) dépasse le maximum autorisé ({self.regulations['max_sector_exposure']:.2%})")

        # Vérification des scores ESG
        for stock in portfolio:
            esg_score = self.get_esg_score(stock['symbol'])
            if esg_score != 'Non disponible' and esg_score < self.regulations["esg_score_threshold"]:
                warnings.append(f"{stock['symbol']} a un score ESG de {esg_score}, inférieur au seuil recommandé de {self.regulations['esg_score_threshold']}")

        # Vérification de l'exposition aux marchés développés
        market_exposure = {'Développé': 0, 'Émergent': 0, 'Non disponible': 0}
        for stock in portfolio:
            market_type = self.get_market_type(stock['symbol'])
            market_exposure[market_type] += stock['weight'] / 100

        if market_exposure['Développé'] < self.regulations["min_developed_markets_exposure"]:
            warnings.append(f"L'exposition aux marchés développés ({market_exposure['Développé']:.2%}) est inférieure au minimum recommandé ({self.regulations['min_developed_markets_exposure']:.2%})")

        # Vérification de la volatilité du portefeuille
        tickers = [stock['symbol'] for stock in portfolio]
        weights = np.array([stock['weight'] / 100 for stock in portfolio])
        volatility = self.calculate_volatility(tickers, weights)
        if volatility != 'Non disponible' and volatility > self.regulations["max_volatility"]:
            warnings.append(f"La volatilité du portefeuille ({volatility:.2%}) dépasse le maximum recommandé ({self.regulations['max_volatility']:.2%})")

        rapport = self.generate_report(violations, warnings, sector_exposure, market_exposure, volatility, portfolio_dict)
        return rapport

    def generate_recommendations(self, violations, sector_exposure, warnings, market_exposure, volatility):
        recommendations = []

        if violations:
            recommendations.append(f"Adresser immédiatement les violations en ajustant les positions dans {', '.join([v.split()[3] for v in violations if 'exposition' in v])}.")
        else:
            recommendations.append("Maintenir la conformité actuelle du portefeuille.")

        if any(sector in self.regulations["restricted_sectors"] for sector in sector_exposure):
            recommendations.append("Réévaluer l'exposition aux secteurs à haut risque et envisager une diversification accrue.")

        if any("ESG" in w for w in warnings):
            recommendations.append("Examiner les positions ayant des scores ESG faibles et envisager des alternatives plus durables.")

        if market_exposure['Développé'] < self.regulations["min_developed_markets_exposure"]:
            recommendations.append("Considérer une augmentation de l'exposition aux marchés développés pour améliorer la stabilité du portefeuille.")

        if volatility != 'Non disponible' and volatility > self.regulations["max_volatility"]:
            recommendations.append("Envisager des stratégies de réduction de la volatilité, comme l'ajout d'actifs à faible corrélation.")

        return recommendations

    def generate_report(self, violations, warnings, sector_exposure, market_exposure, volatility, portfolio_dict):
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

6. Analyse de concentration:
   - Action la plus importante: {max(portfolio_dict, key=portfolio_dict.get)} ({max(portfolio_dict.values()):.2%})
   - Actions représentant plus de 10% du portefeuille: {', '.join([f"{symbol} ({weight:.2%})" for symbol, weight in portfolio_dict.items() if weight > 0.1])}

Conclusion:
Cette analyse de conformité avancée fournit un aperçu détaillé des potentielles violations réglementaires et des zones de risque dans le portefeuille.

Points clés à retenir:
1. {f"Le portefeuille présente {len(violations)} violation(s) majeure(s) qui doivent être adressées immédiatement." if violations else "Aucune violation majeure n'a été détectée."}
2. {f"Il y a {len(warnings)} avertissement(s) qui méritent une attention particulière pour améliorer la conformité et réduire les risques." if warnings else "Aucun avertissement n'a été émis."}
3. L'exposition sectorielle la plus élevée est dans {max(sector_exposure, key=sector_exposure.get)} à {max(sector_exposure.values()):.2%}.
4. L'exposition aux marchés développés est de {market_exposure['Développé']:.2%}, {"ce qui est conforme" if market_exposure['Développé'] >= self.regulations["min_developed_markets_exposure"] else "ce qui est inférieur"} aux recommandations.
5. La volatilité du portefeuille est {"conforme" if volatility != 'Non disponible' and volatility <= self.regulations["max_volatility"] else "supérieure"} aux limites recommandées.
"""
        # Dans la méthode generate_report
        recommendations = self.generate_recommendations(violations, sector_exposure, warnings, market_exposure, volatility)

        rapport += "Recommandations:\n"
        for i, recommendation in enumerate(recommendations, 1):
            rapport += f"{i}. {recommendation}\n"

        rapport += """
Les gestionnaires de portefeuille devraient utiliser ces informations pour ajuster la composition du portefeuille afin d'assurer la conformité réglementaire et d'optimiser le profil risque/rendement en fonction des objectifs d'investissement.
"""
        return rapport

compliance_agent = ComplianceAgent()
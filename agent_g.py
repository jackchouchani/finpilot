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
            return stock.info.get('sector', 'Unknown')
        except:
            return 'Unknown'

    def get_esg_score(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('esgScore', 0)
        except:
            return 0

    def get_market_type(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            country = stock.info.get('country', 'Unknown')
            developed_markets = ['United States', 'Canada', 'United Kingdom', 'Germany', 'France', 'Japan', 'Australia']
            return 'Developed' if country in developed_markets else 'Emerging'
        except:
            return 'Unknown'

    def calculate_volatility(self, tickers, weights):
        data = yf.download(tickers, period="1y")['Adj Close']
        returns = data.pct_change().dropna()
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * np.sqrt(252)
        return portfolio_std

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
            if esg_score < self.regulations["esg_score_threshold"]:
                warnings.append(f"{stock['symbol']} a un score ESG de {esg_score}, inférieur au seuil recommandé de {self.regulations['esg_score_threshold']}")

        # Vérification de l'exposition aux marchés développés
        developed_exposure = sum(stock['weight'] / 100 for stock in portfolio if self.get_market_type(stock['symbol']) == 'Developed')
        if developed_exposure < self.regulations["min_developed_markets_exposure"]:
            warnings.append(f"L'exposition aux marchés développés ({developed_exposure:.2%}) est inférieure au minimum recommandé ({self.regulations['min_developed_markets_exposure']:.2%})")

        # Vérification de la volatilité du portefeuille
        tickers = [stock['symbol'] for stock in portfolio]
        weights = np.array([stock['weight'] / 100 for stock in portfolio])
        volatility = self.calculate_volatility(tickers, weights)
        if volatility > self.regulations["max_volatility"]:
            warnings.append(f"La volatilité du portefeuille ({volatility:.2%}) dépasse le maximum recommandé ({self.regulations['max_volatility']:.2%})")

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
   - Marchés développés: {developed_exposure:.2%}
   - Marchés émergents: {1 - developed_exposure:.2%}

5. Mesures de risque:
   - Volatilité du portefeuille: {volatility:.2%}

Conclusion:
Cette analyse de conformité avancée fournit un aperçu détaillé des potentielles violations réglementaires et des zones de risque dans le portefeuille. 

Points clés à retenir:
1. {f"Le portefeuille présente {len(violations)} violation(s) majeure(s) qui doivent être adressées immédiatement." if violations else "Aucune violation majeure n'a été détectée."}
2. {f"Il y a {len(warnings)} avertissement(s) qui méritent une attention particulière pour améliorer la conformité et réduire les risques." if warnings else "Aucun avertissement n'a été émis."}
3. L'exposition sectorielle la plus élevée est dans {max(sector_exposure, key=sector_exposure.get)} à {max(sector_exposure.values()):.2%}.
4. L'exposition aux marchés développés est de {developed_exposure:.2%}, {"ce qui est conforme" if developed_exposure >= self.regulations["min_developed_markets_exposure"] else "ce qui est inférieur"} aux recommandations.
5. La volatilité du portefeuille est {"conforme" if volatility <= self.regulations["max_volatility"] else "supérieure"} aux limites recommandées.

Recommandations:
1. {f"Adresser immédiatement les violations en ajustant les positions dans {', '.join([v.split()[3] for v in violations if 'exposition' in v])}." if violations else "Maintenir la conformité actuelle du portefeuille."}
2. {"Réévaluer l'exposition aux secteurs à haut risque et envisager une diversification accrue." if any(sector in self.regulations["restricted_sectors"] for sector in sector_exposure) else ""}
3. {"Examiner les positions ayant des scores ESG faibles et envisager des alternatives plus durables." if any("ESG" in w for w in warnings) else ""}
4. {"Considérer une augmentation de l'exposition aux marchés développés pour améliorer la stabilité du portefeuille." if developed_exposure < self.regulations["min_developed_markets_exposure"] else ""}
5. {"Envisager des stratégies de réduction de la volatilité, comme l'ajout d'actifs à faible corrélation." if volatility > self.regulations["max_volatility"] else ""}

Les gestionnaires de portefeuille devraient utiliser ces informations pour ajuster la composition du portefeuille afin d'assurer la conformité réglementaire et d'optimiser le profil risque/rendement en fonction des objectifs d'investissement.
"""
        return rapport

compliance_agent = ComplianceAgent()
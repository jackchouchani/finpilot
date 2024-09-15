class ComplianceAgent:
    def __init__(self):
        self.regulations = {
            "max_single_stock_exposure": 0.30,
            "min_stocks_count": 5,
            "restricted_sectors": ["Tobacco", "Gambling"],
            "esg_score_threshold": 50
        }

    def check_compliance(self, portfolio):
        violations = []
        
         # Convertir le portfolio en dictionnaire pour faciliter la vérification
        portfolio_dict = {stock['symbol']: stock['weight'] for stock in portfolio}
        
        # Vérifier l'exposition maximale à un seul stock
        for stock in portfolio:
            symbol = stock['symbol']
            weight = stock['weight'] / 100  # Convertir le pourcentage en décimal
            if weight > self.regulations["max_single_stock_exposure"]:
                violations.append(f"L'exposition à {symbol} ({weight:.2%}) dépasse le maximum autorisé ({self.regulations['max_single_stock_exposure']:.2%})")

        # Vérifier le nombre minimum de stocks
        if len(portfolio) < self.regulations["min_stocks_count"]:
            violations.append(f"Le portefeuille ne contient que {len(portfolio)} actions. Le minimum requis est de {self.regulations['min_stocks_count']}")

        rapport = f"""
Analyse de conformité

Conformité: {"Conforme" if len(violations) == 0 else "Non conforme"}
Violations:
"""
        if violations:
            for violation in violations:
                rapport += f"- {violation}\n"
        else:
            rapport += "Aucune violation détectée.\n"

        rapport += """
Conclusion:
Cette analyse de conformité fournit un aperçu des violations potentielles dans le portefeuille. Les investisseurs devraient utiliser ces informations pour ajuster leur portefeuille en fonction des réglementations en vigueur.
"""
        return rapport

        # Note: Les vérifications pour les secteurs restreints et les scores ESG nécessiteraient des données supplémentaires
        # que nous n'avons pas dans notre implémentation actuelle. Elles sont omises pour simplifier.

compliance_agent = ComplianceAgent()
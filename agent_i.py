import openai
import yfinance as yf

class InvestmentRecommendationAgent:
    def __init__(self):
        self.client = openai.OpenAI()

    def get_recommendation(self, portfolio, risk_profile):
        # Récupérer les données de base pour le portfolio
        portfolio_data = {}
        for ticker in portfolio:
            stock = yf.Ticker(ticker)
            info = stock.info
            portfolio_data[ticker] = {
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "current_price": info.get('currentPrice', 0),
                "target_price": info.get('targetMeanPrice', 0)
            }

        # Préparer les données pour l'analyse
        portfolio_str = "\n".join([f"{ticker}: {data}" for ticker, data in portfolio_data.items()])
        print(portfolio_data)
        print(portfolio)
        response = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "Vous êtes un conseiller en investissement financier expert. Votre tâche est de fournir des recommandations d'investissement personnalisées basées sur le portefeuille actuel et le profil de risque de l'investisseur."},
                {"role": "system", "content": "Dans votre analyse, veuillez inclure :\n1) Une évaluation de la diversification actuelle du portefeuille.\n2) Des recommandations spécifiques pour optimiser le portefeuille en fonction du profil de risque.\n3) Des suggestions d'actions à acheter, vendre ou conserver, avec des justifications.\n4) Une analyse des secteurs sous-représentés ou surreprésentés.\n5) Des considérations sur l'équilibre entre croissance et valeur.\n6) Des recommandations sur la répartition géographique si pertinent.\n\nAssurez-vous que vos recommandations sont cohérentes avec le profil de risque fourni et basées sur les données financières actuelles."},
                {"role": "user", "content": f"Voici le portefeuille actuel :\n{portfolio_str}\n\nLe profil de risque de l'investisseur est : {risk_profile}\n\nVeuillez fournir des recommandations d'investissement détaillées."}
            ],
            temperature=0.5,
            max_tokens=1500
        )

        return response.choices[0].message.content.strip()

investment_recommendation_agent = InvestmentRecommendationAgent()
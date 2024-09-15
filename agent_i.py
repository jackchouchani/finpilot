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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Vous êtes un conseiller en investissement financier. Fournissez des recommandations d'investissement basées sur le portefeuille et le profil de risque donnés."},
                {"role": "user", "content": f"Sur la base de ce portefeuille:\n{portfolio_str}\n\nEt d'un profil de risque {risk_profile}, fournissez des recommandations d'investissement."}
            ],
            temperature=0.7,
            max_tokens=1200
        )

        return response.choices[0].message.content.strip()

investment_recommendation_agent = InvestmentRecommendationAgent()
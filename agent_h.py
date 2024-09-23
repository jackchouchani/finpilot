import openai
import yfinance as yf

class MarketSentimentAgent:
    def __init__(self):
        self.client = openai.OpenAI()

    def analyze_sentiment(self, ticker, summary="Aucun résumé fourni"):
        # Récupérer les dernières nouvelles sur le ticker
        stock = yf.Ticker(ticker)
        news = stock.news[:5]  # Prendre les 5 dernières nouvelles

        # Analyser le sentiment
        news_texts = [f"Title: {item['title']}\nSummary: {summary}" for item in news]
        combined_text = "\n\n".join(news_texts)

        response = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
            {"role": "system", "content": "Vous êtes un analyste financier expert spécialisé dans l'analyse du sentiment du marché. Votre tâche est d'évaluer le sentiment global pour une action spécifique en vous basant sur les dernières nouvelles. Veuillez fournir :"},
            {"role": "system", "content": "1) Une évaluation du sentiment global (positif, négatif, ou neutre) avec une explication concise.\n2) Les principaux facteurs influençant ce sentiment.\n3) Les implications potentielles à court terme pour le cours de l'action.\n4) Tout risque ou opportunité notable identifié dans les nouvelles.\n\nAssurez-vous que votre analyse est objective, basée sur les faits présentés, et utile pour la prise de décision d'investissement."},
            {"role": "user", "content": f"Analysez le sentiment pour l'action {ticker} basé sur ces nouvelles récentes :\n\n{combined_text}"}
            ],
            temperature=0.5,
            max_tokens=800
        )

        return response.choices[0].message.content.strip()

market_sentiment_agent = MarketSentimentAgent()

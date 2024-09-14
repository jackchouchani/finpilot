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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Vous êtes un analyste de sentiment du marché financier. Analysez les nouvelles suivantes concernant une action et fournissez une analyse globale du sentiment."},
                {"role": "user", "content": f"Analysez le sentiment pour {ticker} basé sur ces nouvelles récentes:\n\n{combined_text}"}
            ],
            temperature=0.7,
            max_tokens=800
        )

        return response.choices[0].message.content.strip()

market_sentiment_agent = MarketSentimentAgent()

from textblob import TextBlob
import requests

class SentimentAnalysisAgent:
    def __init__(self):
        self.news_api_key = "c6cc145ad227419c88756838786b70d1"  # Remplacez par votre clé API

    def get_news(self, query):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.news_api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['articles']
        else:
            return []

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        # La polarité est un float entre -1 et 1, où -1 est très négatif et 1 est très positif
        return analysis.sentiment.polarity

    def analyze(self, company):
        news = self.get_news(company)
        sentiments = [self.analyze_sentiment(article['title'] + ' ' + article['description']) for article in news[:5]]
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        sentiment_category = "Neutre"
        if average_sentiment > 0.1:
            sentiment_category = "Positif"
        elif average_sentiment < -0.1:
            sentiment_category = "Négatif"

        return {
            "entreprise": company,
            "sentiment_moyen": average_sentiment,
            "catégorie_sentiment": sentiment_category,
            "articles_analysés": len(sentiments)
        }

sentiment_agent = SentimentAnalysisAgent()
from textblob import TextBlob
import requests

class SentimentAnalysisAgent:
    def __init__(self):
        self.news_api_key = "c6cc145ad227419c88756838786b70d1"  # Remplacez par votre clé API

    def get_news(self, query):
        url = f"https://newsapi.org/v2/everything"
        params = {
            "q": f'"{query}" AND (stock OR market OR finance OR investor)',
            "language": "fr,en",
            "sortBy": "relevancy",
            "pageSize": 20,
            "domains": "reuters.com,bloomberg.com,ft.com,lesechos.fr,boursorama.com",
            "apiKey": self.news_api_key
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()['articles']
        else:
            return []

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def analyze(self, company):
        news = self.get_news(company)
        sentiments = [self.analyze_sentiment(article['title'] + ' ' + article['description']) for article in news[:10]]
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        if average_sentiment > 0.1:
            sentiment_category = "Positif"
        elif average_sentiment < -0.1:
            sentiment_category = "Négatif"
        else:
            sentiment_category = "Neutre"

        rapport = f"""
Analyse de sentiment pour {company}

Nombre d'articles analysés: {len(sentiments)}
Sentiment moyen: {average_sentiment:.2f} (sur une échelle de -1 à 1)
Catégorie de sentiment: {sentiment_category}

Interprétation:
{f'Le sentiment général concernant {company} est {sentiment_category.lower()}. ' if sentiment_category != "Neutre" else f'Le sentiment général concernant {company} est neutre. '}
{'Cela pourrait indiquer une perception positive de l\'entreprise dans les médias récents.' if sentiment_category == "Positif" else 'Cela pourrait indiquer une perception négative de l\'entreprise dans les médias récents.' if sentiment_category == "Négatif" else 'Cela suggère que les opinions sont mitigées ou que les nouvelles récentes n\'ont pas eu d\'impact significatif sur la perception de l\'entreprise.'}

Articles récents analysés:
"""
        for i, article in enumerate(news[:5], 1):
            rapport += f"{i}. {article['title']}\n   Sentiment: {self.analyze_sentiment(article['title'] + ' ' + article['description']):.2f}\n\n"

        rapport += f"""
Conclusion:
Cette analyse de sentiment fournit un aperçu de la perception actuelle de {company} dans les médias. Les investisseurs devraient utiliser cette information en conjonction avec une analyse financière approfondie et une compréhension plus large du contexte de l'entreprise et de son secteur avant de prendre des décisions d'investissement.
"""
        return rapport

sentiment_agent = SentimentAnalysisAgent()
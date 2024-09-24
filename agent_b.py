from textblob import TextBlob
import requests
import datetime

class SentimentAnalysisAgent:
    def __init__(self):
        self.news_api_key = "c6cc145ad227419c88756838786b70d1"

    def get_news(self, query):
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": yesterday.isoformat(),
            "to": yesterday.isoformat(),
            "sortBy": "popularity",
            "language": "en",
            "apiKey": self.news_api_key
        }
        response = requests.get(url, params=params)
        print(f"Statut de la réponse: {response.status_code}")
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            print(f"Nombre d'articles reçus: {len(articles)}")
            return articles
        else:
            print(f"Erreur lors de la requête: {response.text}")
            return []

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def analyze(self, company):
        news = self.get_news(company)
        if not news:
            return "Analyse impossible : aucun article trouvé."

        sentiments = []
        for article in news[:10]:
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}".strip()
            if text:
                sentiments.append(self.analyze_sentiment(text))

        if not sentiments:
            return "Analyse impossible : aucun texte valide trouvé dans les articles."

        average_sentiment = sum(sentiments) / len(sentiments)

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
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}".strip()
            sentiment = self.analyze_sentiment(text) if text else 0
            rapport += f"{i}. {title}\n Sentiment: {sentiment:.2f}\n\n"

        rapport += f"""
Conclusion:
Cette analyse de sentiment fournit un aperçu de la perception actuelle de {company} dans les médias. Les investisseurs devraient utiliser cette information en conjonction avec une analyse financière approfondie et une compréhension plus large du contexte de l'entreprise et de son secteur avant de prendre des décisions d'investissement.
"""
        return rapport

sentiment_agent = SentimentAnalysisAgent()
import requests
import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np

# Télécharger les ressources nécessaires pour NLTK
nltk.download('vader_lexicon', quiet=True)

class SentimentAnalysisAgent:
    def __init__(self):
        self.news_api_key = "c6cc145ad227419c88756838786b70d1"
        self.sia = SentimentIntensityAnalyzer()

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
            "pageSize": 100,
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
        if not text:
            return 0
        return self.sia.polarity_scores(text)['compound']

    def get_article_text(self, article):
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        return ' '.join(filter(None, [title, description, content]))

    def analyze(self, company):
        news = self.get_news(company)
        if not news:
            return "Analyse impossible : aucun article trouvé."

        sentiments = []
        for article in news:
            text = self.get_article_text(article)
            if text:
                sentiments.append(self.analyze_sentiment(text))

        if not sentiments:
            return "Analyse impossible : aucun texte valide trouvé dans les articles."

        average_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)

        if average_sentiment > 0.05:
            sentiment_category = "Positif"
        elif average_sentiment < -0.05:
            sentiment_category = "Négatif"
        else:
            sentiment_category = "Neutre"

        rapport = f"""
Analyse de sentiment pour {company}
Nombre d'articles analysés: {len(sentiments)}
Sentiment moyen: {average_sentiment:.2f} (sur une échelle de -1 à 1)
Écart type du sentiment: {sentiment_std:.2f}
Catégorie de sentiment: {sentiment_category}

Interprétation:
Le sentiment général concernant {company} est {sentiment_category.lower()}. 
{self.get_interpretation(average_sentiment, sentiment_std)}

Articles récents analysés:
"""
        sorted_articles = sorted(news, key=lambda x: abs(self.analyze_sentiment(self.get_article_text(x))), reverse=True)
        for i, article in enumerate(sorted_articles[:10], 1):
            text = self.get_article_text(article)
            sentiment = self.analyze_sentiment(text)
            title = article.get('title', 'Titre non disponible')
            rapport += f"{i}. {title}\n Sentiment: {sentiment:.2f}\n\n"

        rapport += f"""
Conclusion:
Cette analyse de sentiment offre un aperçu de la perception actuelle de {company} dans les médias. 
L'écart type du sentiment ({sentiment_std:.2f}) indique {'une grande variabilité' if sentiment_std > 0.5 else 'une relative cohérence'} dans les opinions exprimées.
Les investisseurs devraient utiliser cette information en conjonction avec une analyse financière approfondie 
et une compréhension plus large du contexte de l'entreprise et de son secteur avant de prendre des décisions d'investissement.
"""
        return rapport

    def get_interpretation(self, avg_sentiment, std_sentiment):
        if avg_sentiment > 0.2:
            strength = "très positive"
        elif avg_sentiment > 0.05:
            strength = "plutôt positive"
        elif avg_sentiment < -0.2:
            strength = "très négative"
        elif avg_sentiment < -0.05:
            strength = "plutôt négative"
        else:
            strength = "neutre"

        if std_sentiment > 0.5:
            consistency = "Les opinions sont très variées, ce qui suggère une situation complexe ou controversée."
        elif std_sentiment > 0.3:
            consistency = "Il y a une certaine diversité d'opinions, mais une tendance générale se dégage."
        else:
            consistency = "Les opinions sont relativement cohérentes."

        return f"La perception de l'entreprise semble être {strength}. {consistency}"

sentiment_agent = SentimentAnalysisAgent()
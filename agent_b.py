import requests
from textblob import TextBlob
from collections import Counter
import re

class SentimentAnalysisAgent:
    def __init__(self):
        self.news_api_key = "c6cc145ad227419c88756838786b70d1"

    def get_news(self, query):
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 50,
            "apiKey": self.news_api_key
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get('articles', [])
        else:
            print(f"Erreur lors de la requête: {response.text}")
            return []

    def analyze_sentiment(self, text):
        return TextBlob(text).sentiment.polarity

    def extract_keywords(self, text):
        words = re.findall(r'\w+', text.lower())
        return Counter(words).most_common(10)

    def analyze(self, company):
        news = self.get_news(company)
        if not news:
            return "Analyse impossible : aucun article trouvé."

        sentiments = []
        keywords = Counter()

        for article in news:
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
            keywords.update(dict(self.extract_keywords(text)))

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

        if avg_sentiment > 0.1:
            sentiment_category = "Positif"
        elif avg_sentiment < -0.1:
            sentiment_category = "Négatif"
        else:
            sentiment_category = "Neutre"

        rapport = f"""
Analyse de sentiment pour {company}

1. Aperçu général:
   Nombre d'articles analysés: {len(sentiments)}
   Sentiment moyen: {avg_sentiment:.2f} (sur une échelle de -1 à 1)
   Catégorie de sentiment: {sentiment_category}

2. Interprétation:
   {self.get_interpretation(avg_sentiment, company)}

3. Mots-clés fréquents:
   {', '.join([f"{word} ({count})" for word, count in keywords.most_common(5)])}

4. Articles représentatifs:
"""
        sorted_articles = sorted(news, key=lambda x: abs(self.analyze_sentiment(x['title'])), reverse=True)
        for i, article in enumerate(sorted_articles[:3], 1):
            title = article.get('title', 'Titre non disponible')
            sentiment = self.analyze_sentiment(title)
            rapport += f"   {i}. {title}\n      Sentiment: {sentiment:.2f}\n      Source: {article.get('source', {}).get('name', 'Non spécifiée')}\n\n"

        rapport += f"""
5. Analyse approfondie:
   a) Tendance générale: Le sentiment global est {sentiment_category.lower()}, suggérant une perception {'favorable' if avg_sentiment > 0 else 'défavorable' if avg_sentiment < 0 else 'mitigée'} de {company}.
   b) Contexte: Les mots-clés fréquents indiquent que les discussions autour de {company} se concentrent sur {', '.join([word for word, _ in keywords.most_common(3)])}.
   c) Implications: {'Une perception positive pourrait indiquer des opportunités de croissance ou des développements favorables.' if avg_sentiment > 0 else 'Une perception négative pourrait signaler des défis ou des controverses à surveiller.' if avg_sentiment < 0 else 'Une perception neutre suggère un équilibre entre les aspects positifs et négatifs.'}

6. Recommandations:
   - {'Capitaliser sur le sentiment positif pour renforcer la position de l\'entreprise.' if avg_sentiment > 0 else 'Adresser les préoccupations soulevées pour améliorer la perception de l\'entreprise.' if avg_sentiment < 0 else 'Surveiller de près les développements futurs pour détecter tout changement de perception.'}
   - Approfondir l'analyse des thèmes liés à {', '.join([word for word, _ in keywords.most_common(2)])} pour mieux comprendre leur impact sur la perception de {company}.
   - Continuer à suivre l'évolution du sentiment dans le temps pour identifier les tendances à long terme.

Cette analyse offre un aperçu de la perception actuelle de {company} basée sur les articles récents.
Les investisseurs et parties prenantes devraient utiliser ces informations en conjonction avec d'autres
analyses financières et sectorielles pour une compréhension complète de la situation de l'entreprise.
"""
        return rapport

    def get_interpretation(self, avg_sentiment, company):
        if avg_sentiment > 0.3:
            strength = "très positive"
        elif avg_sentiment > 0.1:
            strength = "plutôt positive"
        elif avg_sentiment < -0.3:
            strength = "très négative"
        elif avg_sentiment < -0.1:
            strength = "plutôt négative"
        else:
            strength = "neutre"

        return f"La perception de {company} semble être {strength}. Cela pourrait être influencé par des événements récents, des annonces de l'entreprise, ou des tendances plus larges du marché."

sentiment_agent = SentimentAnalysisAgent()
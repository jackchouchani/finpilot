import requests
import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import numpy as np

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class SentimentAnalysisAgent:
    def __init__(self):
        self.news_api_key = "c6cc145ad227419c88756838786b70d1"
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def get_news(self, query):
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": yesterday.isoformat(),
            "to": yesterday.isoformat(),
            "sortBy": "relevancy",
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

    def extract_keywords(self, text):
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        return Counter(words).most_common(5)

    def analyze(self, company):
        news = self.get_news(company)
        if not news:
            return "Analyse impossible : aucun article trouvé."

        sentiments = []
        keywords = Counter()
        unique_articles = set()

        for article in news:
            text = self.get_article_text(article)
            if text and text not in unique_articles:
                unique_articles.add(text)
                sentiment = self.analyze_sentiment(text)
                sentiments.append(sentiment)
                keywords.update(dict(self.extract_keywords(text)))

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
Nombre d'articles uniques analysés: {len(sentiments)}
Sentiment moyen: {average_sentiment:.2f} (sur une échelle de -1 à 1)
Écart type du sentiment: {sentiment_std:.2f}
Catégorie de sentiment: {sentiment_category}

Interprétation:
{self.get_interpretation(average_sentiment, sentiment_std)}

Mots-clés les plus fréquents:
{', '.join([f"{word} ({count})" for word, count in keywords.most_common(10)])}

Articles récents analysés:
"""
        sorted_articles = sorted(news, key=lambda x: abs(self.analyze_sentiment(self.get_article_text(x))), reverse=True)
        for i, article in enumerate(sorted_articles[:5], 1):
            text = self.get_article_text(article)
            sentiment = self.analyze_sentiment(text)
            title = article.get('title', 'Titre non disponible')
            rapport += f"{i}. {title}\n Sentiment: {sentiment:.2f}\n Source: {article.get('source', {}).get('name', 'Non spécifiée')}\n\n"

        rapport += f"""
Conclusion:
Cette analyse offre un aperçu de la perception actuelle de {company} basée sur {len(sentiments)} articles uniques.
Le sentiment général est {sentiment_category.lower()} avec une moyenne de {average_sentiment:.2f}.
L'écart type de {sentiment_std:.2f} indique {'une grande variabilité' if sentiment_std > 0.3 else 'une relative cohérence'} dans les opinions.
Les mots-clés fréquents suggèrent que les discussions autour de {company} se concentrent sur {', '.join([word for word, _ in keywords.most_common(3)])}.
Les investisseurs devraient compléter cette analyse avec des recherches supplémentaires et une compréhension approfondie du secteur.
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

        if std_sentiment > 0.4:
            consistency = "Les opinions sont très variées, suggérant une situation complexe ou controversée."
        elif std_sentiment > 0.2:
            consistency = "Il y a une diversité d'opinions, mais une tendance générale se dégage."
        else:
            consistency = "Les opinions sont relativement cohérentes."

        return f"La perception de {company} semble être {strength}. {consistency} Cette perception pourrait être influencée par des événements récents ou des tendances du marché."

sentiment_agent = SentimentAnalysisAgent()

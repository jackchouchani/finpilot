import requests
from textblob import TextBlob
from collections import Counter
import re
from datetime import datetime, timedelta

class SentimentAnalysisAgent:
    def __init__(self):
        self.news_api_key = "c6cc145ad227419c88756838786b70d1"
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

    def get_news(self, query, days=7):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 100,
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
        return [word for word in words if word not in self.stop_words and len(word) > 2]

    def analyze(self, company):
        news = self.get_news(company)
        if not news:
            return "Analyse impossible : aucun article trouvé."

        sentiments = []
        keywords = Counter()
        sources = Counter()

        for article in news:
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
            keywords.update(self.extract_keywords(text))
            sources[article.get('source', {}).get('name', 'Unknown')] += 1

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        positive_articles = sum(1 for s in sentiments if s > 0.1)
        negative_articles = sum(1 for s in sentiments if s < -0.1)
        neutral_articles = len(sentiments) - positive_articles - negative_articles

        if avg_sentiment > 0.1:
            sentiment_category = "Positif"
        elif avg_sentiment < -0.1:
            sentiment_category = "Négatif"
        else:
            sentiment_category = "Neutre"

        rapport = f"""
Analyse de sentiment pour {company}

1. Aperçu général:
   Période analysée: 7 derniers jours
   Nombre d'articles analysés: {len(sentiments)}
   Sentiment moyen: {avg_sentiment:.2f} (sur une échelle de -1 à 1)
   Catégorie de sentiment: {sentiment_category}
   Distribution du sentiment:
     - Articles positifs: {positive_articles} ({positive_articles/len(sentiments)*100:.1f}%)
     - Articles neutres: {neutral_articles} ({neutral_articles/len(sentiments)*100:.1f}%)
     - Articles négatifs: {negative_articles} ({negative_articles/len(sentiments)*100:.1f}%)

2. Interprétation:
   {self.get_interpretation(avg_sentiment, company)}

3. Mots-clés fréquents:
   {', '.join([f"{word} ({count})" for word, count in keywords.most_common(10)])}

4. Thèmes principaux:
   {self.identify_themes(keywords)}

5. Principales sources d'information:
   {', '.join([f"{source} ({count})" for source, count in sources.most_common(5)])}

6. Articles représentatifs:
"""
        sorted_articles = sorted(news, key=lambda x: abs(self.analyze_sentiment(x['title'])), reverse=True)
        for i, article in enumerate(sorted_articles[:3], 1):
            title = article.get('title', 'Titre non disponible')
            sentiment = self.analyze_sentiment(title)
            rapport += f"   {i}. {title}\n      Sentiment: {sentiment:.2f}\n      Source: {article.get('source', {}).get('name', 'Non spécifiée')}\n      Date: {article.get('publishedAt', 'Non spécifiée')}\n\n"

        rapport += f"""
7. Analyse approfondie:
   a) Tendance générale: Le sentiment global est {sentiment_category.lower()}, suggérant une perception {'favorable' if avg_sentiment > 0 else 'défavorable' if avg_sentiment < 0 else 'mitigée'} de {company}.
   b) Contexte: Les discussions autour de {company} se concentrent principalement sur {', '.join([word for word, _ in keywords.most_common(3)])}.
   c) Évolution: {self.sentiment_evolution(sentiments)}
   d) Implications: {'Une perception positive pourrait indiquer des opportunités de croissance ou des développements favorables.' if avg_sentiment > 0 else 'Une perception négative pourrait signaler des défis ou des controverses à surveiller.' if avg_sentiment < 0 else 'Une perception neutre suggère un équilibre entre les aspects positifs et négatifs.'}

8. Recommandations:
   - {'Capitaliser sur le sentiment positif pour renforcer la position de l\'entreprise.' if avg_sentiment > 0 else 'Adresser les préoccupations soulevées pour améliorer la perception de l\'entreprise.' if avg_sentiment < 0 else 'Surveiller de près les développements futurs pour détecter tout changement de perception.'}
   - Approfondir l'analyse des thèmes identifiés, en particulier {', '.join([theme for theme, _ in self.identify_themes(keywords)[:2]])}.
   - Diversifier les sources d'information pour obtenir une vue plus complète de la perception de l'entreprise.
   - Continuer à suivre l'évolution du sentiment dans le temps pour identifier les tendances à long terme et les facteurs influençant la perception de {company}.

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

    def identify_themes(self, keywords):
        themes = [
            ("Innovation", ["innovation", "technology", "new", "launch"]),
            ("Financial Performance", ["revenue", "profit", "earnings", "growth"]),
            ("Market Position", ["market", "share", "competition", "leader"]),
            ("Product", ["iphone", "mac", "ipad", "watch"]),
            ("Services", ["services", "cloud", "streaming", "subscription"])
        ]
        theme_scores = {}
        for theme, theme_keywords in themes:
            score = sum(keywords.get(kw, 0) for kw in theme_keywords)
            if score > 0:
                theme_scores[theme] = score
        return sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)

    def sentiment_evolution(self, sentiments):
        first_half = sentiments[:len(sentiments)//2]
        second_half = sentiments[len(sentiments)//2:]
        avg_first = sum(first_half) / len(first_half) if first_half else 0
        avg_second = sum(second_half) / len(second_half) if second_half else 0
        
        if avg_second > avg_first + 0.1:
            return "Le sentiment semble s'améliorer au fil du temps."
        elif avg_second < avg_first - 0.1:
            return "Le sentiment semble se détériorer au fil du temps."
        else:
            return "Le sentiment reste relativement stable sur la période analysée."

sentiment_agent = SentimentAnalysisAgent()

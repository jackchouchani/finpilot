import requests
from textblob import TextBlob
from collections import Counter, defaultdict
import re
from datetime import datetime, timedelta

class SentimentAnalysisAgent:
    def __init__(self):
        self.news_api_key = "d3f7b02481f44c549997455125444661"
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

    def categorize_keywords(self, keywords):
        categories = {
            "Produits": ["iphone", "mac", "ipad", "watch"],
            "Financier": ["revenue", "profit", "earnings", "shares", "stock"],
            "Marché": ["market", "competition", "industry", "consumer"],
            "Innovation": ["technology", "innovation", "development", "research"],
            "Services": ["services", "cloud", "streaming", "subscription"]
        }
        categorized = defaultdict(list)
        for word, count in keywords.items():
            for category, terms in categories.items():
                if word in terms:
                    categorized[category].append((word, count))
                    break
            else:
                categorized["Autres"].append((word, count))
        return dict(categorized)

    def analyze(self, company):
        news = self.get_news(company)
        if not news:
            return "Analyse impossible : aucun article trouvé."

        sentiments = []
        keywords = Counter()
        sources = Counter()
        theme_sentiments = defaultdict(list)

        for article in news:
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
            # Correction : utilisez update() avec un Counter créé à partir de la liste
            keywords.update(Counter(self.extract_keywords(text)))
            sources[article.get('source', {}).get('name', 'Unknown')] += 1
            
            # Analyse de sentiment par thème
            for theme, theme_keywords in self.themes.items():
                if any(keyword in text.lower() for keyword in theme_keywords):
                    theme_sentiments[theme].append(sentiment)

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        positive_articles = sum(1 for s in sentiments if s > 0.1)
        negative_articles = sum(1 for s in sentiments if s < -0.1)
        neutral_articles = len(sentiments) - positive_articles - negative_articles

        categorized_keywords = self.categorize_keywords(keywords)

        rapport = f"""
Analyse de sentiment pour {company}

1. Aperçu général:
   Période analysée: 7 derniers jours
   Nombre d'articles analysés: {len(sentiments)}
   Sentiment moyen: {avg_sentiment:.2f} (sur une échelle de -1 à 1)
   Catégorie de sentiment: {"Positif" if avg_sentiment > 0.1 else "Négatif" if avg_sentiment < -0.1 else "Neutre"}
   Distribution du sentiment:
     - Articles positifs: {positive_articles} ({positive_articles/len(sentiments)*100:.1f}%)
     - Articles neutres: {neutral_articles} ({neutral_articles/len(sentiments)*100:.1f}%)
     - Articles négatifs: {negative_articles} ({negative_articles/len(sentiments)*100:.1f}%)

2. Interprétation:
   {self.get_interpretation(avg_sentiment, company)}

3. Mots-clés fréquents par catégorie:
"""
        for category, words in categorized_keywords.items():
            rapport += f"   {category}: {', '.join([f'{word} ({count})' for word, count in words[:5]])}\n"

        rapport += f"""
4. Thèmes principaux et leur sentiment:
"""
        for theme, sentiments in theme_sentiments.items():
            avg_theme_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            rapport += f"   {theme}: Sentiment moyen {avg_theme_sentiment:.2f}, Mentions: {len(sentiments)}\n"

        rapport += f"""
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
   a) Tendance générale: Le sentiment global est {"positif" if avg_sentiment > 0 else "négatif" if avg_sentiment < 0 else "neutre"}, suggérant une perception {"favorable" if avg_sentiment > 0 else "défavorable" if avg_sentiment < 0 else "mitigée"} de {company}.
   b) Contexte: Les discussions autour de {company} se concentrent principalement sur {', '.join([cat for cat, words in categorized_keywords.items() if words][:3])}.
   c) Évolution: {self.sentiment_evolution(sentiments)}
   d) Implications: {self.get_implications(avg_sentiment, theme_sentiments)}

8. Recommandations:
   {self.get_recommendations(avg_sentiment, categorized_keywords, theme_sentiments, company)}

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

    def get_implications(self, avg_sentiment, theme_sentiments):
        implications = []
        if avg_sentiment > 0.1:
            implications.append("Une perception positive pourrait indiquer des opportunités de croissance ou des développements favorables.")
        elif avg_sentiment < -0.1:
            implications.append("Une perception négative pourrait signaler des défis ou des controverses à surveiller.")
        else:
            implications.append("Une perception neutre suggère un équilibre entre les aspects positifs et négatifs.")

        for theme, sentiments in theme_sentiments.items():
            avg_theme_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            if avg_theme_sentiment > 0.2:
                implications.append(f"Le thème '{theme}' est perçu très positivement, ce qui pourrait être un point fort pour l'entreprise.")
            elif avg_theme_sentiment < -0.2:
                implications.append(f"Le thème '{theme}' est perçu négativement, ce qui pourrait nécessiter une attention particulière.")

        return " ".join(implications)

    def get_recommendations(self, avg_sentiment, categorized_keywords, theme_sentiments, company):
        recommendations = []
        if avg_sentiment > 0.1:
            recommendations.append(f"Capitaliser sur le sentiment positif pour renforcer la position de {company}.")
        elif avg_sentiment < -0.1:
            recommendations.append(f"Adresser les préoccupations soulevées pour améliorer la perception de {company}.")
        else:
            recommendations.append(f"Surveiller de près les développements futurs pour détecter tout changement de perception de {company}.")

        top_categories = [cat for cat, words in categorized_keywords.items() if words][:2]
        recommendations.append(f"Approfondir l'analyse des catégories {' et '.join(top_categories)}.")

        negative_themes = [theme for theme, sentiments in theme_sentiments.items() if sum(sentiments) / len(sentiments) < -0.1]
        if negative_themes:
            recommendations.append(f"Porter une attention particulière aux thèmes perçus négativement : {', '.join(negative_themes)}.")

        recommendations.append("Diversifier les sources d'information pour obtenir une vue plus complète de la perception de l'entreprise.")
        recommendations.append(f"Continuer à suivre l'évolution du sentiment dans le temps pour identifier les tendances à long terme et les facteurs influençant la perception de {company}.")

        return "\n   - ".join(recommendations)

    themes = {
        "Innovation": ["innovation", "technology", "new", "launch"],
        "Financial Performance": ["revenue", "profit", "earnings", "growth"],
        "Market Position": ["market", "share", "competition", "leader"],
        "Product": ["iphone", "mac", "ipad", "watch"],
        "Services": ["services", "cloud", "streaming", "subscription"]
    }

sentiment_agent = SentimentAnalysisAgent()

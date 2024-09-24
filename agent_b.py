import requests
import datetime
from collections import Counter
import numpy as np
from transformers import pipeline
import spacy
import yake

class SentimentAnalysisAgent:
    def __init__(self):
        self.news_api_key = "c6cc145ad227419c88756838786b70d1"
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.nlp = spacy.load("en_core_web_sm")
        self.keyword_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=20, features=None)

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
        result = self.sentiment_analyzer(text[:512])[0]
        score = result['score']
        if result['label'] == 'NEGATIVE':
            score = -score
        return score

    def get_article_text(self, article):
        title = article.get('title', '')
        description = article.get('description', '')
        content = article.get('content', '')
        return ' '.join(filter(None, [title, description, content]))

    def extract_keywords(self, text):
        keywords = self.keyword_extractor.extract_keywords(text)
        return [kw for kw, _ in keywords]

    def extract_named_entities(self, text):
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PERSON', 'PRODUCT']]

    def analyze(self, company):
        news = self.get_news(company)
        if not news:
            return "Analyse impossible : aucun article trouvé."

        sentiments = []
        keywords = Counter()
        named_entities = Counter()
        unique_articles = set()

        for article in news:
            text = self.get_article_text(article)
            if text and text not in unique_articles:
                unique_articles.add(text)
                sentiment = self.analyze_sentiment(text)
                sentiments.append(sentiment)
                keywords.update(self.extract_keywords(text))
                named_entities.update(self.extract_named_entities(text))

        if not sentiments:
            return "Analyse impossible : aucun texte valide trouvé dans les articles."

        average_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)

        if average_sentiment > 0.1:
            sentiment_category = "Positif"
        elif average_sentiment < -0.1:
            sentiment_category = "Négatif"
        else:
            sentiment_category = "Neutre"

        rapport = f"""
Analyse de sentiment approfondie pour {company}
Nombre d'articles uniques analysés: {len(sentiments)}
Sentiment moyen: {average_sentiment:.2f} (sur une échelle de -1 à 1)
Écart type du sentiment: {sentiment_std:.2f}
Catégorie de sentiment: {sentiment_category}

Interprétation:
{self.get_interpretation(average_sentiment, sentiment_std, company)}

Mots-clés les plus pertinents:
{', '.join(keywords.keys())[:150]}

Entités nommées fréquemment mentionnées:
{', '.join([f"{entity} ({count})" for entity, count in named_entities.most_common(5)])}

Articles récents analysés:
"""
        sorted_articles = sorted(news, key=lambda x: abs(self.analyze_sentiment(self.get_article_text(x))), reverse=True)
        for i, article in enumerate(sorted_articles[:5], 1):
            text = self.get_article_text(article)
            sentiment = self.analyze_sentiment(text)
            title = article.get('title', 'Titre non disponible')
            rapport += f"{i}. {title}\n Sentiment: {sentiment:.2f}\n Source: {article.get('source', {}).get('name', 'Non spécifiée')}\n\n"

        rapport += f"""
Analyse approfondie:
1. Tendance générale: Le sentiment global est {sentiment_category.lower()}, avec une moyenne de {average_sentiment:.2f}.
   L'écart type de {sentiment_std:.2f} indique {'une grande variabilité' if sentiment_std > 0.3 else 'une relative cohérence'} dans les opinions.

2. Contexte: Les mots-clés et entités nommées suggèrent que les discussions autour de {company} se concentrent sur:
   - {', '.join(keywords.keys()[:3])}
   - Personnes/Organisations mentionnées: {', '.join([entity for entity, _ in named_entities.most_common(3)])}

3. Implications potentielles:
   {'- Les opinions positives pourraient indiquer des opportunités de croissance ou des lancements de produits réussis.' if average_sentiment > 0 else '- Les sentiments négatifs pourraient signaler des défis ou des controverses à surveiller.'}
   - La {'diversité' if sentiment_std > 0.3 else 'cohérence'} des opinions suggère {'un sujet complexe ou controversé' if sentiment_std > 0.3 else 'une perception relativement stable de l\'entreprise'}.

4. Recommandations:
   - Approfondir l'analyse des {', '.join(keywords.keys()[:2])} pour comprendre leur impact sur {company}.
   - Surveiller les développements liés à {', '.join([entity for entity, _ in named_entities.most_common(2)])}.
   - {'Capitaliser sur le sentiment positif pour renforcer la position de l\'entreprise.' if average_sentiment > 0 else 'Adresser les préoccupations soulevées pour améliorer la perception de l\'entreprise.'}

Cette analyse offre un aperçu détaillé de la perception actuelle de {company}. Les investisseurs et parties prenantes
devraient compléter ces informations avec une analyse financière approfondie et une compréhension globale du secteur
avant de prendre des décisions stratégiques.
"""
        return rapport

    def get_interpretation(self, avg_sentiment, std_sentiment, company):
        if avg_sentiment > 0.3:
            strength = "extrêmement positive"
        elif avg_sentiment > 0.1:
            strength = "très positive"
        elif avg_sentiment > 0:
            strength = "légèrement positive"
        elif avg_sentiment < -0.3:
            strength = "extrêmement négative"
        elif avg_sentiment < -0.1:
            strength = "très négative"
        elif avg_sentiment < 0:
            strength = "légèrement négative"
        else:
            strength = "neutre"

        if std_sentiment > 0.4:
            consistency = "Les opinions sont très variées, suggérant une situation complexe ou controversée."
        elif std_sentiment > 0.2:
            consistency = "Il y a une diversité notable d'opinions, indiquant des perspectives différentes sur l'entreprise."
        else:
            consistency = "Les opinions sont relativement cohérentes, suggérant un consensus général."

        return f"La perception de {company} apparaît comme étant {strength}. {consistency} Cette perception pourrait être influencée par des événements récents, des annonces de l'entreprise, ou des tendances plus larges du marché et de l'industrie."

sentiment_agent = SentimentAnalysisAgent()
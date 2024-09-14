import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import openai
import os
from dotenv import load_dotenv

nltk.download('punkt')
nltk.download('stopwords')

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY') 

class DocumentAnalysisAgent:
    def __init__(self):
        self.client = openai.OpenAI()

    def analyze(self, text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.casefold() not in stop_words]

        sentences = sent_tokenize(text)
        key_sentences = sentences[:3]

        # Utiliser ChatGPT pour le résumé
        summary = self.get_gpt_summary(text)

        return {
            "key_words": filtered_text[:10],
            "key_sentences": key_sentences,
            "summary": summary
        }

    def get_gpt_summary(self, text):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Résumez le texte suivant en 2-3 phrases."},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=100,
            top_p=1
        )
        return response.choices[0].message.content.strip()
    
    def analyze_financial_report(self, text):
        summary = self.get_gpt_summary(text)
        key_metrics = self.extract_key_metrics(text)
        return {
            "summary": summary,
            "key_metrics": key_metrics
        }

    def extract_key_metrics(self, text):
        prompt = "Extrayez les indicateurs clés suivants du rapport financier : chiffre d'affaires, bénéfice net, EBITDA, et liste des principaux risques. Présentez les résultats au format JSON."
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Vous êtes un analyste financier."},
                {"role": "user", "content": prompt + "\n\nRapport:\n" + text}
            ]
        )
        return json.loads(response.choices[0].message.content)

document_agent = DocumentAnalysisAgent()
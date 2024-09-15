# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
from pydantic import BaseModel
from typing import List
import openai
import os
import json
# from dotenv import load_dotenv

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')


openai.api_key = os.getenv('OPENAI_API_KEY') 

class PhraseCle(BaseModel):
    phrase: str

class AnalyseDocument(BaseModel):
    mots_cles: List[str]
    phrases_cles: List[PhraseCle]
    resume: str

class MetriquesFinancieres(BaseModel):
    chiffre_affaires: float
    benefice_net: float
    ebitda: float
    risques: List[str]

class RapportFinancier(BaseModel):
    resume: str
    metriques_cles: MetriquesFinancieres

class AgentAnalyseDocument:
    def __init__(self):
        self.client = openai.OpenAI()

    def analyser(self, texte: str) -> AnalyseDocument:
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "Analysez le texte suivant et extrayez les informations clés en français."},
                    {"role": "user", "content": texte}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1000
            )
            
            resultat = json.loads(completion.choices[0].message.content)
            return AnalyseDocument(**resultat)
        except Exception as e:
            print(f"Erreur dans analyser: {str(e)}")
            return AnalyseDocument(mots_cles=[], phrases_cles=[], resume="Une erreur s'est produite lors de l'analyse.")

    def analyser_rapport_financier(self, texte: str) -> RapportFinancier:
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "Vous êtes un analyste financier. Extrayez les informations financières clés du rapport donné en français."},
                    {"role": "user", "content": texte}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1000
            )
            
            resultat = json.loads(completion.choices[0].message.content)
            return RapportFinancier(**resultat)
        except Exception as e:
            print(f"Erreur dans analyser_rapport_financier: {str(e)}")
            return RapportFinancier(
                resume="Une erreur s'est produite lors de l'analyse.",
                metriques_cles=MetriquesFinancieres(chiffre_affaires=0, benefice_net=0, ebitda=0, risques=[])
            )

document_agent = AgentAnalyseDocument()
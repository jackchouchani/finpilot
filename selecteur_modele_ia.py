
from openai import OpenAI
import anthropic

class SelecteurModeleIA:
    def __init__(self):
        self.client_openai = OpenAI()
        self.client_anthropic = anthropic.Anthropic()

    def selectionner_modele(self, complexite_tache):
        if complexite_tache == "simple":
            return self.client_openai, "gpt-4o-mini"
        elif complexite_tache == "complexe":
            return self.client_anthropic, "claude-3-5-sonnet"
        else:
            return self.client_openai, "gpt-4o"
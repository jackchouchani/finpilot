import openai

class UserProfileAgent:
    def __init__(self):
        self.client = openai.OpenAI()

    def analyze_user_profile(self, user_id, portfolio, chat_history):
        # Vérifier et préparer les données du portfolio
        if isinstance(portfolio, dict) and 'stocks' in portfolio:
            portfolio_summary = "\n".join([f"{stock['symbol']}: {stock['weight']}%" for stock in portfolio['stocks']])
        elif isinstance(portfolio, list):
            portfolio_summary = "\n".join([f"{stock['symbol']}: {stock['weight']}%" for stock in portfolio])
        else:
            portfolio_summary = "No valid portfolio data available"

        # Vérifier et préparer l'historique du chat
        if isinstance(chat_history, list):
            chat_summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])  # Derniers 5 messages
        else:
            chat_summary = "No valid chat history available"

        response = self.client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
            {"role": "system", "content": "Vous êtes un analyste financier expert spécialisé dans l'analyse des profils d'utilisateurs et de leurs comportements d'investissement."},
            {"role": "user", "content": f"Analysez le profil de l'utilisateur {user_id} basé sur les informations suivantes:\n\nPortfolio:\n{portfolio_summary}\n\nHistorique de chat récent:\n{chat_summary}"}
            ],
            temperature=0.5,
            max_tokens=1000
        )

        return response.choices[0].message.content.strip()


user_profile_agent = UserProfileAgent()
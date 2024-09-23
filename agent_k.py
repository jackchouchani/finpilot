import openai

class UserProfileAgent:
    def __init__(self):
        self.client = openai.OpenAI()

    def analyze_user_profile(self, portfolio, chat_history):
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
            model="gpt-4o",
            messages=[
            {"role": "system", "content": "Vous êtes un analyste financier expert spécialisé dans l'analyse des données historiques des actions. Votre tâche est de fournir une analyse approfondie et des insights pertinents basés sur les données fournies."},
            {"role": "system", "content": "Dans votre analyse, veuillez inclure :\n1) Une évaluation de la performance globale de l'action sur la période donnée.\n2) Une analyse de la volatilité et du risque, y compris l'interprétation du ratio de Sharpe.\n3) Une identification des tendances à court et moyen terme.\n4) Une comparaison avec les indices de marché pertinents si possible.\n5) Des facteurs potentiels qui pourraient expliquer les mouvements de prix observés.\n6) Des points d'attention pour les investisseurs basés sur ces données historiques.\n\nAssurez-vous que votre analyse est objective, basée sur les faits présentés, et utile pour la prise de décision d'investissement."},
            {"role": "user", "content": f"Analysez les données suivantes pour {ticker} du {start_date} au {end_date}:\n{data}"}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content.strip()

user_profile_agent = UserProfileAgent()
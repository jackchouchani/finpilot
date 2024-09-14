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
                {"role": "system", "content": "You are a user profiling expert. Analyze the user's behavior and provide insights on their investment style and preferences."},
                {"role": "user", "content": f"Based on this portfolio:\n{portfolio_summary}\n\nAnd these recent chat interactions:\n{chat_summary}\n\nProvide an analysis of the user's investment profile."}
            ],
            temperature=0.7,
            max_tokens=800
        )

        return response.choices[0].message.content.strip()

user_profile_agent = UserProfileAgent()
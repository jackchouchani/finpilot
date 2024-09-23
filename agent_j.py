import yfinance as yf
from datetime import datetime, timedelta
import openai

class HistoricalDataAnalysisAgent:
    def __init__(self):
        self.client = openai.OpenAI()

    def analyze_previous_day(self, ticker, start_date=None, end_date=None):
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return f"Pas de données disponibles pour {ticker} entre {start_date} et {end_date}."

            # Préparation des données pour l'analyse
            data = {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "open_price": hist['Open'].iloc[0],
                "close_price": hist['Close'].iloc[-1],
                "high": hist['High'].max(),
                "low": hist['Low'].min(),
                "volume": hist['Volume'].mean(),
                "return": ((hist['Close'].iloc[-1] / hist['Open'].iloc[0]) - 1) * 100
            }

            # Utilisation de ChatGPT pour l'analyse
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "Vous êtes un analyste financier. Analysez les données boursières suivantes et fournissez des insights."},
                    {"role": "user", "content": f"Analysez les données suivantes pour {ticker} du {start_date} au {end_date}:\n{data}"}
                ],
                temperature=0.5,
                max_tokens=800
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Erreur lors de l'analyse des données historiques pour {ticker}: {str(e)}"

historical_data_agent = HistoricalDataAnalysisAgent()
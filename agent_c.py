import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np

class FinancialModelingAgent:
    def analyze(self, ticker):
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        # Calculer les rendements journaliers
        returns = hist['Close'].pct_change().dropna()
        
        # Calculer la volatilité (écart-type annualisé des rendements)
        volatility = returns.std() * np.sqrt(252)
        
        # Prédiction simple du prix (régression linéaire sur les 30 derniers jours)
        X = np.array(range(30)).reshape(-1, 1)
        y = hist['Close'].tail(30).values
        model = LinearRegression().fit(X, y)
        next_price = model.predict([[30]])[0]
        
        # Calculer les ratios financiers
        info = stock.info
        pe_ratio = info.get('trailingPE', 'N/A')
        pb_ratio = info.get('priceToBook', 'N/A')
        dividend_yield = info.get('dividendYield', 'N/A')
        if dividend_yield != 'N/A':
            dividend_yield *= 100  # Convertir en pourcentage
        
        return {
            "symbole": ticker,
            "prix_actuel": hist['Close'].iloc[-1],
            "prix_prédit": next_price,
            "volatilité": volatility,
            "volume": hist['Volume'].mean(),
            "ratio_pe": pe_ratio,
            "ratio_pb": pb_ratio,
            "rendement_dividende": dividend_yield,
            "plus_haut_52s": hist['High'].max(),
            "plus_bas_52s": hist['Low'].min()
        }

financial_modeling_agent = FinancialModelingAgent()
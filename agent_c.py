import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np

class FinancialModelingAgent:
    def analyze(self, ticker):
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        X = np.array(range(30)).reshape(-1, 1)
        y = hist['Close'].tail(30).values
        model = LinearRegression().fit(X, y)
        next_price = model.predict([[30]])[0]
        info = stock.info
        pe_ratio = info.get('trailingPE', 'N/A')
        pb_ratio = info.get('priceToBook', 'N/A')
        dividend_yield = info.get('dividendYield', 'N/A')
        if dividend_yield != 'N/A':
            dividend_yield *= 100  # Convertir en pourcentage

        rapport = f"""
Analyse financière de {ticker}

Prix actuel: {hist['Close'].iloc[-1]:.2f} USD
Prix prédit (prochain jour): {next_price:.2f} USD
Volatilité annualisée: {volatility:.2%}
Volume moyen d'échanges: {hist['Volume'].mean():.0f} actions

Ratios financiers:
- Ratio Cours/Bénéfice (P/E): {pe_ratio if pe_ratio != 'N/A' else 'Non disponible'}
- Ratio Cours/Valeur Comptable (P/B): {pb_ratio if pb_ratio != 'N/A' else 'Non disponible'}
- Rendement du dividende: {f'{dividend_yield:.2f}%' if dividend_yield != 'N/A' else 'Non disponible'}

Fourchette de prix sur 52 semaines:
- Plus haut: {hist['High'].max():.2f} USD
- Plus bas: {hist['Low'].min():.2f} USD

Interprétation:
1. Performance: {'Le prix prédit est supérieur au prix actuel, suggérant une tendance haussière potentielle.' if next_price > hist['Close'].iloc[-1] else 'Le prix prédit est inférieur au prix actuel, suggérant une possible baisse à court terme.'}
2. Volatilité: {'La volatilité est élevée, indiquant un risque important.' if volatility > 0.3 else 'La volatilité est modérée, suggérant un risque moyen.'}
3. Valorisation: {'Le ratio P/E suggère que l\'action pourrait être surévaluée.' if pe_ratio != 'N/A' and pe_ratio > 25 else 'Le ratio P/E semble raisonnable.' if pe_ratio != 'N/A' else 'Impossible d\'évaluer la valorisation sans le ratio P/E.'}

Conclusion: Cette analyse fournit un aperçu de la situation financière actuelle de {ticker}. Les investisseurs devraient considérer ces informations en conjonction avec une analyse plus approfondie du secteur et des nouvelles récentes de l'entreprise avant de prendre des décisions d'investissement.
        """
        return rapport

financial_modeling_agent = FinancialModelingAgent()
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import yfinance as yf
import numpy as np
import base64

class ReportingAgent:
    def _generate_portfolio_optimization_report(self, optimization_result):
        report = "Rapport d'Optimisation du Portefeuille\n\n"
        report += "Poids Optimaux:\n"
        for ticker, weight in optimization_result['optimal_weights'].items():
            report += f"  - {ticker}: {weight:.2%}\n"
        report += f"\nRendement Annuel Attendu: {optimization_result['expected_return']:.2%}\n"
        report += f"Volatilité Annuelle: {optimization_result['volatility']:.2%}\n"
        report += f"Ratio de Sharpe: {optimization_result['sharpe_ratio']:.2f}\n"
        return report

    def generate_report(self, portfolio):
        if not portfolio:
            return {"error": "Aucun portefeuille trouvé pour cet utilisateur"}
        
        for stock in portfolio:
            report += self._generate_stock_report(stock)
        
        report_md = f"""
# Rapport de Portefeuille

{report}
    """
        # Création des graphiques
        figs = self._create_graphs(portfolio)
        
    
        # Conversion des graphiques en base64
        graphs_base64 = []
        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            graphs_base64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

        # Création du résultat formaté
        formatted_result = {
            "content": report_md,
            "graphs": graphs_base64
        }

        return formatted_result

    def _generate_stock_report(self, stock):
        symbol = stock['symbol']
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        hist = ticker.history(period="1y")
        volatility = hist['Close'].pct_change().std() * (252 ** 0.5)  # Volatilité annualisée
        
        report = f"## Action : {symbol}\n\n"
        report += f"- Prix actuel : {info.get('currentPrice', 'N/A'):.2f} €\n"
        report += f"- Prix cible (moyenne des analystes) : {info.get('targetMeanPrice', 'N/A'):.2f} €\n"
        report += f"- Volatilité (1 an) : {volatility:.2%}\n"
        report += f"- Volume moyen (10 jours) : {info.get('averageVolume10days', 'N/A'):,.0f}\n"
        report += f"- Ratio P/E : {info.get('trailingPE', 'N/A'):.2f}\n"
        report += f"- Ratio P/B : {info.get('priceToBook', 'N/A'):.2f}\n"
        report += f"- Rendement du dividende : {info.get('dividendYield', 0):.2%}\n"
        report += f"- Plus haut sur 52 semaines : {info.get('fiftyTwoWeekHigh', 'N/A'):.2f} €\n"
        report += f"- Plus bas sur 52 semaines : {info.get('fiftyTwoWeekLow', 'N/A'):.2f} €\n"
        report += f"- Poids dans le portefeuille : {stock['weight']:.2f}%\n"
        report += f"- Prix d'entrée : {stock['entry_price']:.2f} €\n\n"
        return report

    def _create_graphs(self, portfolio):
        figs = []
        
        # Graphique 1 : Allocation du portefeuille
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        symbols = [stock['symbol'] for stock in portfolio]
        weights = [stock['weight'] for stock in portfolio]
        ax1.pie(weights, labels=symbols, autopct='%1.1f%%')
        ax1.set_title('Allocation du Portefeuille')
        figs.append(fig1)
        
        # Récupération des données yfinance
        data = {}
        for stock in portfolio:
            ticker = yf.Ticker(stock['symbol'])
            info = ticker.info
            hist = ticker.history(period="1y")
            data[stock['symbol']] = {
                'current_price': info.get('currentPrice', stock['entry_price']),
                'target_price': info.get('targetMeanPrice', stock['entry_price']),
                'volatility': hist['Close'].pct_change().std() * np.sqrt(252),
                'volume': info.get('averageVolume', 0)
            }
        
        # Graphique 2 : Prix actuels vs Prix cibles
        fig2 = Figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        current_prices = [data[s]['current_price'] for s in symbols]
        target_prices = [data[s]['target_price'] for s in symbols]
        x = range(len(symbols))
        width = 0.35
        ax2.bar([i - width/2 for i in x], current_prices, width, label='Prix Actuel')
        ax2.bar([i + width/2 for i in x], target_prices, width, label='Prix Cible')
        ax2.set_ylabel('Prix')
        ax2.set_title('Prix Actuels vs Prix Cibles')
        ax2.set_xticks(x)
        ax2.set_xticklabels(symbols, rotation=45)
        ax2.legend()
        figs.append(fig2)
        
        # Graphique 3 : Volatilité
        fig3 = Figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111)
        volatilities = [data[s]['volatility'] for s in symbols]
        ax3.bar(symbols, volatilities)
        ax3.set_ylabel('Volatilité Annualisée')
        ax3.set_title('Volatilité des Actions')
        ax3.set_xticklabels(symbols, rotation=45)
        figs.append(fig3)
        
        # Graphique 4 : Volume moyen
        fig4 = Figure(figsize=(10, 6))
        ax4 = fig4.add_subplot(111)
        volumes = [data[s]['volume'] for s in symbols]
        ax4.bar(symbols, volumes)
        ax4.set_ylabel('Volume Moyen')
        ax4.set_title('Volume Moyen des Actions')
        ax4.set_xticklabels(symbols, rotation=45)
        figs.append(fig4)
        
        return figs

reporting_agent = ReportingAgent()
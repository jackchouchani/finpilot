./compile.py :
<<<
import os
import argparse
import os
import base64


def lire_fichier(chemin):
    try:
        with open(chemin, 'r', encoding='utf-8') as fichier:
            return fichier.read()
    except Exception as e:
        return f"Erreur lors de la lecture du fichier : {str(e)}"

def compiler_fichiers(dossier_racine, extensions, dossiers_exclus, fichier_sortie):
    with open(fichier_sortie, 'w', encoding='utf-8') as sortie:
        for dossier_actuel, sous_dossiers, fichiers in os.walk(dossier_racine):
            # Exclure les dossiers spécifiés
            sous_dossiers[:] = [d for d in sous_dossiers if d not in dossiers_exclus]
            
            for fichier in fichiers:
                if any(fichier.endswith(ext) for ext in extensions):
                    chemin_complet = os.path.join(dossier_actuel, fichier)
                    contenu = lire_fichier(chemin_complet)
                    
                    sortie.write(f"{chemin_complet} :\n")
                    sortie.write("<<<\n")
                    sortie.write(contenu)
                    sortie.write("\n>>>\n\n")

def main():
    parser = argparse.ArgumentParser(description="Compiler des fichiers de code dans un fichier texte.")
    parser.add_argument("dossier", help="Le dossier racine à parcourir")
    parser.add_argument("--extensions", nargs='+', default=['.py'], help="Les extensions de fichier à inclure")
    parser.add_argument("--exclure", nargs='+', default=['node_modules'], help="Les dossiers à exclure")
    parser.add_argument("--sortie", default="compilation_code.txt", help="Le nom du fichier de sortie")
    
    args = parser.parse_args()
    
    compiler_fichiers(args.dossier, args.extensions, args.exclure, args.sortie)
    print(f"Compilation terminée. Résultat dans {args.sortie}")

if __name__ == "__main__":
    main()
>>>

./agent_i.py :
<<<
import openai
import yfinance as yf

class InvestmentRecommendationAgent:
    def __init__(self):
        self.client = openai.OpenAI()

    def get_recommendation(self, portfolio, risk_profile):
        # Récupérer les données de base pour le portfolio
        portfolio_data = {}
        for ticker in portfolio:
            stock = yf.Ticker(ticker)
            info = stock.info
            portfolio_data[ticker] = {
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "current_price": info.get('currentPrice', 0),
                "target_price": info.get('targetMeanPrice', 0)
            }

        # Préparer les données pour l'analyse
        portfolio_str = "\n".join([f"{ticker}: {data}" for ticker, data in portfolio_data.items()])
        print(portfolio_data)
        print(portfolio)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Vous êtes un conseiller en investissement financier. Fournissez des recommandations d'investissement basées sur le portefeuille et le profil de risque donnés."},
                {"role": "user", "content": f"Sur la base de ce portefeuille:\n{portfolio_str}\n\nEt d'un profil de risque {risk_profile}, fournissez des recommandations d'investissement."}
            ],
            temperature=0.7,
            max_tokens=1200
        )

        return response.choices[0].message.content.strip()

investment_recommendation_agent = InvestmentRecommendationAgent()
>>>

./compilation_text.py :
<<<

>>>

./agent_d.py :
<<<
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy.optimize import minimize

class PortfolioOptimizationAgent:
    def optimize(self, portfolio):
        if not portfolio:
            return {"erreur": "Aucun portefeuille trouvé pour cet utilisateur"}
        print(portfolio)

        tickers = [stock['symbol'] for stock in portfolio]
        weights = [float(stock['weight']) / 100 for stock in portfolio]  # Convertir en float et en pourcentage

        data = yf.download(tickers, period="5y")['Adj Close']
        returns = data.pct_change().dropna()

        def portfolio_performance(weights, mean_returns, cov_matrix):
            returns = np.sum(mean_returns * weights) * 252
            std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return std, returns

        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
            p_std, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
            return -(p_ret - risk_free_rate) / p_std

        def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
            num_assets = len(mean_returns)
            args = (mean_returns, cov_matrix, risk_free_rate)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bound = (0.0, 1.0)
            bounds = tuple(bound for asset in range(num_assets))
            result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)
            return result

        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        risk_free_rate = 0.01

        opt = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
        
        current_std, current_ret = portfolio_performance(weights, mean_returns, cov_matrix)
        current_sharpe = (current_ret - risk_free_rate) / current_std

        opt_std, opt_ret = portfolio_performance(opt.x, mean_returns, cov_matrix)
        opt_sharpe = (opt_ret - risk_free_rate) / opt_std

        rapport = f"""
Analyse de l'optimisation du portefeuille

Portefeuille actuel:
- Poids: {dict(zip(tickers, weights))}
- Rendement attendu: {current_ret:.2f}
- Volatilité: {current_std:.2f}
- Ratio de Sharpe: {current_sharpe:.2f}

Portefeuille optimisé:
- Poids: {dict(zip(tickers, opt.x))}
- Rendement attendu: {opt_ret:.2f}
- Volatilité: {opt_std:.2f}
- Ratio de Sharpe: {opt_sharpe:.2f}

Conclusion:
Cette analyse fournit un aperçu de l'optimisation du portefeuille. Les investisseurs peuvent utiliser ces informations pour ajuster leur stratégie d'investissement.
"""
        return rapport

portfolio_optimization_agent = PortfolioOptimizationAgent()
>>>

./agent_j.py :
<<<
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
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Vous êtes un analyste financier. Analysez les données boursières suivantes et fournissez des insights."},
                    {"role": "user", "content": f"Analysez les données suivantes pour {ticker} du {start_date} au {end_date}:\n{data}"}
                ],
                temperature=0.7,
                max_tokens=800
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Erreur lors de l'analyse des données historiques pour {ticker}: {str(e)}"

historical_data_agent = HistoricalDataAnalysisAgent()
>>>

./wsgi.py :
<<<

from copilote_central import app

if __name__ == "__main__":
    app.run()
>>>

./litefs.yaml :
<<<
fuse:
  dir: "/litefs" # Le répertoire où votre application accédera à la base de données
data:
  dir: "/var/lib/litefs" # Le répertoire où LiteFS stockera ses données internes (sur un volume persistant)
lease:
  type: "consul"
  candidate: ${FLY_REGION == PRIMARY_REGION} # Ce nœud peut-il devenir primaire ?
  promote: true # Devenir primaire automatiquement après la synchronisation ?
  advertise-url: "http://${FLY_ALLOC_ID}.vm.${FLY_APP_NAME}.internal:20202" # URL pour que d'autres nœuds se connectent
  consul:
    url: "${FLY_CONSUL_URL}" 
    key: "${FLY_APP_NAME}/primary"
proxy:
  addr: ":8080" # Adresse sur laquelle le proxy écoutera
  target: "localhost:5000" # Port de votre application backend
  db: "copilot-db.db" # Nom de votre base de données SQLite
>>>

./agent_f.py :
<<<
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
        
        report = ""

        for stock in portfolio:
            report += self._generate_stock_report(stock)
        
        report_md = f"""
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
>>>

./generateur_report.py :
<<<
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import os
import time
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from plotly import graph_objects as go
from plotly import figure_factory as ff
import plotly.express as px
import yfinance as yf
from flask import jsonify
import anthropic

anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def generate_ai_content(prompt):
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

def create_paragraph(text, style_name='Normal'):
    styles = getSampleStyleSheet()
    if style_name not in styles:
        style_name = 'Normal'
    style = styles[style_name]
    
    # Nettoyer le texte et ajouter des retours à la ligne
    cleaned_text = clean_text(text)
    wrapped_text = textwrap.fill(cleaned_text, width=80)
    
    custom_style = ParagraphStyle('CustomStyle', parent=style, wordWrap='CJK')
    
    return Paragraph(wrapped_text, custom_style)

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\n', ' ')
    text = ''.join(char for char in text if ord(char) > 31 or char == ' ')
    return text

def generate_report(data):
    portfolio = data['portfolio']
    start_date = data.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    # Calcul des rendements et des métriques de performance
    portfolio_data, returns, weights = calculate_portfolio_performance(portfolio, start_date, end_date)
    
    # Création du document PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=portrait(letter), 
                               rightMargin=0.5*inch, leftMargin=0.5*inch, 
                               topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    elements = []
    
    # Page de titre
    elements.extend(create_title_page(
        "Rapport de Performance du Portefeuille",
        f"Pour : {data.get('client_name', 'Client Estimé')}",
        f"Date : {datetime.now().strftime('%d/%m/%Y')}"
    ))

    # Table des matières
    elements.append(create_section_header("Table des Matières"))
        # Supprimez cette ligne
    # from reportlab.platypus import TableOfContents

    # Remplacez la section de création de la table des matières par un simple paragraphe
    elements.append(create_section_header("Table des Matières"))
    # Ajoutez manuellement les sections que vous avez dans votre rapport
    elements.append(create_paragraph("1. Résumé Exécutif", 'Normal'))
    elements.append(create_paragraph("2. Vue d'Ensemble du Portefeuille", 'Normal'))
    elements.append(create_paragraph("3. Analyse de Performance", 'Normal'))
    # Ajoutez d'autres sections selon vos besoins
    elements.append(PageBreak())

    # Sections du rapport
    sections = [
        ("Résumé Exécutif", lambda p, pd, r, w: generate_executive_summary(p, pd, r, w)),
        ("Vue d'Ensemble du Portefeuille", lambda p, pd, r, w: generate_portfolio_overview(p, pd, r, w)),
        ("Analyse de Performance", lambda p, pd, r, w: generate_performance_analysis(p, pd, r, w, start_date, end_date)),
        ("Comparaison de Performance des Actions", lambda p, pd, r, w: generate_stock_performance_comparison(pd, w)),
        ("Contribution au Rendement", lambda p, pd, r, w: generate_contribution_to_return(p, pd, r, w)),
        ("Ratios Supplémentaires", lambda p, pd, r, w: generate_additional_ratios_table(p, pd, r, w, start_date, end_date)),
        ("Analyse des Risques", lambda p, pd, r, w: generate_risk_analysis(p, pd, r, w)),
        ("Corrélation des Actions", lambda p, pd, r, w: generate_correlation_heatmap(pd)),
        ("Meilleures et Pires Performances", lambda p, pd, r, w: generate_best_worst_performers(p, pd, r, w)),
        ("Analyse des Dividendes", lambda p, pd, r, w: generate_dividend_table(p)),
        ("Analyse ESG", lambda p, pd, r, w: generate_esg_analysis(p)),
        ("Allocation Sectorielle", lambda p, pd, r, w: generate_sector_allocation(p, pd, r, w)),
        ("Simulation Monte Carlo", lambda p, pd, r, w: generate_monte_carlo_simulation(p, pd, r, w)),
        ("Tests de Stress", lambda p, pd, r, w: generate_stress_tests(p, pd, r, w)),
        ("Perspectives Futures", lambda p, pd, r, w: generate_future_outlook(p, pd, r, w)),
        ("Recommandations", lambda p, pd, r, w: generate_recommendations(p, pd, r, w))
    ]

    def format_time(seconds):
        return f"{seconds:.2f} secondes"

    for title, function in sections:
        elements.append(create_section_header(title))
        start_time = time.time()
        new_elements = function(portfolio, portfolio_data, returns, weights)
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Temps d'exécution pour '{title}': {format_time(execution_time)}")
        
        if isinstance(new_elements, list):
            elements.extend(new_elements)
        else:
            elements.append(create_paragraph(str(new_elements), 'Normal'))
        elements.append(PageBreak())

    # Glossaire et avertissements
    elements.append(create_section_header("Glossaire"))
    elements.extend(generate_glossary())
    
    elements.append(create_section_header("Avertissements et Divulgations"))
    elements.append(create_paragraph(generate_disclaimer(), 'Normal'))

    # Génération du PDF
    doc.build(elements, onFirstPage=add_page_number, onLaterPages=add_page_number)
    pdf = buffer.getvalue()
    buffer.close()

    # Encodage du PDF en base64
    pdf_base64 = base64.b64encode(pdf).decode('utf-8')

    return jsonify({"report": pdf_base64})

def create_title_page(title, subtitle, date):
    elements = []
    styles = getSampleStyleSheet()
    
    # Ajouter le logo
    logo_path = "copilot/public/logo.jpg"  # Remplacez par le chemin de votre logo
    logo = Image(logo_path, width=2*inch, height=1*inch)
    elements.append(logo)
    
    elements.append(Spacer(1, 1*inch))
    
    title_style = ParagraphStyle(name='Title', parent=styles['Title'], fontSize=24, alignment=1)
    elements.append(Paragraph(title, title_style))
    
    elements.append(Spacer(1, 0.5*inch))
    
    subtitle_style = ParagraphStyle(name='Subtitle', parent=styles['Normal'], fontSize=18, alignment=1)
    elements.append(Paragraph(subtitle, subtitle_style))
    
    elements.append(Spacer(1, 0.5*inch))
    
    date_style = ParagraphStyle(name='Date', parent=styles['Normal'], fontSize=14, alignment=1)
    elements.append(Paragraph(date, date_style))
    
    elements.append(PageBreak())
    return elements

def calculate_portfolio_performance(portfolio, start_date, end_date):
    portfolio_data = {}
    for stock in portfolio['stocks']:
        ticker = yf.Ticker(stock['symbol'])
        hist = ticker.history(start=start_date, end=end_date)
        portfolio_data[stock['symbol']] = hist['Close']

    df = pd.DataFrame(portfolio_data)
    returns = df.pct_change().dropna()
    weights = np.array([float(stock['weight']) / 100 for stock in portfolio['stocks']])
    
    return portfolio_data, returns, weights

def generate_executive_summary(portfolio, portfolio_data, returns, weights):
    elements = []
    
    df = pd.DataFrame(portfolio_data)
    total_return = (df.iloc[-1] / df.iloc[0] - 1).sum()
    days = len(df)
    annualized_return = (1 + total_return) ** (252 / days) - 1
    portfolio_returns = df.pct_change().dropna()
    portfolio_returns_weighted = (portfolio_returns * weights).sum(axis=1)
    portfolio_volatility = portfolio_returns_weighted.std() * np.sqrt(252)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / portfolio_volatility

    sp500_data = get_sp500_data(df.index[0], df.index[-1])
    sp500_return = (sp500_data.iloc[-1] / sp500_data.iloc[0]) - 1
    sp500_volatility = sp500_data.pct_change().std() * np.sqrt(252)
    sp500_sharpe = (sp500_return - risk_free_rate) / sp500_volatility

    summary = f"""
    Résumé Exécutif

    Ce rapport présente une analyse détaillée de la performance du portefeuille sur la période du {df.index[0].strftime('%d/%m/%Y')} au {df.index[-1].strftime('%d/%m/%Y')}.

    Points clés :
    • Rendement total du portefeuille : {total_return:.2%}
    • Rendement annualisé : {annualized_return:.2%}
    • Volatilité annualisée : {portfolio_volatility:.2%}
    • Ratio de Sharpe : {sharpe_ratio:.2f}

    Comparaison avec le S&P 500 :
    • Rendement total S&P 500 : {sp500_return:.2%}
    • Volatilité S&P 500 : {sp500_volatility:.2%}
    • Ratio de Sharpe S&P 500 : {sp500_sharpe:.2f}

    Le portefeuille a {['sous-performé', 'sur-performé'][total_return > sp500_return]} l'indice S&P 500 sur la période, avec un {['risque plus élevé', 'risque plus faible'][portfolio_volatility < sp500_volatility]}.
    """
    
    elements.append(create_paragraph(summary, 'BodyText'))

    additional_analysis = generate_ai_content(f"""
    En vous basant sur les données suivantes :
    - Rendement total du portefeuille : {total_return:.2%}
    - Rendement annualisé : {annualized_return:.2%}
    - Volatilité du portefeuille : {portfolio_volatility:.2%}
    - Ratio de Sharpe du portefeuille : {sharpe_ratio:.2f}
    - Rendement total S&P 500 : {sp500_return:.2%}
    - Volatilité S&P 500 : {sp500_volatility:.2%}
    - Ratio de Sharpe S&P 500 : {sp500_sharpe:.2f}

    Fournissez une analyse succincte de la performance du portefeuille. Incluez :
    1. Une évaluation générale de la performance du portefeuille par rapport au marché.
    2. Les principaux facteurs qui ont contribué à cette performance.
    3. Les points forts et les points faibles du portefeuille.
    4. Des recommandations préliminaires pour l'amélioration du portefeuille.
    """)
    
    elements.append(create_paragraph(additional_analysis, 'BodyText'))
    
    return elements


def generate_portfolio_overview(portfolio, portfolio_data, returns, weights):
    elements = []
    
    df = pd.DataFrame(portfolio_data)
    
    data = [['Titre', 'Poids', 'Prix d\'entrée', 'Prix actuel', 'Rendement']]
    for stock, weight in zip(portfolio['stocks'], weights):
        symbol = stock['symbol']
        entry_price = float(stock['entry_price'])
        current_price = df[symbol].iloc[-1]
        stock_return = (current_price / entry_price - 1)
        data.append([symbol, f"{weight:.2%}", f"{entry_price:.2f} €", f"{current_price:.2f} €", f"{stock_return:.2%}"])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))
    
    fig = go.Figure()
    portfolio_value = (df * weights).sum(axis=1)
    fig.add_trace(go.Scatter(x=df.index, y=portfolio_value,
                             mode='lines', name='Valeur du Portefeuille'))
    fig.update_layout(title="Évolution de la Valeur du Portefeuille",
                      xaxis_title="Date", yaxis_title="Valeur")
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=300))
    
    return elements

def generate_glossary():
    elements = []
    styles = getSampleStyleSheet()
    body_style = styles['BodyText']
    
    glossary_terms = [
        ("Rendement Total", "Le gain ou la perte sur un investissement exprimé en pourcentage du capital investi."),
        ("Volatilité", "Une mesure de la variation des prix d'un actif financier."),
        ("Ratio de Sharpe", "Une mesure du rendement ajusté au risque, calculée en divisant le rendement excédentaire par l'écart-type des rendements."),
        ("VaR (Value at Risk)", "Une mesure statistique de la perte potentielle maximale d'un portefeuille sur une période donnée pour un niveau de confiance spécifié."),
        ("Drawdown", "La baisse en pourcentage d'un investissement par rapport à son pic le plus récent.")
    ]
    
    for term, definition in glossary_terms:
        elements.append(Paragraph(f"<b>{term}</b>: {definition}", body_style))
        elements.append(Spacer(1, 6))
    
    return elements

def generate_disclaimer():
    disclaimer_text = """
    Ce rapport est fourni à titre informatif uniquement et ne constitue pas un conseil en investissement. 
    Les performances passées ne garantissent pas les résultats futurs. La valeur des investissements peut fluctuer 
    et les investisseurs peuvent perdre une partie ou la totalité de leur capital investi. Avant de prendre toute 
    décision d'investissement, il est recommandé de consulter un conseiller financier professionnel. 
    Les informations contenues dans ce rapport sont considérées comme fiables, mais leur exactitude et leur 
    exhaustivité ne peuvent être garanties. Ce rapport ne doit pas être reproduit, distribué ou publié sans 
    autorisation préalable.
    """
    return disclaimer_text.strip()

def generate_stock_performance_comparison(portfolio_data, weights):
    elements = []
    
    df = pd.DataFrame(portfolio_data)
    stock_returns = df.pct_change().mean() * 252
    
    stock_performance = list(zip(stock_returns.index, stock_returns.values * 100))
    stock_performance.sort(key=lambda x: x[1], reverse=True)
    
    fig = go.Figure([go.Bar(
        x=[s[0] for s in stock_performance],
        y=[s[1] for s in stock_performance],
        text=[f"{s[1]:.2f}%" for s in stock_performance],
        textposition='auto',
    )])
    fig.update_layout(title="Comparaison de Performance des Actions",
                      xaxis_title="Action", yaxis_title="Rendement Annualisé (%)")
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=300))
    
    explanation = generate_ai_content(f"""
    Analysez la performance relative des actions du portefeuille en vous basant sur les données suivantes:
    {', '.join([f"{s[0]}: {s[1]:.2f}%" for s in stock_performance])}
    Identifiez les meilleures et les pires performances, et suggérez des explications possibles pour ces écarts de performance.
    Considérez également l'impact de la pondération de chaque action (poids: {weights}) sur la performance globale du portefeuille.
    """)
    elements.append(create_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_contribution_to_return(portfolio, portfolio_data, returns, weights):
    elements = []
    
    df = pd.DataFrame(portfolio_data)
    total_return = (df.iloc[-1] / df.iloc[0] - 1).sum()
    contributions = []
    for stock, weight in zip(portfolio['stocks'], weights):
        symbol = stock['symbol']
        stock_return = df[symbol].iloc[-1] / df[symbol].iloc[0] - 1
        contribution = stock_return * weight
        contributions.append((symbol, contribution, contribution / total_return))
    
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    data = [['Action', 'Contribution', '% du Total']]
    for symbol, contribution, percentage in contributions:
        data.append([symbol, f"{contribution:.2%}", f"{percentage:.2%}"])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    explanation = generate_ai_content(f"""
    Analysez la contribution de chaque action au rendement total du portefeuille en vous basant sur les données suivantes:
    {', '.join([f"{s[0]}: {s[1]:.2f}%" for s in contributions])}
    Identifiez les actions qui ont le plus contribué positivement et négativement, et expliquez l'impact de la pondération sur ces contributions.
    """)
    elements.append(create_paragraph(explanation, 'BodyText'))
    
    return elements


def get_sp500_returns(start_date, end_date):
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(start=start_date, end=end_date)['Close']
    return sp500_data.pct_change().dropna()

def get_sp500_data(start_date, end_date):
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(start=start_date, end=end_date)['Close']
    return sp500_data

def generate_additional_ratios_table(portfolio, portfolio_data, returns, weights, start_date, end_date):
    elements = []
    
    # Recalculer les rendements du portefeuille
    df = pd.DataFrame(portfolio_data)
    portfolio_returns = df.pct_change().dropna()
    portfolio_returns = (portfolio_returns * weights).sum(axis=1)

    # Obtenir les rendements du S&P 500
    benchmark_returns = get_sp500_returns(start_date, end_date)

    # S'assurer que les deux séries ont le même index
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]

    risk_free_rate = 0.02 / 252  # Taux journalier
    
    excess_returns = portfolio_returns - risk_free_rate
    benchmark_excess_returns = benchmark_returns - risk_free_rate
    
    beta = np.cov(portfolio_returns, benchmark_returns)[0][1] / np.var(benchmark_returns)
    alpha = np.mean(excess_returns) - beta * np.mean(benchmark_excess_returns)
    tracking_error = np.std(portfolio_returns - benchmark_returns) * np.sqrt(252)
    information_ratio = (np.mean(portfolio_returns - benchmark_returns) * 252) / tracking_error
    downside_returns = np.minimum(excess_returns - np.mean(excess_returns), 0)
    sortino_ratio = np.mean(excess_returns) / (np.std(downside_returns) * np.sqrt(252))
    
    ratios = {
        "Beta": beta,
        "Alpha": alpha * 252,  # Annualisé
        "Tracking Error": tracking_error,
        "Information Ratio": information_ratio,
        "Sortino Ratio": sortino_ratio
    }
    
    data = [['Ratio', 'Valeur']]
    for ratio, value in ratios.items():
        data.append([ratio, f"{value:.4f}"])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    explanation = generate_ai_content(f"""
    Analysez les ratios supplémentaires du portefeuille en vous basant sur les données suivantes:
    Beta: {ratios['Beta']:.4f}
    Alpha: {ratios['Alpha']:.4f}
    Tracking Error: {ratios['Tracking Error']:.4f}
    Information Ratio: {ratios['Information Ratio']:.4f}
    Sortino Ratio: {ratios['Sortino Ratio']:.4f}
    Expliquez ce que chaque ratio signifie et comment interpréter ces valeurs dans le contexte de ce portefeuille.
    """)
    elements.append(create_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_correlation_heatmap(portfolio_data):
    elements = []
    
    correlation_matrix = pd.DataFrame(portfolio_data).pct_change().corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.index,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    fig.update_layout(title="Matrice de Corrélation des Actions")
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=500))
    
    explanation = generate_ai_content(f"""
    Analysez la matrice de corrélation des actions du portefeuille.
    Identifiez les paires d'actions les plus corrélées et les moins corrélées.
    Discutez de l'impact de ces corrélations sur la diversification du portefeuille.
    Suggérez des moyens d'améliorer la diversification du portefeuille en fonction de ces corrélations.
    """)
    elements.append(create_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_performance_analysis(portfolio, portfolio_data, returns, weights, start_date, end_date):
    elements = []
    
    df = pd.DataFrame(portfolio_data)
    portfolio_returns = (returns * weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility
    
    sp500_data = get_sp500_data(start_date, end_date)
    sp500_returns = sp500_data.pct_change().dropna()
    sp500_cumulative_returns = (1 + sp500_returns).cumprod()
    sp500_total_return = sp500_cumulative_returns.iloc[-1] - 1
    sp500_annualized_return = (1 + sp500_total_return) ** (252 / len(sp500_returns)) - 1
    sp500_volatility = sp500_returns.std() * np.sqrt(252)
    sp500_sharpe_ratio = (sp500_annualized_return - 0.02) / sp500_volatility
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                             mode='lines', name='Portefeuille'))
    fig.add_trace(go.Scatter(x=sp500_cumulative_returns.index, y=sp500_cumulative_returns,
                             mode='lines', name='S&P 500'))
    fig.update_layout(title="Comparaison des Rendements Cumulés",
                      xaxis_title="Date", yaxis_title="Rendement Cumulé")
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=300))

    performance_comparison = [
        ['Métrique', 'Portefeuille', 'S&P 500'],
        ['Rendement Total', f"{total_return:.2%}", f"{sp500_total_return:.2%}"],
        ['Rendement Annualisé', f"{annualized_return:.2%}", f"{sp500_annualized_return:.2%}"],
        ['Volatilité Annualisée', f"{volatility:.2%}", f"{sp500_volatility:.2%}"],
        ['Ratio de Sharpe', f"{sharpe_ratio:.2f}", f"{sp500_sharpe_ratio:.2f}"]
    ]
    
    table = Table(performance_comparison)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)

    explanation = generate_ai_content(f"""
    Expliquez la performance du portefeuille par rapport au S&P 500 en vous basant sur les données suivantes:
    - Rendement total du portefeuille: {total_return:.2%}
    - Rendement total du S&P 500: {sp500_total_return:.2%}
    - Volatilité du portefeuille: {volatility:.2%}
    - Volatilité du S&P 500: {sp500_volatility:.2%}
    - Ratio de Sharpe du portefeuille: {sharpe_ratio:.2f}
    - Ratio de Sharpe du S&P 500: {sp500_sharpe_ratio:.2f}
    """)
    elements.append(create_paragraph(explanation, 'BodyText'))
    # Graphique des rendements cumulés
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                             mode='lines', name='Rendements Cumulés'))
    fig.update_layout(title="Rendements Cumulés du Portefeuille",
                      xaxis_title="Date", yaxis_title="Rendement Cumulé")
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=300))
    
    return elements

def generate_risk_analysis(portfolio, portfolio_data, returns, weights):
    elements = []
    
    portfolio_returns = (returns * weights).sum(axis=1)
    
    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    max_drawdown = np.min(portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax())
    
    risk_data = [
        ['Métrique', 'Valeur'],
        ['VaR (95%)', f"{var_95:.2%}"],
        ['CVaR (95%)', f"{cvar_95:.2%}"],
        ['Drawdown Maximum', f"{max_drawdown:.2%}"]
    ]
    
    table = Table(risk_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))
    
    fig = ff.create_distplot([portfolio_returns], ['Rendements du Portefeuille'], show_hist=False, show_rug=False)
    fig.update_layout(title="Distribution des Rendements du Portefeuille",
                      xaxis_title="Rendement", yaxis_title="Densité")
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=300))
    
    return elements

def generate_sector_allocation(portfolio, portfolio_data, returns, weights):
    elements = []
    
    # Obtenir les secteurs pour chaque action
    sectors = {}
    for stock in portfolio['stocks']:
        ticker = yf.Ticker(stock['symbol'])
        info = ticker.info
        sector = info.get('sector', 'Unknown')
        sectors[stock['symbol']] = sector
    
    # Calculer l'allocation sectorielle
    sector_weights = {}
    for stock, weight in zip(portfolio['stocks'], weights):
        sector = sectors[stock['symbol']]
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    
    # Créer le graphique
    fig = px.pie(values=list(sector_weights.values()), names=list(sector_weights.keys()),
                 title="Allocation Sectorielle du Portefeuille")
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=300))
    
    explanation = generate_ai_content(f"""
    Analysez l'allocation sectorielle du portefeuille en vous basant sur les données suivantes :
    {', '.join([f"{sector}: {weight:.2%}" for sector, weight in sector_weights.items()])}
    Discutez de la diversification sectorielle du portefeuille.
    Identifiez les secteurs surpondérés et sous-pondérés par rapport à un indice de référence (comme le S&P 500).
    Commentez sur les risques et opportunités potentiels liés à cette allocation sectorielle.
    """)
    elements.append(create_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_best_worst_performers(portfolio, portfolio_data, returns, weights):
    elements = []
    
    df = pd.DataFrame(portfolio_data)
    stock_returns = df.pct_change().mean() * 252  # rendements annualisés
    stock_volatility = df.pct_change().std() * np.sqrt(252)  # volatilité annualisée
    stock_sharpe = stock_returns / stock_volatility  # ratio de Sharpe

    performance_data = []
    for stock, weight in zip(portfolio['stocks'], weights):
        symbol = stock['symbol']
        performance_data.append({
            'symbol': symbol,
            'return': stock_returns[symbol],
            'volatility': stock_volatility[symbol],
            'sharpe': stock_sharpe[symbol],
            'weight': weight
        })

    performance_data.sort(key=lambda x: x['return'], reverse=True)
    
    best_worst_data = [['Rang', 'Titre', 'Rendement', 'Volatilité', 'Sharpe', 'Poids']]
    for i in range(min(3, len(performance_data))):
        best = performance_data[i]
        worst = performance_data[-(i+1)]
        best_worst_data.append([
            f"Top {i+1}",
            best['symbol'],
            f"{best['return']:.2%}",
            f"{best['volatility']:.2%}",
            f"{best['sharpe']:.2f}",
            f"{best['weight']:.2%}"
        ])
        best_worst_data.append([
            f"Bottom {i+1}",
            worst['symbol'],
            f"{worst['return']:.2%}",
            f"{worst['volatility']:.2%}",
            f"{worst['sharpe']:.2f}",
            f"{worst['weight']:.2%}"
        ])
    
    table = Table(best_worst_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    explanation = generate_ai_content(f"""
    Analysez les meilleures et pires performances du portefeuille en vous basant sur les données suivantes:
    Meilleures performances:
    {', '.join([f"{p['symbol']}: rendement {p['return']:.2%}, volatilité {p['volatility']:.2%}, Sharpe {p['sharpe']:.2f}" for p in performance_data[:3]])}
    Pires performances:
    {', '.join([f"{p['symbol']}: rendement {p['return']:.2%}, volatilité {p['volatility']:.2%}, Sharpe {p['sharpe']:.2f}" for p in performance_data[-3:]])}
    Discutez des facteurs qui pourraient expliquer ces performances. Commentez sur l'impact de ces performances sur l'ensemble du portefeuille, en tenant compte des pondérations.
    """)
    elements.append(create_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_dividend_table(portfolio):
    elements = []
    
    dividend_data = [['Titre', 'Rendement du Dividende', 'Fréquence', 'Dernier Dividende']]
    for stock in portfolio['stocks']:
        ticker = yf.Ticker(stock['symbol'])
        info = ticker.info
        dividend_yield = info.get('dividendYield', 0)
        dividend_rate = info.get('dividendRate', 0)
        
        dividend_data.append([
            stock['symbol'],
            f"{dividend_yield:.2%}" if dividend_yield else "N/A",
            info.get('dividendFrequency', 'N/A'),
            f"${dividend_rate:.2f}" if dividend_rate else "N/A"
        ])
    
    table = Table(dividend_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    explanation = generate_ai_content(f"""
    Analysez la politique de dividendes des actions du portefeuille en vous basant sur les données du tableau.
    Identifiez les actions avec les rendements de dividendes les plus élevés et les plus bas.
    Discutez de l'impact des dividendes sur le rendement total du portefeuille.
    Commentez sur la durabilité des dividendes en fonction des taux de distribution (si disponibles).
    """)
    elements.append(create_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_esg_analysis(portfolio):
    elements = []
    
    esg_data = [['Titre', 'Score ESG', 'Environnement', 'Social', 'Gouvernance']]
    portfolio_esg_score = 0
    
    for stock in portfolio['stocks']:
        ticker = yf.Ticker(stock['symbol'])
        info = ticker.info
        esg_score = info.get('esgScore', 'N/A')
        environment_score = info.get('environmentScore', 'N/A')
        social_score = info.get('socialScore', 'N/A')
        governance_score = info.get('governanceScore', 'N/A')
        
        if esg_score != 'N/A':
            portfolio_esg_score += esg_score * float(stock['weight']) / 100
        
        esg_data.append([
            stock['symbol'],
            f"{esg_score:.2f}" if esg_score != 'N/A' else 'N/A',
            f"{environment_score:.2f}" if environment_score != 'N/A' else 'N/A',
            f"{social_score:.2f}" if social_score != 'N/A' else 'N/A',
            f"{governance_score:.2f}" if governance_score != 'N/A' else 'N/A'
        ])
    
    table = Table(esg_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    elements.append(create_paragraph(f"Score ESG du portefeuille : {portfolio_esg_score:.2f}", 'BodyText'))
    
    explanation = generate_ai_content(f"""
    Analysez les scores ESG du portefeuille en vous basant sur les données du tableau.
    Le score ESG moyen pondéré du portefeuille est de {portfolio_esg_score:.2f}.
    Identifiez les entreprises les plus performantes et les moins performantes en termes d'ESG.
    Discutez de l'importance des critères ESG dans la gestion de portefeuille moderne.
    Suggérez des moyens d'améliorer le profil ESG global du portefeuille.
    """)
    elements.append(create_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_monte_carlo_simulation(portfolio, portfolio_data, returns, weights):
    elements = []
    
    # Paramètres de la simulation
    num_simulations = 1000
    num_days = 252  # un an de trading
    
    # Calcul des paramètres de la distribution des rendements
    portfolio_returns = (returns * weights).sum(axis=1)
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    
    # Simulation
    simulations = np.random.normal(mean_return, std_return, (num_simulations, num_days))
    simulations = np.cumprod(1 + simulations, axis=1)
    
    # Calcul des percentiles
    final_values = simulations[:, -1]
    percentiles = np.percentile(final_values, [5, 50, 95])
    
    # Graphique
    fig = go.Figure()
    for i in range(100):  # Tracer 100 simulations pour la lisibilité
        fig.add_trace(go.Scatter(y=simulations[i], mode='lines', opacity=0.1,
                                 line=dict(color='blue'), showlegend=False))
    fig.update_layout(title="Simulation Monte Carlo de la Valeur du Portefeuille",
                      xaxis_title="Jours", yaxis_title="Valeur du Portefeuille")
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=300))
    
    # Tableau des résultats
    results_data = [
        ['Percentile', 'Valeur Finale'],
        ['5%', f"{percentiles[0]:.2f}"],
        ['50%', f"{percentiles[1]:.2f}"],
        ['95%', f"{percentiles[2]:.2f}"]
    ]
    table = Table(results_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    explanation = generate_ai_content(f"""
    Interprétez les résultats de la simulation Monte Carlo pour le portefeuille.
    Les valeurs finales du portefeuille après un an de trading sont :
    - 5% de chance d'être inférieure à {percentiles[0]:.2f}
    - 50% de chance d'être supérieure à {percentiles[1]:.2f}
    - 5% de chance d'être supérieure à {percentiles[2]:.2f}
    Discutez des implications de ces résultats pour l'investisseur.
    Commentez sur la dispersion des résultats et ce qu'elle signifie en termes de risque pour le portefeuille.
    """)
    elements.append(create_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_stress_tests(portfolio, portfolio_data, returns, weights):
    elements = []
    
    # Définition des scénarios de stress
    scenarios = {
        "Crise financière": -0.4,
        "Récession modérée": -0.2,
        "Hausse des taux d'intérêt": -0.1,
        "Choc pétrolier": -0.15
    }
    
    stress_data = [['Scénario', 'Impact sur le Portefeuille']]
    for scenario, impact in scenarios.items():
        portfolio_impact = sum(impact * weight for weight in weights)
        stress_data.append([scenario, f"{portfolio_impact:.2%}"])
    
    table = Table(stress_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    # Graphique des impacts des scénarios
    fig = go.Figure(data=[go.Bar(x=list(scenarios.keys()), 
                                 y=[sum(impact * weight for weight in weights) for impact in scenarios.values()])])
    fig.update_layout(title="Impact des Scénarios de Stress sur le Portefeuille",
                      xaxis_title="Scénario", yaxis_title="Impact (%)")
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=300))
    
    explanation = generate_ai_content(f"""
    Analysez les résultats des tests de stress sur le portefeuille.
    Pour chaque scénario, discutez de l'impact potentiel sur le portefeuille :
    {', '.join([f"{scenario}: {sum(impact * weight for weight in weights):.2%}" for scenario, impact in scenarios.items()])}
    Identifiez les scénarios qui posent le plus grand risque pour le portefeuille.
    Suggérez des stratégies pour atténuer ces risques, comme la diversification ou l'utilisation d'instruments de couverture.
    """)
    elements.append(create_paragraph(explanation, 'BodyText'))
    
    return elements

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 9)
    page_number_text = "%d" % (doc.page)
    canvas.drawCentredString(4.25 * inch, 0.75 * inch, page_number_text)
    canvas.restoreState()

def create_section_header(text, level=1):
    style = getSampleStyleSheet()[f'Heading{level}']
    return Paragraph(text, style)

def generate_recommendations(portfolio, portfolio_data, returns, weights):
    elements = []
    
    # Calculer quelques métriques pour baser nos recommandations
    df = pd.DataFrame(portfolio_data)
    total_return = (df.iloc[-1] / df.iloc[0] - 1).sum()
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    # Calculer la volatilité du portefeuille
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
    
    sharpe_ratio = (annualized_return - 0.02) / portfolio_volatility  # Assuming 2% risk-free rate

    # Comparer avec le S&P 500
    sp500_data = get_sp500_data(df.index[0], df.index[-1])
    sp500_return = (sp500_data.iloc[-1] / sp500_data.iloc[0]) - 1
    sp500_volatility = sp500_data.pct_change().std() * np.sqrt(252)

    # Calculer la corrélation moyenne entre les actions
    correlation_matrix = returns.corr()
    avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values,1)].mean()

    # Générer des recommandations basées sur ces métriques
    recommendations_text = generate_ai_content(f"""
    En vous basant sur les données suivantes du portefeuille :
    - Rendement total : {total_return:.2%}
    - Rendement annualisé : {annualized_return:.2%}
    - Volatilité : {portfolio_volatility:.2%}
    - Ratio de Sharpe : {sharpe_ratio:.2f}
    - Rendement du S&P 500 : {sp500_return:.2%}
    - Volatilité du S&P 500 : {sp500_volatility:.2%}
    - Corrélation moyenne entre les actions : {avg_correlation:.2f}

    Générez une liste de 5 à 7 recommandations spécifiques pour améliorer la performance et réduire le risque du portefeuille. 
    Tenez compte des éléments suivants dans vos recommandations :
    1. La performance relative par rapport au S&P 500
    2. Le niveau de risque du portefeuille
    3. La diversification actuelle du portefeuille
    4. Les tendances récentes du marché
    5. Les opportunités potentielles dans différents secteurs

    Pour chaque recommandation, fournissez une brève explication de son raisonnement et de son impact potentiel.
    """)

    elements.append(create_paragraph("Recommandations", 'Heading2'))
    elements.append(create_paragraph(recommendations_text, 'BodyText'))

    return elements

def generate_future_outlook(portfolio, portfolio_data, returns, weights):
    elements = []
    
    # Calculer quelques métriques supplémentaires pour l'analyse
    sector_allocation = calculate_sector_allocation(portfolio)
    stock_performance = calculate_stock_performance(portfolio_data)
    
    # Calculer la volatilité du portefeuille
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
    
    # Calculer le rendement total du portefeuille
    df = pd.DataFrame(portfolio_data)
    total_return = (df.iloc[-1] / df.iloc[0] - 1).sum()
    
    # Générer le texte des perspectives futures
    outlook_text = generate_ai_content(f"""
    En vous basant sur les données suivantes :
    - Allocation sectorielle : {sector_allocation}
    - Performance des actions : {stock_performance}
    - Rendement total du portefeuille : {total_return:.2%}
    - Volatilité du portefeuille : {portfolio_volatility:.2%}

    Générez des perspectives futures pour le portefeuille. Incluez :
    1. Une analyse des tendances économiques et de marché qui pourraient affecter le portefeuille.
    2. Des prévisions pour les secteurs représentés dans le portefeuille.
    3. Des recommandations pour des ajustements potentiels du portefeuille.
    4. Une discussion sur les risques potentiels et les opportunités à venir.
    5. Des suggestions pour diversifier davantage le portefeuille si nécessaire.
    """)
    
    elements.append(create_paragraph("Perspectives Futures", 'Heading2'))
    elements.append(create_paragraph(outlook_text, 'BodyText'))
    
    return elements


def calculate_sector_allocation(portfolio):
    sector_weights = {}
    for stock in portfolio['stocks']:
        ticker = yf.Ticker(stock['symbol'])
        info = ticker.info
        sector = info.get('sector', 'Unknown')
        weight = float(stock['weight']) / 100
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    return sector_weights

def calculate_stock_performance(portfolio_data):
    df = pd.DataFrame(portfolio_data)
    return {symbol: (data.iloc[-1] / data.iloc[0] - 1) for symbol, data in df.items()}

>>>

./agent_h.py :
<<<
import openai
import yfinance as yf

class MarketSentimentAgent:
    def __init__(self):
        self.client = openai.OpenAI()

    def analyze_sentiment(self, ticker, summary="Aucun résumé fourni"):
        # Récupérer les dernières nouvelles sur le ticker
        stock = yf.Ticker(ticker)
        news = stock.news[:5]  # Prendre les 5 dernières nouvelles

        # Analyser le sentiment
        news_texts = [f"Title: {item['title']}\nSummary: {summary}" for item in news]
        combined_text = "\n\n".join(news_texts)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Vous êtes un analyste de sentiment du marché financier. Analysez les nouvelles suivantes concernant une action et fournissez une analyse globale du sentiment."},
                {"role": "user", "content": f"Analysez le sentiment pour {ticker} basé sur ces nouvelles récentes:\n\n{combined_text}"}
            ],
            temperature=0.7,
            max_tokens=800
        )

        return response.choices[0].message.content.strip()

market_sentiment_agent = MarketSentimentAgent()

>>>

./selecteur_modele_ia.py :
<<<

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
>>>

./agent_c.py :
<<<
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
>>>

./Dockerfile :
<<<
FROM python:3.12.5 AS builder
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

# OU pour les images basées sur debian/ubuntu
# If you're using a Debian/Ubuntu-based image
RUN apt-get update && apt-get install -y ca-certificates fuse3 sqlite3

# Copy the LiteFS binary into your container
COPY --from=flyio/litefs:0.5 /usr/local/bin/litefs /usr/local/bin/litefs
ENTRYPOINT ["litefs", "mount"]

FROM python:3.12.5-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
CMD ["gunicorn", "--timeout", "180", "wsgi:app", "--bind", "0.0.0.0:8080"]

>>>

./agent_g.py :
<<<
class ComplianceAgent:
    def __init__(self):
        self.regulations = {
            "max_single_stock_exposure": 0.30,
            "min_stocks_count": 5,
            "restricted_sectors": ["Tobacco", "Gambling"],
            "esg_score_threshold": 50
        }

    def check_compliance(self, portfolio):
        violations = []
        
         # Convertir le portfolio en dictionnaire pour faciliter la vérification
        portfolio_dict = {stock['symbol']: stock['weight'] for stock in portfolio}
        
        # Vérifier l'exposition maximale à un seul stock
        for stock in portfolio:
            symbol = stock['symbol']
            weight = stock['weight'] / 100  # Convertir le pourcentage en décimal
            if weight > self.regulations["max_single_stock_exposure"]:
                violations.append(f"L'exposition à {symbol} ({weight:.2%}) dépasse le maximum autorisé ({self.regulations['max_single_stock_exposure']:.2%})")

        # Vérifier le nombre minimum de stocks
        if len(portfolio) < self.regulations["min_stocks_count"]:
            violations.append(f"Le portefeuille ne contient que {len(portfolio)} actions. Le minimum requis est de {self.regulations['min_stocks_count']}")

        rapport = f"""
Analyse de conformité

Conformité: {"Conforme" if len(violations) == 0 else "Non conforme"}
Violations:
"""
        if violations:
            for violation in violations:
                rapport += f"- {violation}\n"
        else:
            rapport += "Aucune violation détectée.\n"

        rapport += """
Conclusion:
Cette analyse de conformité fournit un aperçu des violations potentielles dans le portefeuille. Les investisseurs devraient utiliser ces informations pour ajuster leur portefeuille en fonction des réglementations en vigueur.
"""
        return rapport

        # Note: Les vérifications pour les secteurs restreints et les scores ESG nécessiteraient des données supplémentaires
        # que nous n'avons pas dans notre implémentation actuelle. Elles sont omises pour simplifier.

compliance_agent = ComplianceAgent()
>>>

./agent_b.py :
<<<
from textblob import TextBlob
import requests

class SentimentAnalysisAgent:
    def __init__(self):
        self.news_api_key = "c6cc145ad227419c88756838786b70d1"  # Remplacez par votre clé API

    def get_news(self, query):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.news_api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['articles']
        else:
            return []

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def analyze(self, company):
        news = self.get_news(company)
        sentiments = [self.analyze_sentiment(article['title'] + ' ' + article['description']) for article in news[:5]]
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        if average_sentiment > 0.1:
            sentiment_category = "Positif"
        elif average_sentiment < -0.1:
            sentiment_category = "Négatif"
        else:
            sentiment_category = "Neutre"

        rapport = f"""
Analyse de sentiment pour {company}

Nombre d'articles analysés: {len(sentiments)}
Sentiment moyen: {average_sentiment:.2f} (sur une échelle de -1 à 1)
Catégorie de sentiment: {sentiment_category}

Interprétation:
{f'Le sentiment général concernant {company} est {sentiment_category.lower()}. ' if sentiment_category != "Neutre" else f'Le sentiment général concernant {company} est neutre. '}
{'Cela pourrait indiquer une perception positive de l\'entreprise dans les médias récents.' if sentiment_category == "Positif" else 'Cela pourrait indiquer une perception négative de l\'entreprise dans les médias récents.' if sentiment_category == "Négatif" else 'Cela suggère que les opinions sont mitigées ou que les nouvelles récentes n\'ont pas eu d\'impact significatif sur la perception de l\'entreprise.'}

Articles récents analysés:
"""
        for i, article in enumerate(news[:5], 1):
            rapport += f"{i}. {article['title']}\n   Sentiment: {self.analyze_sentiment(article['title'] + ' ' + article['description']):.2f}\n\n"

        rapport += f"""
Conclusion:
Cette analyse de sentiment fournit un aperçu de la perception actuelle de {company} dans les médias. Les investisseurs devraient utiliser cette information en conjonction avec une analyse financière approfondie et une compréhension plus large du contexte de l'entreprise et de son secteur avant de prendre des décisions d'investissement.
"""
        return rapport

sentiment_agent = SentimentAnalysisAgent()
>>>

./fly.toml :
<<<
app = 'finpilot'
primary_region = 'cdg'

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = false
  auto_start_machines = true
  min_machines_running = 1

  [http_service.concurrency]
    type = "connections"
    hard_limit = 25
    soft_limit = 20

[mounts]
  source = "litefs_data"
  destination = "/var/lib/litefs"

[[services]]
  internal_port = 8080
  protocol = "tcp"

[[services.ports]]
  handlers = ["http"]
  port = 80

[[services.ports]]
  handlers = ["tls", "http"]
  port = 443

[[vm]]
  size = "shared-cpu-1x"
  memory = "512mb"

[metrics]
  port = 9091
  path = "/metrics"

[env]
  NGINX_WORKER_TIMEOUT = "300s"  # Set the worker timeout to 300 seconds (5 minutes)
>>>

./agent_e.py :
<<<
import numpy as np
import yfinance as yf
from scipy.stats import norm

class RiskManagementAgent:
    def analyze(self, portfolio, confidence_level=0.95):
        if not portfolio:
            return {"erreur": "Aucun portefeuille trouvé pour cet utilisateur"}

        tickers = [stock['symbol'] for stock in portfolio]
        weights = np.array([stock['weight'] / 100 for stock in portfolio])

        data = yf.download(tickers, period="1y")['Adj Close']
        returns = data.pct_change().dropna()

        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * np.sqrt(252)

        VaR = norm.ppf(1-confidence_level) * portfolio_std
        CVaR = -1 * (norm.pdf(norm.ppf(1-confidence_level)) / (1-confidence_level)) * portfolio_std

        rapport = f"""
Analyse de gestion des risques

Rendement du portefeuille: {portfolio_return:.2f}
Volatilité du portefeuille: {portfolio_std:.2f}
Valeur à Risque (VaR) à {confidence_level * 100:.0f}% de confiance: {-VaR:.2f}
VaR Conditionnelle (CVaR) à {confidence_level * 100:.0f}% de confiance: {-CVaR:.2f}

Conclusion:
Cette analyse fournit un aperçu des risques associés au portefeuille. Les investisseurs devraient utiliser ces informations pour ajuster leur stratégie d'investissement en fonction de leur tolérance au risque.
"""
        return rapport

risk_management_agent = RiskManagementAgent()
>>>

./agent_k.py :
<<<
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
>>>

./copilote_central.py :
<<<
#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from threading import Thread
import time
import json
from openai import OpenAI
import anthropic
from PyPDF2 import PdfReader
import uuid
import random
from datetime import datetime, timedelta
import io
from io import BytesIO
# from dotenv import load_dotenv
import os
import re
import bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, set_access_cookies
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List

from agent_a import document_agent
from agent_b import sentiment_agent
from agent_c import financial_modeling_agent
from agent_d import portfolio_optimization_agent
from agent_e import risk_management_agent
from agent_f import reporting_agent
from agent_g import compliance_agent
from agent_h import market_sentiment_agent
from agent_i import investment_recommendation_agent
from agent_j import historical_data_agent
from agent_k import user_profile_agent
from selecteur_modele_ia import SelecteurModeleIA
from generateur_report import generate_report

# Load environment variables
# load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": ["https://www.finpilot.one", "http://localhost:3000"],
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
    "supports_credentials": True
}})

@app.after_request
def add_security_headers(response):
    response.headers['X-Frame-Options'] = 'ALLOWALL'  # Permettre tous les domaines
    response.headers['Content-Security-Policy'] = "frame-ancestors *"  # Permettre tous les domaines
    return response

# Initialize OpenAI and Anthropic clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Configuration de la base de données SQLite
DATABASE = '/litefs/copilot-db.db'

# Configure JWT
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
jwt = JWTManager(app)

# Augmentez la durée de validité du token JWT
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)

functions = [
    {
        "type": "function",
        "function": {
            "name": "analyze_documents",
            "description": "Analyze and summarize financial documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": "Analyze sentiment of financial news",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "The company to analyze sentiment for"
                    }
                },
                "required": ["company"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "model_financials",
            "description": "Perform financial modeling for a given stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_portfolio",
            "description": "Optimize a portfolio of stocks",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock ticker symbols"
                    }
                },
                "required": ["tickers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_risks",
            "description": "Analyze risks for a portfolio",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock ticker symbols"
                    },
                    "portfolio_value": {
                        "type": "number",
                        "description": "Total value of the portfolio"
                    }
                },
                "required": ["tickers", "portfolio_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate a financial report",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Financial data to include in the report"
                    }
                },
                "required": ["data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_compliance",
            "description": "Check portfolio compliance with regulations",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_data": {
                        "type": "object",
                        "description": "Portfolio data to check for compliance"
                    }
                },
                "required": ["portfolio_data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_market_sentiment",
            "description": "Analyze the market sentiment for a given stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_investment_recommendation",
            "description": "Get investment recommendations based on a portfolio and risk profile",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of stock ticker symbols in the portfolio"
                    },
                    "risk_profile": {
                        "type": "string",
                        "description": "The investor's risk profile (e.g., conservative, moderate, aggressive)"
                    }
                },
                "required": ["portfolio", "risk_profile"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_previous_day_data",
            "description": "Analyze the previous day's trading data for a given stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_user_profile",
            "description": "Analyze the user's investment profile based on interactions and portfolio history",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_interactions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of recent user interactions"
                    },
                    "portfolio_history": {
                        "type": "object",
                        "description": "Historical portfolio values"
                    }
                },
                "required": ["user_interactions", "portfolio_history"]
            }
        }
    }
]

assistant = openai_client.beta.assistants.create(
    name="Finance Copilot",
    instructions="You are a financial assistant. Use the provided functions to analyze documents, sentiment, perform financial modeling, optimize portfolios, manage risks, generate reports, and check compliance.",
    model="gpt-4o",
    tools=functions
)


ai_selector = SelecteurModeleIA()

# Variable globale pour stocker les résultats des appels de fonction
financial_data = {}

def get_value(x):
    return x.iloc[0] if isinstance(x, pd.Series) else x

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def get_user_by_username(username):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        return user if user else None

@app.route('/register', methods=['POST'])
def register():
    app.logger.info(f"Received registration request: {request.json}")
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    hashed_password = hash_password(password)
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                         (username, hashed_password))
        return jsonify({"message": "User registered successfully"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists"}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    user = get_user_by_username(username)
    if user and check_password(password, user[2]):
        access_token = create_access_token(identity=user[0])
        response = jsonify(access_token=access_token)
        set_access_cookies(response, access_token)
        return response, 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401
    
    # Fonction pour enregistrer le chat
def save_chat_message(user_id, role, content):
    with sqlite3.connect('copilote.db') as conn:
        cursor = conn.cursor()
        # Convertir le contenu en JSON s'il s'agit d'un dictionnaire
        if isinstance(content, dict):
            content = json.dumps(content)
        cursor.execute("""
            INSERT INTO chat_history (user_id, role, content, timestamp)
            VALUES (?, ?, ?, datetime('now'))
        """, (user_id, role, content))
        conn.commit()

# Fonction pour récupérer l'historique des chats
def get_chat_history(user_id):
    with sqlite3.connect('copilote.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content, timestamp FROM chat_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
        """, (user_id,))
        results = cursor.fetchall()
        return [
            {
                "role": role,
                "content": json.loads(content) if content.startswith('{') else content,
                "timestamp": timestamp
            }
            for role, content, timestamp in results
        ]

class ConversationManager:
    def __init__(self):
        self.conversations = {}

    def start_conversation(self):
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            "messages": [],
            "last_updated": datetime.now()
        }
        return conversation_id

    def add_message(self, conversation_id, message):
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["messages"].append(message)
            self.conversations[conversation_id]["last_updated"] = datetime.now()

    def get_messages(self, conversation_id):
        return self.conversations.get(conversation_id, {}).get("messages", [])

    def clean_old_conversations(self, max_age_hours=24):
        now = datetime.now()
        for conv_id, conv_data in list(self.conversations.items()):
            if now - conv_data["last_updated"] > timedelta(hours=max_age_hours):
                del self.conversations[conv_id]

conversation_manager = ConversationManager()

def parse_portfolio(message):
    portfolio = {}
    pattern = r'(\d+(?:\.\d+)?)\s*(?:de|d\')\s*(\w+)'
    matches = re.findall(pattern, message.lower())
    for allocation, stock in matches:
        # Vérifiez si le symbole est valide (vous devrez implémenter cette fonction)
        if is_valid_symbol(stock):
            portfolio[stock.upper()] = float(allocation)
        else:
            return {"error": f"Invalid stock symbol: {stock}"}
    return portfolio if portfolio else {"error": "No valid portfolio information found"}

def is_valid_symbol(symbol):
    # Implémentez cette fonction pour vérifier si le symbole est valide
    # Par exemple, vous pouvez avoir une liste de symboles valides ou faire une requête à une API
    valid_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'FB', 'NVDA', 'JPM', 'JNJ', 'V']
    return symbol.upper() in valid_symbols


def execute_function(function_name, arguments, user_message):
    global financial_data
    print(f"Executing function: {function_name}")
    print(f"Arguments received: {arguments}")
    try:
        if function_name == "check_compliance":
            portfolio = parse_portfolio(user_message)
            if portfolio:
                return compliance_agent.check_compliance({"portfolio_data": portfolio})
            else:
                return {
                    "error": "missing_data",
                    "message": "Informations sur le portefeuille manquantes. Veuillez fournir les détails du portefeuille."
                }
        elif function_name == "analyze_documents":
            return document_agent.analyze(arguments.get("text", ""))
        elif function_name == "analyze_sentiment":
            return sentiment_agent.analyze(arguments.get("company", ""))
        elif function_name == "model_financials":
            ticker = arguments.get("ticker", "")
            result = financial_modeling_agent.analyze(ticker)
            financial_data[ticker] = result
            return result
        elif function_name == "optimize_portfolio":
            result = portfolio_optimization_agent.optimize(arguments.get("tickers", []))
            financial_data['portfolio_optimization'] = result
            return result
        elif function_name == "manage_risks":
            return risk_management_agent.analyze(
                arguments.get("tickers", []), 
                arguments.get("portfolio_value", 100000)
            )
        elif function_name == "generate_report":
            return reporting_agent.generate_report(financial_data)
        elif function_name == "analyze_market_sentiment":
            data = request.json
            ticker = data.get('ticker')
            sentiment = market_sentiment_agent.analyze_sentiment(ticker)
            return jsonify({"sentiment": sentiment})
        elif function_name == "get_investment_recommendation":
            return investment_recommendation_agent.get_recommendation(
                arguments.get("portfolio"),
                arguments.get("risk_profile")
            )
        elif function_name == "analyze_previous_day_data":
            return historical_data_agent.analyze_previous_day(arguments.get("ticker"))
        elif function_name == "analyze_user_profile":
            return user_profile_agent.analyze_user_profile(
                arguments.get("user_interactions"),
                arguments.get("portfolio_history")
            )
        else:
            return {"error": f"Function {function_name} not found"}
    except KeyError as e:
        return {
            "error": "missing_data",
            "message": f"Données manquantes : {str(e)}. Pouvez-vous fournir plus d'informations ?"
        }
    except Exception as e:
        return {
            "error": "execution_error",
            "message": f"Une erreur s'est produite lors de l'exécution de {function_name}: {str(e)}"
        }

# def init_db():
#     with sqlite3.connect(DATABASE) as conn:
#         conn.execute('''CREATE TABLE IF NOT EXISTS users
#                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                          username TEXT UNIQUE NOT NULL,
#                          password TEXT NOT NULL)''')
#         conn.execute('''CREATE TABLE IF NOT EXISTS portfolios
#                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                          user_id INTEGER,
#                          name TEXT NOT NULL,
#                          data TEXT NOT NULL)''')
#         conn.execute('''CREATE TABLE IF NOT EXISTS portfolio (
#                         id INTEGER PRIMARY KEY AUTOINCREMENT,
#                         user_id INTEGER,
#                         name TEXT,
#                         symbol TEXT,
#                         weight REAL,
#                         entry_price REAL,
#                         FOREIGN KEY (user_id) REFERENCES users(id))''')
#         conn.execute('''CREATE TABLE IF NOT EXISTS user_settings (
#                         user_id INTEGER PRIMARY KEY,
#                         setting_name TEXT,
#                         setting_value TEXT,
#                         FOREIGN KEY (user_id) REFERENCES users(id)
#                     )''')
#         conn.execute('''CREATE TABLE IF NOT EXISTS tasks (
#                         id INTEGER PRIMARY KEY AUTOINCREMENT,
#                         task_type TEXT NOT NULL,
#                         status TEXT NOT NULL,
#                         result TEXT
#                     )''')
#         conn.execute('''DROP TABLE IF EXISTS chat_history''')
#         conn.execute('''CREATE TABLE chat_history
#                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                          user_id INTEGER,
#                          role TEXT,
#                          content TEXT,
#                          timestamp DATETIME,
#                          FOREIGN KEY (user_id) REFERENCES users(id))''')

def init_db():
    print(f"Current working directory: {os.getcwd()}")
    print(f"DATABASE path: {DATABASE}")
    print(f"Directory contents of /litefs: {os.listdir('/litefs')}")
    print(f"Directory contents of /var/lib/litefs: {os.listdir('/var/lib/litefs')}")
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            print("Successfully connected to the database")
            # Rest of your initialization code
    except sqlite3.OperationalError as e:
        print(f"SQLite operational error: {e}")
        print(f"SQLite version: {sqlite3.sqlite_version}")
    except Exception as e:
        print(f"Unexpected error: {e}")

@app.route('/clear_chat', methods=['POST'])
@jwt_required()
def clear_chat():
    user_id = get_jwt_identity()
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        return jsonify({"message": "Chat history cleared successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error clearing chat history: {str(e)}")
        return jsonify({"error": "Failed to clear chat history"}), 500

def get_user_setting(setting_name, default_value):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT setting_value FROM user_settings WHERE setting_name = ?", (setting_name,))
        result = cursor.fetchone()
        return result[0] if result else default_value

def set_user_setting(setting_name, setting_value):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("INSERT OR REPLACE INTO user_settings (setting_name, setting_value) VALUES (?, ?)",
                     (setting_name, setting_value))

def save_portfolio(user_id, name, stocks):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        # Supprimer l'ancien portfolio
        cursor.execute("DELETE FROM portfolio WHERE user_id = ?", (user_id,))
        # Insérer le nouveau portfolio
        for stock in stocks:
            cursor.execute("""
                INSERT INTO portfolio (user_id, name, symbol, weight, entry_price)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, name, stock['symbol'], stock['weight'], stock['entry_price']))  # Changé 'entryPrice' en 'entry_price'
        conn.commit()

def get_portfolio(user_id, name=None):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        if name:
            cursor.execute("SELECT * FROM portfolio WHERE user_id = ? AND name = ?", (user_id, name))
        else:
            cursor.execute("SELECT * FROM portfolio WHERE user_id = ?", (user_id,))
        portfolio = cursor.fetchall()
    return [{"symbol": row[3], "weight": row[4], "entry_price": row[5]} for row in portfolio]

print(f"Current working directory: {os.getcwd()}")
print(f"Contents of /: {os.listdir('/')}")
print(f"Contents of /var: {os.listdir('/var')}")
print(f"Contents of /var/lib: {os.listdir('/var/lib')}")

if not os.path.exists('/litefs'):
    print("WARNING: /litefs does not exist!")
else:
    print(f"Contents of /litefs: {os.listdir('/litefs')}")

print(f"Contents of /proc/mounts: {open('/proc/mounts').read()}")
init_db()

class Agents:
    @staticmethod
    def analyze_documents(data):
        text = data.get('text', '')
        return document_agent.analyze(text)

    @staticmethod
    def analyze_sentiment(data):
        company = data.get('company', '')
        return sentiment_agent.analyze(company)

    @staticmethod
    def model_financials(data):
        ticker = data.get('ticker', '')
        return financial_modeling_agent.analyze(ticker)

    @staticmethod
    def optimize_portfolio(data):
        tickers = data.get('tickers', [])
        return portfolio_optimization_agent.optimize(tickers)

    @staticmethod
    def manage_risks(data):
        tickers = data.get('tickers', [])
        portfolio_value = data.get('portfolio_value', 100000)
        return risk_management_agent.analyze(tickers, portfolio_value)

    @staticmethod
    def generate_report(data):
        return reporting_agent.generate_report(data)

    @staticmethod
    def check_compliance(data):
        return compliance_agent.check_compliance(data)
    
def structure_data(data):
    client = openai_client.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Vous êtes un assistant chargé de structurer des données financières."},
            {"role": "user", "content": f"Veuillez structurer les données suivantes en JSON : {data}"}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def generate_verbose_response(result, function_name):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Vous êtes un assistant financier expert. Expliquez les résultats de manière détaillée et compréhensible."},
            {"role": "user", "content": f"Voici les résultats de la fonction {function_name}: {result}. Veuillez les expliquer de manière détaillée."}
        ]
    )
    return response.choices[0].message.content

def get_function_info(function_name):
    function_info = {
        "optimize_portfolio": {
            "default_value": 10000,
            "description": "Cette fonction optimise votre portefeuille. La valeur par défaut du portefeuille est de 10 000 $."
        },
        "analyze_documents": {
            "default_value": None,
            "description": "Cette fonction analyse et résume des documents financiers."
        },
        "analyze_sentiment": {
            "default_value": None,
            "description": "Cette fonction analyse le sentiment du marché pour un actif donné."
        },
        "financial_modeling": {
            "default_value": None,
            "description": "Cette fonction effectue une modélisation financière pour un titre donné."
        },
        "manage_risks": {
            "default_value": 100000,
            "description": "Cette fonction analyse les risques de votre portefeuille. La valeur par défaut du portefeuille est de 100 000 $."
        },
        "generate_report": {
            "default_value": None,
            "description": "Cette fonction génère un rapport financier basé sur les données fournies."
        },
        "check_compliance": {
            "default_value": None,
            "description": "Cette fonction vérifie la conformité de votre portefeuille avec les réglementations en vigueur."
        }
    }
    return function_info.get(function_name, {"description": "Aucune information supplémentaire disponible."})

# Gestionnaire de tâches
def task_manager():
    while True:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, task_type FROM tasks WHERE status = 'pending' LIMIT 1")
            task = cursor.fetchone()
            if task:
                task_id, task_type = task
                cursor.execute("UPDATE tasks SET status = 'processing' WHERE id = ?", (task_id,))
                conn.commit()

                # Exécution de la tâche
                result = getattr(Agents, task_type)({"task_id": task_id})

                cursor.execute("UPDATE tasks SET status = 'completed', result = ? WHERE id = ?", (result, task_id))
                conn.commit()
        time.sleep(1)  # Attente d'une seconde avant de vérifier à nouveau

# Démarrage du gestionnaire de tâches en arrière-plan
Thread(target=task_manager, daemon=True).start()

@app.route('/submit_task', methods=['POST'])
def submit_task():
    data = request.json
    task_type = data.get('task_type')
    if task_type not in dir(Agents):
        return jsonify({"error": "Invalid task type"}), 400

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO tasks (task_type, status) VALUES (?, 'pending')", (task_type,))
        task_id = cursor.lastrowid
        conn.commit()

    return jsonify({"task_id": task_id, "status": "pending"})

@app.route('/task_status/<int:task_id>', methods=['GET'])
def task_status(task_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, result FROM tasks WHERE id = ?", (task_id,))
        task = cursor.fetchone()

    if task:
        return jsonify({"task_id": task_id, "status": task[0], "result": task[1]})
    else:
        return jsonify({"error": "Task not found"}), 404

@app.route('/portfolio_analysis', methods=['POST'])
def portfolio_analysis():
    data = request.json
    task_sequence = ['analyze_documents', 'analyze_sentiment', 'model_financials', 
                     'optimize_portfolio', 'manage_risks', 'check_compliance', 'generate_report']

    results = {}
    for task in task_sequence:
        results[task] = getattr(Agents, task)(data)

    return jsonify(results)

@app.route('/test_agents', methods=['POST'])
def test_agents():
    data = request.json
    results = {}

    if 'document' in data:
        results['document_analysis'] = Agents.analyze_documents({'text': data['document']})

    if 'company' in data:
        results['sentiment_analysis'] = Agents.analyze_sentiment({'company': data['company']})

    if 'ticker' in data:
        results['financial_modeling'] = Agents.model_financials({'ticker': data['ticker']})

    if 'tickers' in data:
        results['portfolio_optimization'] = Agents.optimize_portfolio({'tickers': data['tickers']})
        results['risk_management'] = Agents.manage_risks({'tickers': data['tickers'], 'portfolio_value': data.get('portfolio_value', 100000)})

    return jsonify(results)

@app.route('/chat', methods=['POST'])
@jwt_required()
def chat():
    global financial_data
    
    user_id = get_jwt_identity()
    user_message = request.json['message']
    conversation_id = request.json.get('conversation_id')
    use_claude = request.json.get('use_claude', False)
    
    # Sauvegardez le message de l'utilisateur
    save_chat_message(user_id, 'user', user_message)

    if not conversation_id:
        conversation_id = conversation_manager.start_conversation()
        financial_data = {}  # Réinitialiser pour chaque nouvelle conversation
    
    messages = conversation_manager.get_messages(conversation_id)
    messages.append({"role": "user", "content": user_message})

    claude_tools = [
            {
                "name": func["function"]["name"],
                "description": func["function"]["description"],
                "input_schema": {
                    "type": "object",
                    "properties": func["function"]["parameters"]["properties"],
                    "required": func["function"]["parameters"]["required"]
                }
            } for func in functions
        ]

    client = anthropic_client if use_claude else openai_client
    model = "claude-3-5-sonnet-20240620" if use_claude else "gpt-4o-mini"
    tools = claude_tools if use_claude else functions

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=1024 if use_claude else None
        )

        if use_claude:
            if response.content[0].type == 'text':
                conversation_manager.add_message(conversation_id, {"role": "assistant", "content": response.content[0].text})
                return jsonify({"reply": response.content[0].text, "conversation_id": conversation_id})
            assistant_message = response.content[0]
            # Sauvegardez la réponse de l'assistant
            save_chat_message(user_id, 'assistant', assistant_message.content)

        else:
            assistant_message = response.choices[0].message
            # Sauvegardez la réponse de l'assistant
            save_chat_message(user_id, 'assistant', assistant_message.content)

        messages.append(assistant_message)

        if not getattr(assistant_message, 'tool_calls', None):
            conversation_manager.add_message(conversation_id, assistant_message)
            return jsonify({"reply": assistant_message.content, "conversation_id": conversation_id})

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Obtenir des informations sur la fonction
            function_info = get_function_info(function_name)
            
            # Structurer les données si nécessaire
            structured_args = structure_data(function_args)
            
            # Exécuter la fonction
            function_response = execute_function(function_name, structured_args, user_message)
            
            if isinstance(function_response, dict) and function_response.get("error") == "missing_data":
                conversation_manager.add_message(conversation_id, {"role": "assistant", "content": function_response["message"]})
                return jsonify({"reply": function_response["message"], "conversation_id": conversation_id})
            
            # Générer une réponse verbeuse
            verbose_response = generate_verbose_response(function_response, function_name)
            
            # Ajouter les informations importantes
            verbose_response += f"\n\nInformation importante : {function_info['description']}"

            tool_message = {
                "role": "tool",
                "name": function_name,
                "content": verbose_response,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_message)

            if use_claude:
                messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_call.id, "content": json.dumps(function_response)}]})

    # Si nous sommes arrivés ici, c'est que nous avons terminé le traitement.
    final_response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    final_message = final_response.choices[0].message
    conversation_manager.add_message(conversation_id, final_message)
    
    return jsonify({"reply": final_message.content, "conversation_id": conversation_id})

@app.route('/agent/<agent_name>', methods=['POST'])
@jwt_required()
def call_agent(agent_name):
    data = request.json
    user_id = get_jwt_identity()
    portfolio = get_portfolio(user_id)
    try:
        if agent_name == "document":
            result = document_agent.analyze(data['text'])
        elif agent_name == "sentiment":
            result = sentiment_agent.analyze(data['company'])
        elif agent_name == "financial_modeling":
            result = financial_modeling_agent.analyze(data['ticker'])
        elif agent_name == "portfolio_optimization":
            result = portfolio_optimization_agent.optimize(portfolio)
        elif agent_name == "risk_management":
            result = risk_management_agent.analyze(portfolio)
        elif agent_name == "reporting":
            result = reporting_agent.generate_report(portfolio)
            return jsonify(result), 200  # Utilisez jsonify pour convertir le résultat en JSON
        elif agent_name == "compliance":
            result = compliance_agent.check_compliance(portfolio)
        elif agent_name == "market_sentiment":
            if 'summary' not in data:
                data['summary'] = "No summary provided"
            result = market_sentiment_agent.analyze_sentiment(data['ticker'], data['summary'])
        elif agent_name == "investment_recommendation":
            portfolio = data.get('portfolio', [])
            if isinstance(portfolio, dict):
                portfolio = [portfolio]
            portfolio = [stock['symbol'].upper() if isinstance(stock, dict) and 'symbol' in stock else stock.upper() for stock in portfolio]
            risk_profile = data.get('risk_profile', 'moderate')
            result = investment_recommendation_agent.get_recommendation(portfolio, risk_profile)
        elif agent_name == "historical_data_analysis":
            result = historical_data_agent.analyze_previous_day(data['ticker'])
        elif agent_name == "user_profile_analysis":
            chat_history = get_chat_history(user_id)
            result = user_profile_agent.analyze_user_profile(portfolio, chat_history)
        else:
            return jsonify({"error": "Agent not found"}), 404
        
        # Formatage du résultat en Markdown
        formatted_result = f"""
# Résultat de l'analyse par l'agent {agent_name}

{result}

---
*Cette analyse a été générée automatiquement. Veuillez l'utiliser avec discernement.*
        """
        
        return formatted_result, 200, {'Content-Type': 'text/markdown; charset=utf-8'}
    except KeyError as e:
        app.logger.error(f"Missing required data for agent {agent_name}: {str(e)}")
        return jsonify({"error": f"Missing required data: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"Error in {agent_name} agent: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.pdf'):
        pdf_reader = PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Analyse du texte extrait avec l'agent d'analyse de documents
        result = analyze_financial_report(text)
        return jsonify(result)
    else:
        return jsonify({"error": "Invalid file type"}), 400

def analyze_financial_report(text):
    client, model = ai_selector.select_model("complex")
    
    if isinstance(client, OpenAI):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Extract key financial information from the given text."},
                {"role": "user", "content": text}
            ],
            functions=[{
                "name": "extract_financial_data",
                "description": "Extract key financial data from the text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "revenue": {"type": "number"},
                        "net_income": {"type": "number"},
                        "ebitda": {"type": "number"},
                        "risks": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["revenue", "net_income", "ebitda", "risks"]
                }
            }],
            function_call={"name": "extract_financial_data"}
        )
        return json.loads(response.choices[0].message.function_call.arguments)
    elif isinstance(client, anthropic.Anthropic):
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Extract key financial information from the given text."},
                {"role": "user", "content": f"Extract the following information from this financial report: revenue, net income, EBITDA, and key risks. Present the results in JSON format.\n\nReport:\n{text}"}
            ]
        )
        # Parse the JSON from the response
        return json.loads(response.content[0].text)

@app.route('/market_analysis', methods=['POST'])
def market_analysis():
    data = request.json
    tickers = data.get('tickers', [])
    
    results = {}
    for ticker in tickers:
        sentiment = sentiment_agent.analyze(ticker)
        financials = financial_modeling_agent.analyze(ticker)
        results[ticker] = {
            "sentiment": sentiment,
            "financials": financials
        }
    
    return jsonify(results)

@app.route('/news_impact', methods=['POST'])
def news_impact():
    data = request.json
    tickers = data.get('tickers', [])
    
    news_impacts = {}
    for ticker in tickers:
        news = sentiment_agent.get_recent_news(ticker)
        impact = sentiment_agent.analyze_news_impact(news)
        news_impacts[ticker] = impact
    
    return jsonify(news_impacts)

@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier n'a été fourni"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400
    if file and file.filename.endswith('.pdf'):
        pdf_reader = PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Analyse du texte extrait avec l'agent d'analyse de documents
        result = agent_document.analyser_rapport_financier(text)
        
        # Convertir le résultat Pydantic en dictionnaire pour la sérialisation JSON
        result_dict = result.dict()
        
        # Ajuster la structure du résultat pour correspondre à l'ancienne structure si nécessaire
        adjusted_result = {
            "resume": result_dict["resume"],
            "metriques_cles": {
                "chiffre_affaires": result_dict["metriques_cles"]["chiffre_affaires"],
                "benefice_net": result_dict["metriques_cles"]["benefice_net"],
                "ebitda": result_dict["metriques_cles"]["ebitda"],
                "risques": result_dict["metriques_cles"]["risques"]
            }
        }
        
        return jsonify(adjusted_result)
    else:
        return jsonify({"error": "Type de fichier invalide"}), 400
    
@app.route('/clean_conversations', methods=['POST'])
def clean_conversations():
    conversation_manager.clean_old_conversations()
    return jsonify({"message": "Old conversations cleaned"})

@app.route('/settings', methods=['GET', 'POST'])
@jwt_required()
def settings():
    user_id = get_jwt_identity()
    if request.method == 'POST':
        data = request.json
        try:
            for key, value in data.items():
                set_user_setting(user_id, key, json.dumps(value))
            return jsonify({"message": "Settings updated successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        try:
            settings = {
                "default_portfolio_value": float(get_user_setting(user_id, "default_portfolio_value", 100000)),
                "risk_tolerance": get_user_setting(user_id, "risk_tolerance", "moderate"),
                "preferred_sectors": json.loads(get_user_setting(user_id, "preferred_sectors", "[]"))
            }
            return jsonify(settings)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

def get_user_setting(user_id, setting_name, default_value):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT setting_value FROM user_settings WHERE user_id = ? AND setting_name = ?", (user_id, setting_name))
        result = cursor.fetchone()
        return result[0] if result else default_value

def set_user_setting(user_id, setting_name, setting_value):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("INSERT OR REPLACE INTO user_settings (user_id, setting_name, setting_value) VALUES (?, ?, ?)",
                     (user_id, setting_name, setting_value))
        
def generate_ai_content(prompt):
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

@app.route('/portfolio', methods=['GET', 'POST'])
@jwt_required()
def portfolio():
    current_user = get_jwt_identity()
    if request.method == 'POST':
        data = request.json
        try:
            save_portfolio(current_user, data['name'], data['stocks'])
            return jsonify({"message": "Portfolio sauvegardé avec succès"}), 200
        except sqlite3.Error as e:
            print(f"Erreur SQLite lors de la sauvegarde du portfolio: {e}")
            return jsonify({"error": "Erreur lors de la sauvegarde du portfolio"}), 500
    elif request.method == 'GET':
        try:
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM portfolio WHERE user_id = ?", (current_user,))
                portfolio = cursor.fetchall()
                if not portfolio:
                    return jsonify({"name": "default", "stocks": []})
                return jsonify({"name": "default", "stocks": [{"symbol": row[3], "weight": row[4], "entry_price": row[5]} for row in portfolio]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/latest_price', methods=['GET'])
def latest_price():
    symbol = request.args.get('symbol')
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            latest_price = data['Close'].iloc[-1]
            return jsonify({"symbol": symbol, "price": float(latest_price)})
        else:
            return jsonify({"error": "No data available"}), 404
    except Exception as e:
        app.logger.error(f"Error fetching latest price for {symbol}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/historical_prices', methods=['GET'])
def historical_prices():
    symbol = request.args.get('symbol')
    start_date = request.args.get('startDate')
    end_date = request.args.get('endDate')
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        return jsonify(data.to_dict(orient='index'))
    except Exception as e:
        app.logger.error(f"Error fetching historical prices for {symbol}: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/ticker_metadata', methods=['GET'])
def ticker_metadata():
    symbol = request.args.get('symbol')
    try:
        response = client.get_ticker_metadata(symbol)
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error fetching metadata for {symbol}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/news', methods=['GET'])
def get_news():
    tickers = request.args.get('tickers', '').split(',')
    tickers = [ticker.strip() for ticker in tickers if ticker.strip()]
    print(f"Tickers reçus pour les nouvelles : {tickers}")
    if not tickers:
        print("Aucun ticker fourni pour les nouvelles")
        return jsonify({"message": "Aucun ticker fourni"}), 200
    try:
        news = []
        for ticker in tickers:
            ticker_news = sentiment_agent.get_news(ticker)
            news.extend(ticker_news[:3])  # Limiter à 3 nouvelles par ticker
        return jsonify(news)
    except Exception as e:
        app.logger.error(f"Error fetching news: {str(e)}")
        return jsonify({"error": "Service de nouvelles temporairement indisponible"}), 503

@app.route('/backtest', methods=['POST'])
@jwt_required()
def backtest():
    data = request.json
    portfolio = data.get('portfolio', {})
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    if not portfolio or 'stocks' not in portfolio or not portfolio['stocks']:
        return jsonify({"error": "Portfolio is empty or invalid"}), 400

    # Télécharger les données historiques
    stocks_data = {}
    weights = {}
    for stock in portfolio['stocks']:
        symbol = stock.get('symbol')
        weight = float(stock.get('weight', 0)) / 100  # Convertir en décimal
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        if hist.empty:
            return jsonify({"error": f"No historical data available for {symbol}"}), 400
        stocks_data[symbol] = hist['Close']
        weights[symbol] = weight

    # Créer un DataFrame avec les prix de clôture de tous les stocks
    df = pd.DataFrame(stocks_data)

    # Calculer les rendements journaliers
    returns = df.pct_change().fillna(0)

    # Calculer les rendements pondérés du portefeuille
    portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)

    # Calculer la valeur du portefeuille au fil du temps
    initial_value = 10000
    portfolio_values = (1 + portfolio_returns).cumprod() * initial_value

    # Calculer les métriques de performance
    total_return = (portfolio_values.iloc[-1] / initial_value) - 1
    days = len(returns)
    annualized_return = (1 + total_return) ** (252 / days) - 1
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
    risk_free_rate = 0.02  # Taux sans risque supposé de 2%
    sharpe_ratio = (annualized_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0

    results = {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "volatility": float(portfolio_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "portfolio_values": portfolio_values.tolist()
    }

    return jsonify(results)

@app.route('/chat_history', methods=['GET', 'POST'])
@jwt_required()
def chat_history():
    user_id = get_jwt_identity()

    if request.method == 'GET':
        try:
            history = get_chat_history(user_id)
            return jsonify(history), 200
        except Exception as e:
            app.logger.error(f"Error retrieving chat history: {str(e)}")
            return jsonify({"error": "Failed to retrieve chat history"}), 500

    elif request.method == 'POST':
        data = request.json
        try:
            save_chat_message(user_id, 'user', data)
            return jsonify({"message": "Chat history updated successfully"}), 200
        except Exception as e:
            app.logger.error(f"Error saving chat history: {str(e)}")
            return jsonify({"error": "Failed to save chat history"}), 500

@app.route('/compare_portfolios', methods=['POST'])
@jwt_required()
def compare_portfolios():
    data = request.json
    print("Received data in compare_portfolios:", data)
    
    portfolio = data['portfolio']
    benchmark = data['benchmark']
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    print("Type of portfolio:", type(portfolio))
    print("Content of portfolio:", portfolio)

    # Vérifier si le portfolio est vide
    if not portfolio:
        return jsonify({"error": "Portfolio is empty"}), 400

    # Vérifier et ajuster les dates
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Télécharger les données historiques du portefeuille
    portfolio_data = {}
    portfolio_weights = {}

    # Vérifier si portfolio est une liste ou un dictionnaire
    if isinstance(portfolio, list):
        stocks = portfolio
    elif isinstance(portfolio, dict) and 'stocks' in portfolio:
        stocks = portfolio['stocks']
    else:
        return jsonify({"error": "Invalid portfolio structure"}), 400

    for stock in stocks:
        try:
            symbol = stock['symbol']
            weight = float(stock['weight']) / 100  # Convertir le poids en décimal
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            portfolio_data[symbol] = hist['Close']
            portfolio_weights[symbol] = weight
        except Exception as e:
            print(f"Error processing stock {stock}: {str(e)}")
            return jsonify({"error": f"Error processing stock {stock}: {str(e)}"}), 400

    # Vérifier si des données ont été récupérées
    if not portfolio_data:
        return jsonify({"error": "No data could be retrieved for the given portfolio"}), 400

    # Calculer les rendements du portefeuille
    portfolio_returns = pd.DataFrame(portfolio_data).pct_change().dropna()
    weighted_returns = portfolio_returns.mul(pd.Series(portfolio_weights))
    portfolio_return = weighted_returns.sum(axis=1)

    # Calculer le rendement total et annualisé du portefeuille
    total_return = (1 + portfolio_return).prod() - 1
    days = len(portfolio_return)
    annualized_return = (1 + total_return) ** (252 / days) - 1

    # Télécharger les données historiques du benchmark
    benchmark_ticker = yf.Ticker(benchmark)
    benchmark_data = benchmark_ticker.history(start=start_date, end=end_date)
    benchmark_returns = benchmark_data['Close'].pct_change().dropna()

    # Calculer le rendement total et annualisé du benchmark
    benchmark_total_return = (1 + benchmark_returns).prod() - 1
    benchmark_annualized_return = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1

    # Calculer les volatilités
    portfolio_volatility = portfolio_return.std() * np.sqrt(252)
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

    # Calculer les ratios de Sharpe (en supposant un taux sans risque de 2%)
    risk_free_rate = 0.02
    portfolio_sharpe = (annualized_return - risk_free_rate) / portfolio_volatility
    benchmark_sharpe = (benchmark_annualized_return - risk_free_rate) / benchmark_volatility

    # Calculer la performance cumulée
    portfolio_cumulative = (1 + portfolio_return).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()

    results = {
        "portfolio_return": float(annualized_return),
        "benchmark_return": float(benchmark_annualized_return),
        "portfolio_volatility": float(portfolio_volatility),
        "benchmark_volatility": float(benchmark_volatility),
        "portfolio_sharpe": float(portfolio_sharpe),
        "benchmark_sharpe": float(benchmark_sharpe),
        "portfolio_cumulative": portfolio_cumulative.tolist(),
        "benchmark_cumulative": benchmark_cumulative.tolist()
    }

    return jsonify(results)


@app.route('/simulate_scenario', methods=['POST'])
@jwt_required()
def simulate_scenario():
    data = request.json
    portfolio = data['portfolio']
    scenario = data['scenario']
    
    scenarios = {
        "market_crash": {"mean": -0.001, "volatility": 0.03},
        "bull_market": {"mean": 0.0008, "volatility": 0.015},
        "high_inflation": {"mean": 0.0003, "volatility": 0.02},
    }
    
    if scenario not in scenarios:
        return jsonify({"error": "Invalid scenario"}), 400
    
    scenario_params = scenarios[scenario]
    
    initial_value = 10000
    days = 252  # 1 year of trading days
    
    daily_returns = np.random.normal(scenario_params['mean'], scenario_params['volatility'], days)
    cumulative_returns = (1 + daily_returns).cumprod()
    portfolio_values = initial_value * cumulative_returns
    final_value = portfolio_values[-1]
    
    results = {
        "scenario": scenario,
        "initial_value": initial_value,
        "final_value": final_value,
        "total_return": (final_value / initial_value) - 1,
        "daily_returns": daily_returns.tolist(),
        "portfolio_values": portfolio_values.tolist()
    }
    
    return jsonify(results)


@app.route('/generate_report', methods=['POST'])
@jwt_required()
def generate_report_route():
    data = request.json
    return generate_report(data)

@app.route('/update_portfolio_value', methods=['POST'])
@jwt_required()
def update_portfolio_value():
    data = request.json
    user_id = get_jwt_identity()
    new_value = data.get('portfolio_value')
    
    if not new_value:
        return jsonify({"error": "Portfolio value is required"}), 400
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("INSERT OR REPLACE INTO user_settings (user_id, setting_name, setting_value) VALUES (?, 'portfolio_value', ?)",
                         (user_id, str(new_value)))
        return jsonify({"message": "Portfolio value updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_portfolio_value', methods=['GET'])
@jwt_required()
def get_portfolio_value():
    user_id = get_jwt_identity()
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT setting_value FROM user_settings WHERE user_id = ? AND setting_name = 'portfolio_value'", (user_id,))
            result = cursor.fetchone()
            if result:
                return jsonify({"portfolio_value": float(result[0])}), 200
            else:
                return jsonify({"portfolio_value": 100000}), 200  # Valeur par défaut
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
>>>

./agent_a.py :
<<<
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
>>>


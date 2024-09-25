# Bibliothèques standard
import os
import base64
import json
import re
import textwrap
from datetime import datetime, timedelta
from io import BytesIO
import time

# Bibliothèques tierces
import numpy as np
import pandas as pd
import yfinance as yf
import anthropic
from flask import jsonify
from tqdm import tqdm
from sklearn.decomposition import PCA
import scipy.stats as stats
from functools import lru_cache, wraps
import asyncio
import aiohttp
from fredapi import Fred

# Bibliothèques de visualisation
import plotly.express as px
from plotly import graph_objects as go
from plotly import figure_factory as ff

# Bibliothèques de génération de rapports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, portrait
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image

anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
fred_api_key = os.environ.get('FRED_API_KEY')  # Stockez votre clé API dans une variable d'environnement
fred = Fred(api_key=fred_api_key)

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} a pris {execution_time:.2f} secondes")
        return result
    return wrapper

@timing_decorator
@lru_cache(maxsize=None)
def get_stock_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info:
            raise ValueError(f"Aucune information disponible pour {symbol}")
        return ticker
    except Exception as e:
        print(f"Erreur lors de la récupération des informations pour {symbol}: {e}")
        return None

@timing_decorator
def get_bulk_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date)
    return {symbol: data['Close'][symbol] for symbol in symbols}

@timing_decorator
async def fetch_stock_data(session, url):
    async with session.get(url) as response:
        return await response.json()

@timing_decorator
async def get_multiple_stock_data(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_stock_data(session, url) for url in urls]
        return await asyncio.gather(*tasks)

@timing_decorator
def process_section(section_func, *args):
    return section_func(*args)

@timing_decorator
def get_fred_data(series_id, start_date, end_date):
    """
    Récupère les données de FRED pour une série spécifique.

    Parameters:
    series_id (str): Identifiant de la série FRED.
    start_date (str): Date de début au format 'YYYY-MM-DD'.
    end_date (str): Date de fin au format 'YYYY-MM-DD'.

    Returns:
    DataFrame: Données de la série FRED.
    """
    try:
        data = fred.get_series(series_id, start_date, end_date)
        if data.empty:
            return pd.DataFrame({series_id: ['N/A']})
        return pd.DataFrame(data, columns=[series_id])
    except Exception as e:
        print(f"Erreur lors de la récupération des données FRED pour {series_id}: {e}")
        return pd.DataFrame({series_id: ['N/A']})


@timing_decorator
def generate_ai_content(prompt):
    try:
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Erreur lors de la génération de contenu AI : {e}")
        return "Analyse non disponible en raison d'une erreur technique."

@timing_decorator
def create_formatted_paragraph(text, style_name='Normal'):
    """
    Crée un paragraphe formaté avec des retours à la ligne et des puces.
    
    Parameters:
    text (str): Texte brut à formater.
    style_name (str): Nom du style à appliquer.
    
    Returns:
    Paragraph: Un objet Paragraph prêt à être ajouté au document.
    """
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        name='CustomStyle',
        parent=styles[style_name],
        spaceAfter=12,
        leftIndent=35,
        bulletIndent=20
    )
    
    # Remplacer les retours à la ligne par <br/>
    text = text.replace('\n', '<br/>')
    
    # Remplacer les puces manuelles par des listes HTML
    # Identifier les lignes commençant par '•' et les convertir en éléments de liste
    lines = text.split('<br/>')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('•'):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            # Supprimer le caractère de puce et ajouter l'élément de liste
            item = stripped_line.lstrip('•').strip()
            formatted_lines.append(f'<li>{item}</li>')
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            formatted_lines.append(line)
    
    if in_list:
        formatted_lines.append('</ul>')
    
    formatted_text = '<br/>'.join(formatted_lines)
    return Paragraph(formatted_text, custom_style)

@timing_decorator
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    # Remplacer les retours à la ligne multiples par un seul
    text = re.sub(r'\n+', '\n', text)
    # Diviser le texte en paragraphes
    paragraphs = text.split('\n')
    # Wrapper chaque paragraphe individuellement
    wrapped_paragraphs = [textwrap.fill(p.strip(), width=80) for p in paragraphs]
    # Rejoindre les paragraphes avec des retours à la ligne doubles
    return '\n\n'.join(wrapped_paragraphs)

@timing_decorator
def calculate_portfolio_performance(portfolio, start_date, end_date):
    """
    Calcule la performance du portefeuille sur une période donnée.

    Parameters:
    portfolio (dict): Informations sur le portefeuille, incluant les actions et leurs poids.
    start_date (str): Date de début au format 'YYYY-MM-DD'.
    end_date (str): Date de fin au format 'YYYY-MM-DD'.

    Returns:
    tuple: (portfolio_data, returns, weights)
        - portfolio_data (dict): Données de clôture des actions.
        - returns (DataFrame): Rendements quotidiens des actions.
        - weights (ndarray): Poids des actions dans le portefeuille.
    """
    symbols = [stock['symbol'] for stock in portfolio['stocks']]
    portfolio_data = get_bulk_stock_data(symbols, start_date, end_date)
    
    df = pd.DataFrame(portfolio_data)
    if df.empty:
        raise ValueError("Aucune donnée disponible pour les actions du portefeuille")
    
    returns = df.pct_change().dropna()
    weights = np.array([float(stock['weight']) / 100 for stock in portfolio['stocks']])
    
    if len(returns) < 2:
        raise ValueError("Pas assez de données pour calculer les rendements")
    
    return portfolio_data, returns, weights

@timing_decorator
def calculate_portfolio_returns(portfolio_data, weights):
    """
    Calcule les rendements pondérés du portefeuille.

    Parameters:
    portfolio_data (dict): Données de clôture des actions.
    weights (ndarray): Poids des actions dans le portefeuille.

    Returns:
    tuple: (weighted_returns, total_return, annualized_return)
        - weighted_returns (Series): Rendements pondérés quotidiens du portefeuille.
        - total_return (float): Rendement total du portefeuille.
        - annualized_return (float): Rendement annualisé du portefeuille.
    """
    df = pd.DataFrame(portfolio_data)
    returns = df.pct_change().dropna()
    weighted_returns = (returns * weights).sum(axis=1)
    total_return = (1 + weighted_returns).prod() - 1
    annualized_return = (1 + total_return) ** (safe_division(252.0, len(returns))) - 1
    return weighted_returns, total_return, annualized_return

@timing_decorator
def generate_report(data):
    """
    Génère un rapport de performance du portefeuille.

    Parameters:
    data (dict): Données nécessaires pour générer le rapport.

    Returns:
    dict: Un dictionnaire contenant le rapport encodé en base64.
    """
    start_time = time.time()
    
    portfolio = data['portfolio']
    start_date = data.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
    end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    # Calcul des rendements et des métriques de performance
    portfolio_data, returns, weights = calculate_portfolio_performance(portfolio, start_date, end_date)
    
    # Calcul des rendements pondérés une seule fois
    weighted_returns, total_return, annualized_return = calculate_portfolio_returns(portfolio_data, weights)
    
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
    elements.append(create_formatted_paragraph("1. Résumé Exécutif", 'Normal'))
    elements.append(create_formatted_paragraph("2. Vue d'Ensemble du Portefeuille", 'Normal'))
    elements.append(create_formatted_paragraph("3. Analyse de Performance", 'Normal'))
    # Ajoutez d'autres sections selon vos besoins
    elements.append(PageBreak())

    # Sections du rapport
    sections = [
        ("Résumé Exécutif", lambda p, pd, r, w, wr, tr, ar: generate_executive_summary(p, pd, r, w, wr, tr, ar)),
        ("Vue d'Ensemble du Portefeuille", lambda p, pd, r, w, wr, tr, ar: generate_portfolio_overview(p, pd, r, w)),
        ("Analyse de Performance", lambda p, pd, r, w, wr, tr, ar: generate_performance_analysis(p, pd, r, w, wr, tr, ar, start_date, end_date)),
        ("Comparaison de Performance des Actions", lambda p, pd, r, w, wr, tr, ar: generate_stock_performance_comparison(pd, w)),
        ("Contribution au Rendement", lambda p, pd, r, w, wr, tr, ar: generate_contribution_to_return(p, pd, r, w)),
        ("Ratios Supplémentaires", lambda p, pd, r, w, wr, tr, ar: generate_additional_ratios_table(p, pd, r, w, start_date, end_date)),
        ("Analyse des Risques", lambda p, pd, r, w, wr, tr, ar: generate_risk_analysis(p, pd, r, w, wr)),
        ("Corrélation des Actions", lambda p, pd, r, w, wr, tr, ar: generate_correlation_heatmap(pd)),
        ("Meilleures et Pires Performances", lambda p, pd, r, w, wr, tr, ar: generate_best_worst_performers(p, pd, r, w)),
        ("Analyse des Dividendes", lambda p, pd, r, w, wr, tr, ar: generate_dividend_table(p)),
        ("Allocation Sectorielle", lambda p, pd, r, w, wr, tr, ar: generate_sector_allocation(p, pd, r, w)),
        ("Simulation Monte Carlo", lambda p, pd, r, w, wr, tr, ar: generate_monte_carlo_simulation(p, pd, r, w, wr)),
        ("Tests de Stress", lambda p, pd, r, w, wr, tr, ar: generate_stress_tests(p, pd, r, w, wr)),
        ("Perspectives Futures", lambda p, pd, r, w, wr, tr, ar: generate_future_outlook(p, pd, r, w)),
        ("Recommandations", lambda p, pd, r, w, wr, tr, ar: generate_recommendations(p, pd, r, w, wr, tr, ar))
    ]
    total_steps = len(sections)
    
    for index, (title, function) in enumerate(sections, 1):
        section_start_time = time.time()
        
        elements.append(create_section_header(title))
        new_elements = function(portfolio, portfolio_data, returns, weights, weighted_returns, total_return, annualized_return)
        
        if isinstance(new_elements, list):
            elements.extend(new_elements)
        else:
            elements.append(create_formatted_paragraph(str(new_elements)))
        elements.append(PageBreak())
        
        section_end_time = time.time()
        section_execution_time = section_end_time - section_start_time
        
        print(f"Section '{title}' traitée en {section_execution_time:.2f} secondes")
        print(f"Progression : {index}/{len(sections)} sections traitées")
    
    # Glossaire et avertissements
    elements.append(create_section_header("Glossaire"))
    elements.extend(generate_glossary())
    
    elements.append(create_section_header("Avertissements et Divulgations"))
    elements.append(create_formatted_paragraph(generate_disclaimer(), 'Normal'))

    # Génération du PDF
    doc.build(elements, onFirstPage=add_page_number, onLaterPages=add_page_number)
    pdf = buffer.getvalue()
    buffer.close()

    # Encodage du PDF en base64
    pdf_base64 = base64.b64encode(pdf).decode('utf-8')

    end_time = time.time()
    total_execution_time = end_time - start_time
    print(f"Temps total d'exécution: {total_execution_time:.2f} secondes")

    return {"report": pdf_base64}

@timing_decorator
def create_title_page(title, subtitle, date):
    """
    Crée une page de titre pour le rapport.

    Parameters:
    title (str): Titre du rapport.
    subtitle (str): Sous-titre du rapport.
    date (str): Date du rapport.

    Returns:
    list: Une liste d'éléments à ajouter à la page de titre.
    """
    elements = []
    styles = getSampleStyleSheet()
    
    # Ajouter le logo
    logo = Image("/app/logo.jpg", width=2*inch, height=1*inch)
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

def calculate_portfolio_metrics(portfolio_data, weights, risk_free_rate=0.02):
    df = pd.DataFrame(portfolio_data)
    weighted_returns = (df.pct_change() * weights).sum(axis=1)
    cumulative_returns = (1 + weighted_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    time_period = (df.index[-1] - df.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / time_period) - 1
    volatility = weighted_returns.std() * np.sqrt(252)
    sharpe_ratio = safe_division((annualized_return - risk_free_rate), volatility)
    return total_return, annualized_return, volatility, sharpe_ratio, weighted_returns, cumulative_returns


def calculate_benchmark_metrics(benchmark_data, risk_free_rate=0.02):
    returns = benchmark_data.pct_change()
    cumulative_returns = (1 + returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    time_period = (benchmark_data.index[-1] - benchmark_data.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / time_period) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = safe_division((annualized_return - risk_free_rate), volatility)
    return total_return, annualized_return, volatility, sharpe_ratio, cumulative_returns

def generate_executive_summary(portfolio, portfolio_data, returns, weights, weighted_returns, total_return, annualized_return):
    """
    Génère le résumé exécutif du rapport.
    
    Parameters:
    portfolio (dict): Informations sur le portefeuille.
    portfolio_data (dict): Données de clôture des actions.
    returns (DataFrame): Rendements quotidiens des actions.
    weights (ndarray): Poids des actions dans le portefeuille.
    weighted_returns (Series): Rendements pondérés quotidiens du portefeuille.
    total_return (float): Rendement total du portefeuille.
    annualized_return (float): Rendement annualisé du portefeuille.
    
    Returns:
    list: Éléments à ajouter au rapport.
    """
    total_return, annualized_return, portfolio_volatility, sharpe_ratio, _, cumulative_returns = calculate_portfolio_metrics(portfolio_data, weights)
    
    sp500_data = get_sp500_data(portfolio_data[list(portfolio_data.keys())[0]].index[0], portfolio_data[list(portfolio_data.keys())[0]].index[-1])
    sp500_return, sp500_annualized_return, sp500_volatility, sp500_sharpe = calculate_benchmark_metrics(sp500_data)
    
    summary = (
        f"Résumé Exécutif\n"
        f"Ce rapport présente une analyse détaillée de la performance du portefeuille sur la période du "
        f"{portfolio_data[list(portfolio_data.keys())[0]].index[0].strftime('%d/%m/%Y')} au "
        f"{portfolio_data[list(portfolio_data.keys())[0]].index[-1].strftime('%d/%m/%Y')}.\n\n"
        f"Points clés :\n"
        f"• Rendement total du portefeuille : {total_return:.2%}\n"
        f"• Rendement annualisé : {annualized_return:.2%}\n"
        f"• Volatilité annualisée : {portfolio_volatility:.2%}\n"
        f"• Ratio de Sharpe : {sharpe_ratio:.2f}\n\n"
        f"Comparaison avec le S&P 500 :\n"
        f"• Rendement total S&P 500 : {sp500_return:.2%}\n"
        f"• Rendement annualisé S&P 500 : {sp500_annualized_return:.2%}\n"
        f"• Volatilité S&P 500 : {sp500_volatility:.2%}\n"
        f"• Ratio de Sharpe S&P 500 : {sp500_sharpe:.2f}\n\n"
        f"Le portefeuille a {'sous-performé' if total_return < sp500_return else 'sur-performé'} l'indice S&P 500 sur la période, avec un "
        f"{'risque plus élevé' if portfolio_volatility > sp500_volatility else 'risque plus faible'}."
    )
    
    elements = []
    elements.append(create_formatted_paragraph(summary, 'BodyText'))
    
    # Ajout de l'analyse générée par l'IA
    try:
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
    except Exception as e:
        print(f"Erreur lors de la génération de l'analyse AI: {e}")
        additional_analysis = "Analyse supplémentaire indisponible."
    
    elements.append(create_formatted_paragraph(additional_analysis, 'BodyText'))
    
    return elements


def generate_portfolio_overview(portfolio, portfolio_data, returns, weights):
    """
    Génère la vue d'ensemble du portefeuille.

    Parameters:
    portfolio (dict): Informations sur le portefeuille.
    portfolio_data (dict): Données de clôture des actions.
    returns (DataFrame): Rendements quotidiens des actions.
    weights (ndarray): Poids des actions dans le portefeuille.

    Returns:
    list: Une liste d'éléments à ajouter à la section de la vue d'ensemble du portefeuille.
    """
    elements = []
    
    data = [['Titre', 'Poids', 'Prix d\'entrée', 'Prix actuel', 'Rendement']]
    for stock, weight in zip(portfolio['stocks'], weights):
        symbol = stock['symbol']
        entry_price = float(stock['entry_price'])
        current_price = pd.DataFrame(portfolio_data)[symbol].iloc[-1]
        stock_return = (safe_division(current_price, entry_price) - 1)
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
    portfolio_value = (pd.DataFrame(portfolio_data) * weights).sum(axis=1)
    fig.add_trace(go.Scatter(x=pd.DataFrame(portfolio_data).index, y=portfolio_value,
                             mode='lines', name='Valeur du Portefeuille'))
    fig.update_layout(title="Évolution de la Valeur du Portefeuille",
                      xaxis_title="Date", yaxis_title="Valeur")
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=300))
    
    return elements

def generate_glossary():
    """
    Génère le glossaire du rapport.

    Returns:
    list: Une liste d'éléments à ajouter à la section du glossaire.
    """
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
    """
    Génère les avertissements et divulgations du rapport.

    Returns:
    str: Le texte des avertissements et divulgations.
    """
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
    """
    Génère la comparaison de performance des actions.

    Parameters:
    portfolio_data (dict): Données de clôture des actions.
    weights (ndarray): Poids des actions dans le portefeuille.

    Returns:
    list: Une liste d'éléments à ajouter à la section de la comparaison de performance des actions.
    """
    elements = []
    
    stock_returns = pd.DataFrame(portfolio_data).pct_change().mean() * 252
    
    stock_performance = list(zip(stock_returns.index, stock_returns.values * 100))
    stock_performance.sort(key=lambda x: x[1], reverse=True)
    
    fig = go.Figure([go.Bar(
        x=[s[0] for s in stock_performance],
        y=[s[1] for s in stock_performance],
        text=[f"{s[1]:.2%}%" for s in stock_performance],
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_contribution_to_return(portfolio, portfolio_data, returns, weights):
    """
    Génère la contribution de chaque action au rendement total du portefeuille.

    Parameters:
    portfolio (dict): Informations sur le portefeuille.
    portfolio_data (dict): Données de clôture des actions.
    returns (DataFrame): Rendements quotidiens des actions.
    weights (ndarray): Poids des actions dans le portefeuille.

    Returns:
    list: Une liste d'éléments à ajouter à la section de la contribution de chaque action au rendement total du portefeuille.
    """
    elements = []
    
    total_return = (safe_division(pd.DataFrame(portfolio_data).iloc[-1], pd.DataFrame(portfolio_data).iloc[0]) - 1).sum()
    contributions = []
    for stock, weight in zip(portfolio['stocks'], weights):
        symbol = stock['symbol']
        stock_return = safe_division(pd.DataFrame(portfolio_data)[symbol].iloc[-1], pd.DataFrame(portfolio_data)[symbol].iloc[0]) - 1
        contribution = stock_return * weight
        contributions.append((symbol, contribution, safe_division(contribution, total_return)))
    
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
    {', '.join([f"{s[0]}: {s[1]:.2%}%" for s in contributions])}
    Identifiez les actions qui ont le plus contribué positivement et négativement, et expliquez l'impact de la pondération sur ces contributions.
    """)
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements


def get_sp500_returns(start_date, end_date):
    """
    Récupère les rendements quotidiens de l'indice S&P 500 sur une période donnée.

    Parameters:
    start_date (str): Date de début au format 'YYYY-MM-DD'.
    end_date (str): Date de fin au format 'YYYY-MM-DD'.

    Returns:
    Series: Rendements quotidiens de l'indice S&P 500.
    """
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(start=start_date, end=end_date)['Close']
    return sp500_data.pct_change().dropna()

def get_sp500_data(start_date, end_date):
    """
    Récupère les données de clôture de l'indice S&P 500 sur une période donnée.

    Parameters:
    start_date (str): Date de début au format 'YYYY-MM-DD'.
    end_date (str): Date de fin au format 'YYYY-MM-DD'.

    Returns:
    Series: Données de clôture de l'indice S&P 500.
    """
    sp500 = yf.Ticker("^GSPC")
    sp500_data = sp500.history(start=start_date, end=end_date)['Close']
    return sp500_data

def generate_additional_ratios_table(portfolio, portfolio_data, returns, weights, start_date, end_date):
    """
    Génère un tableau de ratios supplémentaires du portefeuille.

    Parameters:
    portfolio (dict): Informations sur le portefeuille.
    portfolio_data (dict): Données de clôture des actions.
    returns (DataFrame): Rendements quotidiens des actions.
    weights (ndarray): Poids des actions dans le portefeuille.
    start_date (str): Date de début au format 'YYYY-MM-DD'.
    end_date (str): Date de fin au format 'YYYY-MM-DD'.

    Returns:
    list: Une liste d'éléments à ajouter à la section des ratios supplémentaires du portefeuille.
    """
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
    
    beta = safe_division(np.cov(portfolio_returns, benchmark_returns)[0][1], np.var(benchmark_returns))
    alpha = np.mean(excess_returns) - beta * np.mean(benchmark_excess_returns)
    tracking_error = np.std(portfolio_returns - benchmark_returns) * np.sqrt(252)
    information_ratio = safe_division((np.mean(portfolio_returns - benchmark_returns) * 252), tracking_error)
    downside_returns = np.minimum(excess_returns - np.mean(excess_returns), 0)
    sortino_ratio = safe_division(np.mean(excess_returns), (np.std(downside_returns)) * np.sqrt(252))
    
    ratios = {
        "Beta": beta if not np.isnan(beta) else "N/A",
        "Alpha": alpha * 252 if not np.isnan(alpha) else "N/A",
        "Tracking Error": tracking_error if not np.isnan(tracking_error) else "N/A",
        "Information Ratio": information_ratio if not np.isnan(information_ratio) else "N/A",
        "Sortino Ratio": sortino_ratio if not np.isnan(sortino_ratio) else "N/A"
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def analyze_portfolio_correlation(returns):
    """
    Analyse la corrélation des actions du portefeuille.

    Parameters:
    returns (DataFrame): Rendements quotidiens des actions.

    Returns:
    tuple: (correlation_matrix, explained_variance_ratio)
        - correlation_matrix (DataFrame): Matrice de corrélation des actions.
        - explained_variance_ratio (ndarray): Proportion de la variance expliquée par chaque composante principale.
    """
    correlation_matrix = returns.corr()
    pca = PCA()
    pca.fit(returns)
    explained_variance_ratio = pca.explained_variance_ratio_
    return correlation_matrix, explained_variance_ratio

def generate_correlation_heatmap(portfolio_data):
    """
    Génère un heatmap de corrélation des actions du portefeuille.

    Parameters:
    portfolio_data (dict): Données de clôture des actions.

    Returns:
    list: Une liste d'éléments à ajouter à la section du heatmap de corrélation des actions du portefeuille.
    """
    elements = []
    
    returns = pd.DataFrame(portfolio_data).pct_change().dropna()
    correlation_matrix, explained_variance_ratio = analyze_portfolio_correlation(returns)
    
    # Créer le heatmap de corrélation
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
    
    # Trouver les paires les plus et moins corrélées
    corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, 1)]
    max_corr = np.max(corr_values)
    min_corr = np.min(corr_values)
    max_corr_pair = np.unravel_index(np.argmax(correlation_matrix.values), correlation_matrix.shape)
    min_corr_pair = np.unravel_index(np.argmin(correlation_matrix.values), correlation_matrix.shape)
    
    # Calculer la corrélation moyenne
    avg_correlation = np.mean(corr_values)
    
    # Calculer le nombre de composantes principales nécessaires pour expliquer 80% de la variance
    num_components = np.sum(np.cumsum(explained_variance_ratio) <= 0.8) + 1
    
    explanation = generate_ai_content(f"""
    Analysez la matrice de corrélation des actions du portefeuille en vous basant sur les informations suivantes :
    
    1. Corrélation moyenne du portefeuille : {avg_correlation:.2f}
    2. Paire d'actions la plus corrélée : {correlation_matrix.index[max_corr_pair[0]]} et {correlation_matrix.index[max_corr_pair[1]]} (corrélation de {max_corr:.2f})
    3. Paire d'actions la moins corrélée : {correlation_matrix.index[min_corr_pair[0]]} et {correlation_matrix.index[min_corr_pair[1]]} (corr��lation de {min_corr:.2f})
    4. Nombre de composantes principales nécessaires pour expliquer 80% de la variance : {num_components}
    5. Proportion de la variance expliquée par la première composante principale : {explained_variance_ratio[0]:.2%}
    
    Basez votre analyse sur ces points :
    1. Interprétez la corrélation moyenne du portefeuille. Est-elle élevée, moyenne ou faible ?
    2. Discutez des implications des paires d'actions les plus et les moins corrélées.
    3. Analysez l'impact de ces corrélations sur la diversification du portefeuille.
    4. Commentez sur le nombre de composantes principales nécessaires pour expliquer 80% de la variance. Qu'est-ce que cela nous dit sur la diversification du portefeuille ?
    5. Interprétez la proportion de variance expliquée par la première composante principale. Qu'est-ce que cela implique pour le risque du portefeuille ?
    6. Suggérez des moyens spécifiques d'améliorer la diversification du portefeuille en fonction de ces corrélations et de l'analyse en composantes principales.
    7. Discutez des limites potentielles de cette analyse de corrélation (par exemple, la non-linéarité des relations entre les actifs).
    """)
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_performance_analysis(portfolio, portfolio_data, returns, weights, weighted_returns, total_return, annualized_return, start_date, end_date):
    """
    Génère l'analyse de performance du portefeuille.

    Parameters:
    portfolio (dict): Informations sur le portefeuille.
    portfolio_data (dict): Données de clôture des actions.
    returns (DataFrame): Rendements quotidiens des actions.
    weights (ndarray): Poids des actions dans le portefeuille.
    weighted_returns (Series): Rendements pondérés quotidiens du portefeuille.
    total_return (float): Rendement total du portefeuille.
    annualized_return (float): Rendement annualisé du portefeuille.
    start_date (str): Date de début au format 'YYYY-MM-DD'.
    end_date (str): Date de fin au format 'YYYY-MM-DD'.

    Returns:
    list: Une liste d'éléments à ajouter à la section de l'analyse de performance du portefeuille.
    """
    elements = []
    
    total_return, annualized_return, volatility, sharpe_ratio, _, cumulative_returns = calculate_portfolio_metrics(portfolio_data, weights)
    
    sp500_data = get_sp500_data(start_date, end_date)
    sp500_total_return, sp500_annualized_return, sp500_volatility, sp500_sharpe_ratio, sp500_cumulative_returns = calculate_benchmark_metrics(sp500_data)
    
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
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

def generate_risk_analysis(portfolio, portfolio_data, returns, weights, weighted_returns):
    elements = []
    
    var_95 = np.percentile(weighted_returns, 5)
    cvar_95 = weighted_returns[weighted_returns <= var_95].mean()
    cumulative_returns = (1 + weighted_returns).cumprod()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    
    risk_data = [
        ['Métrique', 'Valeur'],
        ['VaR (95%)', f"{var_95:.2%}"],
        ['CVaR (95%)', f"{cvar_95:.2%}"],
        ['Drawdown Maximum', f"{max_drawdown:.2%}"]
    ]

    explanation = generate_ai_content(f"""
    Analysez les métriques de risque suivantes pour le portefeuille :
    - VaR (95%) : {var_95:.2%}
    - CVaR (95%) : {cvar_95:.2%}
    - Drawdown Maximum : {max_drawdown:.2%}
    Expliquez ce que ces métriques signifient pour l'investisseur et comment elles se comparent aux benchmarks du marché.
    """)
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
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
    
    fig = ff.create_distplot([weighted_returns], ['Rendements du Portefeuille'], show_hist=False, show_rug=False)
    fig.update_layout(title="Distribution des Rendements du Portefeuille",
                      xaxis_title="Rendement", yaxis_title="Densité")
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=300))
    
    return elements

def generate_sector_allocation(portfolio, portfolio_data, returns, weights):
    elements = []

    async def get_sectors():
        urls = [f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{stock['symbol']}?modules=assetProfile" for stock in portfolio['stocks']]
        results = await get_multiple_stock_data(urls)
        sectors = {}
        for stock, result in zip(portfolio['stocks'], results):
            sector = result.get('quoteSummary', {}).get('result', [{}])[0].get('assetProfile', {}).get('sector', 'Unknown')
            sectors[stock['symbol']] = sector
        return sectors

    # Obtenir les secteurs pour chaque action
    sectors = asyncio.run(get_sectors())
    
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_best_worst_performers(portfolio, portfolio_data, returns, weights):
    elements = []
    
    df = pd.DataFrame(portfolio_data)
    stock_returns = df.pct_change().mean() * 252  # rendements annualisés
    stock_volatility = df.pct_change().std() * np.sqrt(252)  # volatilité annualisée
    stock_sharpe = safe_division(stock_returns, stock_volatility)  # ratio de Sharpe

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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_dividend_table(portfolio):
    elements = []
    
    dividend_data = [['Titre', 'Rendement du Dividende', 'Fréquence', 'Dernier Dividende']]
    for stock in portfolio['stocks']:
        ticker = get_stock_info(stock['symbol'])
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
    
    if all(row[1] == "N/A" for row in dividend_data[1:]):
        explanation = "Aucune donnée de dividende n'est disponible pour les actions de ce portefeuille. Cela peut indiquer que les entreprises ne versent pas de dividendes ou que les données sont temporairement indisponibles."
    else:
        dividend_info = "\n".join([f"{row[0]}: Rendement {row[1]}, Fréquence {row[2]}, Dernier dividende {row[3]}" for row in dividend_data[1:]])
        explanation = generate_ai_content(f"""
        Analysez la politique de dividendes des actions du portefeuille en vous basant sur les données suivantes:
        
        {dividend_info}
        
        Identifiez les actions avec les rendements de dividendes les plus élevés et les plus bas.
        Discutez de l'impact des dividendes sur le rendement total du portefeuille.
        Commentez sur la durabilité des dividendes en fonction des rendements et des montants.
        """)
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_monte_carlo_simulation(portfolio, portfolio_data, returns, weights, weighted_returns):
    elements = []
    
    # Paramètres de la simulation
    num_simulations = 1000
    num_days = 252  # un an de trading
    
    # Calcul des paramètres de la distribution log-normale
    mu = np.mean(weighted_returns)
    sigma = np.std(weighted_returns)
    
    # Simulation
    simulations = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        simulations[i] = np.random.lognormal(mu, sigma, num_days)
    simulations = np.cumprod(simulations, axis=1)
    
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
    Expliquez comment cette simulation Monte Carlo utilisant une distribution log-normale diffère d'un simple rééchantillonnage et pourquoi elle pourrait être plus appropriée pour modéliser les rendements futurs.
    """)
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_stress_tests(portfolio, portfolio_data, returns, weights, weighted_returns):
    elements = []
    
    # Définition des scénarios de stress avec impacts différenciés par secteur
    scenarios = {
        "Crise financière": {
            "Financials": -0.40,
            "Technology": -0.25,
            "Consumer Discretionary": -0.30,
            "default": -0.20
        },
        "Récession économique": {
            "Consumer Discretionary": -0.35,
            "Industrials": -0.30,
            "Materials": -0.25,
            "default": -0.15
        },
        "Pandémie": {
            "Health Care": 0.10,
            "Consumer Staples": 0.05,
            "Energy": -0.30,
            "default": -0.10
        },
        "Guerre commerciale": {
            "Industrials": -0.20,
            "Technology": -0.15,
            "Materials": -0.25,
            "default": -0.10
        },
        "Catastrophe naturelle": {
            "Energy": -0.20,
            "Utilities": -0.15,
            "Real Estate": -0.10,
            "default": -0.05
        }
    }
    
    # Obtenir les secteurs pour chaque action
    stock_sectors = {}
    for stock in portfolio['stocks']:
        ticker = get_stock_info(stock['symbol'])
        info = ticker.info
        stock_sectors[stock['symbol']] = info.get('sector', 'Unknown')
    
    # Calcul de l'impact des scénarios sur le portefeuille
    portfolio_impacts = {}
    for scenario, sector_impacts in scenarios.items():
        impact = 0
        for stock, weight in zip(portfolio['stocks'], weights):
            sector = stock_sectors[stock['symbol']]
            sector_impact = sector_impacts.get(sector, sector_impacts['default'])
            impact += sector_impact * weight
        portfolio_impacts[scenario] = impact
    
    # Création du texte d'analyse
    analysis = f"""
    Tests de Stress du Portefeuille

    Les scénarios suivants ont été simulés pour évaluer la résilience du portefeuille :
    {', '.join([f"{scenario}: {impact:.2%}" for scenario, impact in portfolio_impacts.items()])}

    Ces tests de stress montrent comment le portefeuille pourrait réagir dans différentes conditions de marché extrêmes,
    en tenant compte des impacts différenciés sur les secteurs. Il est important de noter que ces scénarios sont
    hypothétiques et ne prédisent pas nécessairement des événements futurs.
    """
    
    elements.append(create_formatted_paragraph(analysis, 'BodyText'))
    
    # Création d'un graphique pour visualiser les impacts
    fig = go.Figure(data=[go.Bar(
        x=list(scenarios.keys()),
        y=[impact * 100 for impact in portfolio_impacts.values()],
        text=[f"{impact:.2%}" for impact in portfolio_impacts.values()],
        textposition='auto',
    )])
    fig.update_layout(
        title="Impact des Scénarios de Stress sur le Portefeuille",
        xaxis_title="Scénarios",
        yaxis_title="Impact sur la Valeur du Portefeuille (%)",
        yaxis_tickformat='.2%'
    )
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    elements.append(Image(img_buffer, width=500, height=300))
    
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

def generate_recommendations(portfolio, portfolio_data, returns, weights, weighted_returns, total_return, annualized_return):
    elements = []
    
    portfolio_volatility = weighted_returns.std() * np.sqrt(252)
    
    risk_free_rate = 0.02  # Taux sans risque de 2%
    sharpe_ratio = safe_division((annualized_return - risk_free_rate), portfolio_volatility)

    # Comparer avec le S&P 500
    sp500_data = get_sp500_data(portfolio_data[list(portfolio_data.keys())[0]].index[0], portfolio_data[list(portfolio_data.keys())[0]].index[-1])
    sp500_return = (safe_division(sp500_data.iloc[-1], sp500_data.iloc[0])) - 1
    sp500_volatility = sp500_data.pct_change().std() * np.sqrt(252)

    # Calculer la corrélation moyenne entre les actions
    correlation_matrix = returns.corr()
    avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values,1)].mean()

    # Calculer les ratios P/E et la croissance des bénéfices pour chaque action
    pe_ratios = {}
    earnings_growth = {}
    for stock in portfolio['stocks']:
        ticker = get_stock_info(stock['symbol'])
        info = ticker.info
        pe_ratios[stock['symbol']] = info.get('trailingPE', 'N/A')
        earnings_growth[stock['symbol']] = info.get('earningsQuarterlyGrowth', 'N/A')

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
    - Ratios P/E des actions : {pe_ratios}
    - Croissance des bénéfices des actions : {earnings_growth}

    Générez une liste de 5 à 7 recommandations spécifiques pour améliorer la performance et réduire le risque du portefeuille. 
    Tenez compte des éléments suivants dans vos recommandations :
    1. La performance relative par rapport au S&P 500
    2. Le niveau de risque du portefeuille
    3. La diversification actuelle du portefeuille
    4. Les tendances récentes du marché
    5. Les opportunités potentielles dans différents secteurs
    6. Les ratios P/E des actions par rapport à leurs moyennes historiques ou sectorielles
    7. Les tendances de croissance des bénéfices des actions

    Pour chaque recommandation, fournissez une brève explication de son raisonnement et de son impact potentiel.
    Assurez-vous d'inclure des recommandations spécifiques sur les actions à acheter, vendre ou conserver, en vous basant sur leur valorisation et leur croissance.
    """)

    elements.append(create_section_header("Recommandations", level=2))
    elements.append(create_formatted_paragraph(recommendations_text, 'BodyText'))

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
    total_return = safe_division((pd.DataFrame(portfolio_data).iloc[-1], pd.DataFrame(portfolio_data).iloc[0]) - 1).sum()
    
    # Obtenir des données macroéconomiques actuelles
    current_date = datetime.now()
    start_date = current_date - timedelta(days=30)  # Données du dernier mois
    
    try:
        inflation = get_fred_data('CPIAUCSL', start_date, current_date)
        unemployment = get_fred_data('UNRATE', start_date, current_date)
        gdp_growth = get_fred_data('GDP', start_date, current_date)
        
        latest_inflation = inflation.iloc[-1][inflation.columns[0]]
        latest_unemployment = unemployment.iloc[-1][unemployment.columns[0]]
        latest_gdp_growth = gdp_growth.iloc[-1][gdp_growth.columns[0]]
    except Exception as e:
        print(f"Erreur lors de la récupération des données macroéconomiques : {e}")
        latest_inflation = latest_unemployment = latest_gdp_growth = "N/A"
    
    # Générer le texte des perspectives futures
    outlook_text = generate_ai_content(f"""
    En vous basant sur les données suivantes :
    - Allocation sectorielle : {sector_allocation}
    - Performance des actions : {stock_performance}
    - Rendement total du portefeuille : {total_return:.2%}
    - Volatilité du portefeuille : {portfolio_volatility:.2%}
    - Inflation actuelle : {latest_inflation}
    - Taux de chômage actuel : {latest_unemployment}%
    - Croissance du PIB actuelle : {latest_gdp_growth}%

    Générez des perspectives futures pour le portefeuille. Incluez :
    1. Une analyse des tendances économiques et de marché qui pourraient affecter le portefeuille, en tenant compte des données macroéconomiques actuelles.
    2. Des prévisions pour les secteurs représentés dans le portefeuille, en considérant leur performance récente et les conditions économiques actuelles.
    3. Des recommandations pour des ajustements potentiels du portefeuille, basées sur les perspectives sectorielles et macroéconomiques.
    4. Une discussion sur les risques potentiels et les opportunités à venir, en tenant compte de l'environnement économique actuel.
    5. Des suggestions pour diversifier davantage le portefeuille si nécessaire, en considérant les secteurs qui pourraient bien performer dans le contexte économique actuel.
    """)
    
    elements.append(create_formatted_paragraph("Perspectives Futures", 'Heading2'))
    elements.append(create_formatted_paragraph(outlook_text, 'BodyText'))
    
    return elements

def safe_division(a, b):
    if np.isscalar(a) and np.isscalar(b):
        return a / b if b != 0 else 0.0
    else:
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b!=0)

def calculate_sector_allocation(portfolio):
    sector_weights = {}
    sector_cache = {}
    for stock in portfolio['stocks']:
        if stock['symbol'] not in sector_cache:
            ticker = get_stock_info(stock['symbol'])
            info = ticker.info
            sector_cache[stock['symbol']] = info.get('sector', 'Unknown')
        sector = sector_cache[stock['symbol']]
        weight = safe_division(float(stock['weight']),100)
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    return sector_weights

def calculate_stock_performance(portfolio_data):
    df = pd.DataFrame(portfolio_data)
    return {symbol: (safe_division(data.iloc[-1], data.iloc[0]) - 1) for symbol, data in df.items()}

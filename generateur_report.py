# Bibliothèques standard
import os
import base64
import json
import re
import textwrap
from datetime import datetime, timedelta
from io import BytesIO

# Bibliothèques tierces
import numpy as np
import pandas as pd
import yfinance as yf
import anthropic
from flask import jsonify
from tqdm import tqdm
from sklearn.decomposition import PCA

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

def generate_ai_content(prompt):
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

def create_formatted_paragraph(text, style_name='Normal'):
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(
        'CustomStyle',
        parent=styles[style_name],
        spaceAfter=12,
        bulletIndent=20,
        leftIndent=35
    )
    
    # Remplacer les balises <bullet> par le caractère de puce HTML
    text = text.replace('<bullet>•</bullet>', '&bull;')
    
    # Traiter les puces
    lines = text.split('<br/>')
    formatted_lines = []
    for line in lines:
        if line.strip().startswith('•'):
            formatted_lines.append(f'<bullet>&bull;</bullet>{line.strip()[1:]}')
        else:
            formatted_lines.append(line)
    
    formatted_text = '<br/>'.join(formatted_lines)
    return Paragraph(formatted_text, custom_style)

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

def calculate_portfolio_performance(portfolio, start_date, end_date):
    portfolio_data = {}
    for stock in portfolio['stocks']:
        try:
            ticker = yf.Ticker(stock['symbol'])
            hist = ticker.history(start=start_date, end=end_date)
            portfolio_data[stock['symbol']] = hist['Close']
        except Exception as e:
            print(f"Erreur lors de la récupération des données pour {stock['symbol']}: {e}")
            continue

    df = pd.DataFrame(portfolio_data)
    returns = df.pct_change().dropna()
    weights = np.array([float(stock['weight']) / 100 for stock in portfolio['stocks']])
    
    return portfolio_data, returns, weights


def generate_report(data):
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
        ("Analyse ESG", lambda p, pd, r, w, wr, tr, ar: generate_esg_analysis(p)),
        ("Allocation Sectorielle", lambda p, pd, r, w, wr, tr, ar: generate_sector_allocation(p, pd, r, w)),
        ("Simulation Monte Carlo", lambda p, pd, r, w, wr, tr, ar: generate_monte_carlo_simulation(p, pd, r, w, wr)),
        ("Tests de Stress", lambda p, pd, r, w, wr, tr, ar: generate_stress_tests(p, pd, r, w)),
        ("Perspectives Futures", lambda p, pd, r, w, wr, tr, ar: generate_future_outlook(p, pd, r, w)),
        ("Recommandations", lambda p, pd, r, w, wr, tr, ar: generate_recommendations(p, pd, r, w, wr, tr, ar))
    ]
    total_steps = len(sections)
    
    for index, (title, function) in enumerate(sections, 1):
        elements.append(create_section_header(title))
        new_elements = function(portfolio, portfolio_data, returns, weights)
        
        if isinstance(new_elements, list):
            elements.extend(new_elements)
        else:
            elements.append(create_formatted_paragraph(str(new_elements)))
        elements.append(PageBreak())
        

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

    return {"report": pdf_base64}

def create_title_page(title, subtitle, date):
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

def calculate_portfolio_returns(portfolio_data, weights):
    df = pd.DataFrame(portfolio_data)
    returns = df.pct_change().dropna()
    weighted_returns = (returns * weights).sum(axis=1)
    total_return = (1 + weighted_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    return weighted_returns, total_return, annualized_return

def generate_executive_summary(portfolio, portfolio_data, returns, weights, weighted_returns, total_return, annualized_return):
    elements = []
    
    portfolio_volatility = weighted_returns.std() * np.sqrt(252)
    
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / portfolio_volatility

    sp500_data = get_sp500_data(portfolio_data[list(portfolio_data.keys())[0]].index[0], portfolio_data[list(portfolio_data.keys())[0]].index[-1])
    sp500_return = (sp500_data.iloc[-1] / sp500_data.iloc[0]) - 1
    sp500_volatility = sp500_data.pct_change().std() * np.sqrt(252)
    sp500_sharpe = (sp500_return - risk_free_rate) / sp500_volatility

    summary = f"""
    Résumé Exécutif

    Ce rapport présente une analyse détaillée de la performance du portefeuille sur la période du {portfolio_data[list(portfolio_data.keys())[0]].index[0].strftime('%d/%m/%Y')} au {portfolio_data[list(portfolio_data.keys())[0]].index[-1].strftime('%d/%m/%Y')}.

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
    
    elements.append(create_formatted_paragraph(summary, 'BodyText'))

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
    
    elements.append(create_formatted_paragraph(additional_analysis, 'BodyText'))
    
    return elements


def generate_portfolio_overview(portfolio, portfolio_data, returns, weights):
    elements = []
    
    data = [['Titre', 'Poids', 'Prix d\'entrée', 'Prix actuel', 'Rendement']]
    for stock, weight in zip(portfolio['stocks'], weights):
        symbol = stock['symbol']
        entry_price = float(stock['entry_price'])
        current_price = pd.DataFrame(portfolio_data)[symbol].iloc[-1]
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
    
    stock_returns = pd.DataFrame(portfolio_data).pct_change().mean() * 252
    
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_contribution_to_return(portfolio, portfolio_data, returns, weights):
    elements = []
    
    total_return = (pd.DataFrame(portfolio_data).iloc[-1] / pd.DataFrame(portfolio_data).iloc[0] - 1).sum()
    contributions = []
    for stock, weight in zip(portfolio['stocks'], weights):
        symbol = stock['symbol']
        stock_return = pd.DataFrame(portfolio_data)[symbol].iloc[-1] / pd.DataFrame(portfolio_data)[symbol].iloc[0] - 1
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def analyze_portfolio_correlation(returns):
    correlation_matrix = returns.corr()
    pca = PCA()
    pca.fit(returns)
    explained_variance_ratio = pca.explained_variance_ratio_
    return correlation_matrix, explained_variance_ratio

def generate_correlation_heatmap(portfolio_data):
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
    3. Paire d'actions la moins corrélée : {correlation_matrix.index[min_corr_pair[0]]} et {correlation_matrix.index[min_corr_pair[1]]} (corrélation de {min_corr:.2f})
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
    elements = []
    
    volatility = weighted_returns.std() * np.sqrt(252)
    cumulative_returns = (1 + weighted_returns).cumprod()
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
    
    var_95 = np.percentile(weighted_returns, 5) * np.sqrt(252)
    cvar_95 = weighted_returns[weighted_returns <= var_95].mean() * np.sqrt(252)
    cumulative_returns = (1 + weighted_returns).cumprod()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_esg_analysis(portfolio):
    elements = []
    
    def normalize_esg_score(score):
    if score is None or score == 'N/A':
        return 0
    return (score - min_score) / (max_score - min_score)  # Adapté selon l'échelle réelle

    esg_data = [['Titre', 'Score ESG', 'Environnement', 'Social', 'Gouvernance']]
    total_weight = 0
    weighted_score = 0
    
    for stock in portfolio['stocks']:
        ticker = yf.Ticker(stock['symbol'])
        info = ticker.info
        esg_score = info.get('esgScore', 'N/A')
        environment_score = info.get('environmentScore', 'N/A')
        social_score = info.get('socialScore', 'N/A')
        governance_score = info.get('governanceScore', 'N/A')
        
        if esg_score != 'N/A':
            weight = float(stock['weight']) / 100
            total_weight += weight
            weighted_score += normalize_esg_score(esg_score) * weight
        
        esg_data.append([
            stock['symbol'],
            f"{esg_score:.2f}" if esg_score != 'N/A' else 'N/A',
            f"{environment_score:.2f}" if environment_score != 'N/A' else 'N/A',
            f"{social_score:.2f}" if social_score != 'N/A' else 'N/A',
            f"{governance_score:.2f}" if governance_score != 'N/A' else 'N/A'
        ])
    
    portfolio_esg_score = weighted_score / total_weight if total_weight > 0 else None

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
    
    elements.append(create_formatted_paragraph(f"Score ESG du portefeuille : {portfolio_esg_score:.2f}", 'BodyText'))
    
    explanation = generate_ai_content(f"""
    Analysez les scores ESG du portefeuille en vous basant sur les données du tableau.
    Le score ESG moyen pondéré du portefeuille est de {portfolio_esg_score:.2f}.
    Identifiez les entreprises les plus performantes et les moins performantes en termes d'ESG.
    Discutez de l'importance des critères ESG dans la gestion de portefeuille moderne.
    Suggérez des moyens d'améliorer le profil ESG global du portefeuille.
    """)
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_monte_carlo_simulation(portfolio, portfolio_data, returns, weights, weighted_returns):
    elements = []
    
    # Paramètres de la simulation
    num_simulations = 1000
    num_days = 252  # un an de trading
    
    # Calcul des paramètres de la distribution des rendements
    mean_return = weighted_returns.mean()
    std_return = weighted_returns.std()
    
    # Simulation
    simulations = np.zeros((num_simulations, num_days))
    for i in range(num_simulations):
        simulations[i] = np.random.choice(weighted_returns, size=num_days, replace=True)
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
    return elements

def generate_stress_tests(portfolio, portfolio_data, returns, weights, weighted_returns):
    elements = []
    
    def historical_stress_test(portfolio_data, weights, start_date, end_date):
        df = pd.DataFrame(portfolio_data)
        if start_date not in df.index or end_date not in df.index:
            return None  # Ou une valeur par défaut appropriée
        period_returns = df.loc[start_date:end_date].pct_change().dropna()
        portfolio_returns = (period_returns * weights).sum(axis=1)
        return (1 + portfolio_returns).prod() - 1

    scenarios = {
        "Crise financière 2008": ("2008-09-01", "2009-03-31"),
        "Éclatement bulle tech 2000": ("2000-03-01", "2002-10-31"),
        "Crise du Covid-19": ("2020-02-20", "2020-03-23"),
    }

    stress_data = [['Scénario', 'Impact sur le Portefeuille']]
    for scenario, (start_date, end_date) in scenarios.items():
        impact = historical_stress_test(portfolio_data, weights, start_date, end_date)
        if impact is not None:
            stress_data.append([scenario, f"{impact:.2%}"])
        else:
            stress_data.append([scenario, "N/A"])
    
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
    # Utiliser les impacts calculés précédemment
    fig = go.Figure(data=[go.Bar(
        x=[scenario for scenario in scenarios.keys()],
        y=[impact[1] * 100 for impact in stress_data[1:]]  # impact[1] est l'impact en pourcentage
    )])
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
    elements.append(create_formatted_paragraph(explanation, 'BodyText'))
    
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
    
    sharpe_ratio = (annualized_return - 0.02) / portfolio_volatility  # Assuming 2% risk-free rate

    # Comparer avec le S&P 500
    sp500_data = get_sp500_data(portfolio_data[list(portfolio_data.keys())[0]].index[0], portfolio_data[list(portfolio_data.keys())[0]].index[-1])
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
    total_return = (pd.DataFrame(portfolio_data).iloc[-1] / pd.DataFrame(portfolio_data).iloc[0] - 1).sum()
    
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
    
    elements.append(create_formatted_paragraph("Perspectives Futures", 'Heading2'))
    elements.append(create_formatted_paragraph(outlook_text, 'BodyText'))
    
    return elements


def calculate_sector_allocation(portfolio):
    sector_weights = {}
    sector_cache = {}
    for stock in portfolio['stocks']:
        if stock['symbol'] not in sector_cache:
            ticker = yf.Ticker(stock['symbol'])
            info = ticker.info
            sector_cache[stock['symbol']] = info.get('sector', 'Unknown')
        sector = sector_cache[stock['symbol']]
        weight = float(stock['weight']) / 100
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
    return sector_weights

def calculate_stock_performance(portfolio_data):
    df = pd.DataFrame(portfolio_data)
    return {symbol: (data.iloc[-1] / data.iloc[0] - 1) for symbol, data in df.items()}

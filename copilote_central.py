#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import sqlite3
from threading import Thread
import time
import json
from openai import OpenAI
import anthropic
from groq import Groq
import deepl
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
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'  # Restriction plus sûre
    response.headers['Content-Security-Policy'] = "default-src 'self';"  # Politique de sécurité plus restrictive
    return response


# @app.route('/<path:path>', methods=['OPTIONS'])
# def handle_options(path):
#     response = jsonify({'status': 'OK'})
#     response.headers.add('Access-Control-Allow-Origin', 'https://www.finpilot.one')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     return response

# @app.before_request
# def log_request_info():
#     print(f"Headers: {request.headers}")
#     print(f"Body: {request.get_data()}")

# Initialize OpenAI and Anthropic clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
DEEPL_API_KEY = os.environ.get('DEEPL_API_KEY')
translator = deepl.Translator(DEEPL_API_KEY)

# Configuration de la base de données SQLite
DATABASE = '/litefs/copilot-db.db'

def get_db():
    return sqlite3.connect(DATABASE)

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
            "description": "Analyser et résumer des documents financiers",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Le texte à analyser"
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
            "description": "Analyser le sentiment des actualités financières",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "L'entreprise pour laquelle analyser le sentiment"
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
            "description": "Effectuer une modélisation financière pour une action donnée",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Le symbole boursier de l'action"
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
            "description": "Optimiser un portefeuille d'actions",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste des symboles boursiers des actions"
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
            "description": "Analyser les risques d'un portefeuille",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste des symboles boursiers des actions"
                    },
                    "portfolio_value": {
                        "type": "number",
                        "description": "Valeur totale du portefeuille"
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
            "description": "Générer un rapport financier",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Données financières à inclure dans le rapport"
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
            "description": "Vérifier la conformité du portefeuille avec les réglementations",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_data": {
                        "type": "object",
                        "description": "Données du portefeuille à vérifier pour la conformité"
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
            "description": "Analyser le sentiment du marché pour une action donnée",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Le symbole boursier de l'action"
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
            "description": "Obtenir des recommandations d'investissement basées sur un portefeuille et un profil de risque",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Liste des symboles boursiers dans le portefeuille"
                    },
                    "risk_profile": {
                        "type": "string",
                        "description": "Le profil de risque de l'investisseur (ex: conservateur, modéré, agressif)"
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
            "description": "Analyser les données de trading du jour précédent pour une action donnée",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Le symbole boursier de l'action"
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
            "description": "Analyser le profil d'investissement de l'utilisateur basé sur ses interactions et l'historique de son portefeuille",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_interactions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Liste des interactions récentes de l'utilisateur"
                    },
                    "portfolio_history": {
                        "type": "object",
                        "description": "Valeurs historiques du portefeuille"
                    }
                },
                "required": ["user_interactions", "portfolio_history"]
            }
        }
    }
]

assistant = openai_client.beta.assistants.create(
    name="Finance Copilot",
    instructions="Vous êtes un assistant financier. Utilisez les fonctions fournies pour analyser des documents, évaluer le sentiment, effectuer des modélisations financières, optimiser des portefeuilles, gérer les risques, générer des rapports et vérifier la conformité.",
    model="gpt-4o-2024-08-06",
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
    print(f"Requête d'inscription reçue : {request.json}")
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"erreur": "Le nom d'utilisateur et le mot de passe sont requis"}), 400
    
    hashed_password = hash_password(password)
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                         (username, hashed_password))
        return jsonify({"message": "Utilisateur enregistré avec succès"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"erreur": "Le nom d'utilisateur existe déjà"}), 400

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
        return jsonify({"erreur": "Nom d'utilisateur ou mot de passe invalide"}), 401
    
    # Fonction pour enregistrer le chat
def save_chat_message(user_id, role, content):
    try:
        print(f"Tentative de sauvegarde du message pour l'utilisateur {user_id}")
        print(f"Role : {role}")
        print(f"Contenu : {content}")
        
        if content is None:
            content = ""
        elif hasattr(content, 'content'):
            content = content.content
        elif isinstance(content, (dict, list)):
            content = json.dumps(content)
        else:
            content = str(content)
        
        if not content.strip():
            print("Le contenu du message est vide, abandon de la sauvegarde")
            return
        
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (user_id, role, content, timestamp)
                VALUES (?, ?, ?, datetime('now'))
            """, (user_id, role, content))
            conn.commit()
        print("Message sauvegardé avec succès")
    except Exception as e:
        app.logger.error(f"Erreur lors de la sauvegarde du message: {str(e)}", exc_info=True)
        raise

# Fonction pour récupérer l'historique des chats
def get_chat_history(user_id):
    with sqlite3.connect(DATABASE) as conn:
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
            return {"erreur": f"Symbole d'action invalide: {stock}"}
    return portfolio if portfolio else {"erreur": "Aucune information de portefeuille valide trouvée"}

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
                    "erreur": "missing_data",
                    "message": "Informations sur le portefeuille manquantes. Veuillez fournir les détails du portefeuille."
                }
        elif function_name == "analyze_documents":
            return document_agent.analyser(arguments.get("text", ""))
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
                arguments.get("user_id"),
                arguments.get("portfolio"),
                arguments.get("chat_history")
            )
        else:
            return {"erreur": f"Fonction {function_name} introuvable"}
    except KeyError as e:
        return {
            "erreur": "missing_data",
            "message": f"Données manquantes : {str(e)}. Pouvez-vous fournir plus d'informations ?"
        }
    except Exception as e:
        return {
            "erreur": "execution_error",
            "message": f"Une erreur s'est produite lors de l'exécution de {function_name}: {str(e)}"
        }

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         username TEXT UNIQUE NOT NULL,
                         password TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS portfolios
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         user_id INTEGER,
                         name TEXT NOT NULL,
                         data TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS portfolio (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        name TEXT,
                        symbol TEXT,
                        weight REAL,
                        entry_price REAL,
                        FOREIGN KEY (user_id) REFERENCES users(id))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS user_settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        setting_name TEXT NOT NULL,
                        setting_value TEXT,
                        UNIQUE(user_id, setting_name))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        result TEXT
                    )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS chat_history
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         user_id INTEGER NOT NULL,
                         role TEXT NOT NULL,
                         content TEXT NOT NULL,
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

@app.route('/clear_chat', methods=['POST'])
@jwt_required()
def clear_chat():
    user_id = get_jwt_identity()
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        return jsonify({"message": "Historique de chat effacé avec succès"}), 200
    except Exception as e:
        app.logger.error(f"Erreur lors de l'effacement de l'historique de chat: {str(e)}")
        return jsonify({"erreur": "Échec de l'effacement de l'historique de chat"}), 500

def get_user_setting(user_id, setting_name, default_value=None):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT setting_value FROM user_settings WHERE user_id = ? AND setting_name = ?", (user_id, setting_name))
        result = cursor.fetchone()
        if result:
            try:
                return json.loads(result[0])
            except json.JSONDecodeError:
                return result[0]
    return default_value

def set_user_setting(user_id, setting_name, setting_value):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        if isinstance(setting_value, (list, dict)):
            setting_value = json.dumps(setting_value)
        cursor.execute("""
            INSERT OR REPLACE INTO user_settings (user_id, setting_name, setting_value)
            VALUES (?, ?, ?)
        """, (user_id, setting_name, setting_value))
        conn.commit()
    print(f"Paramètre '{setting_name}' sauvegardé pour l'utilisateur {user_id}: {setting_value}")


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

class Agents:
    @staticmethod
    def analyze_documents(data):
        text = data.get('text', '')
        return document_agent.analyser(text)

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
    client = openai_client
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Vous êtes un assistant chargé de structurer des données financières."},
            {"role": "user", "content": f"Veuillez structurer les données suivantes en JSON : {data}"}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def generate_verbose_response(result, function_name):
    client = openai_client
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
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


@app.route('/translate_news', methods=['POST'])
@jwt_required()
def translate_news():
    data = request.json
    news = data.get('news', [])
    translated_news = []

    print(f"Données reçues pour la traduction : {news}")

    try:
        for item in news:
            if isinstance(item, dict):
                title = item.get('title', '')
                description = item.get('description') or item.get('content', '')
                content = item.get('content', '')
            elif isinstance(item, str):
                title = item
                description = ""
            else:
                app.logger.warning(f"Format d'élément de nouvelles non reconnu : {item}")
                continue

            # Traduire le titre
            translated_title = translator.translate_text(
                title,
                target_lang="FR",
                formality="less"
            )

            # Traduire ou générer la description
            if description:
                translated_description = translator.translate_text(
                    description,
                    target_lang="FR",
                    formality="less"
                )
            else:
                # Si pas de description, on utilise Llama pour générer un résumé
                llama_response = anthropic_client.completions.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens_to_sample=200,
                    prompt=f"Générez un bref résumé en français de 200 caractères maximum pour cet article dont voici le titre : {content}"
                )
                generated_description = llama_response.completion.strip()
                
                # Traduire le résumé généré si nécessaire
                if not is_french(generated_description):
                    translated_description = translator.translate_text(
                        generated_description,
                        target_lang="FR",
                        formality="less"
                    )
                else:
                    translated_description = generated_description

            translated_news.append({
                'title': translated_title.text,
                'description': translated_description.text if isinstance(translated_description, deepl.TextResult) else translated_description
            })

        print(f"Nouvelles traduites : {translated_news}")
        return jsonify(translated_news)
    except Exception as e:
        app.logger.error(f"Erreur lors de la traduction des nouvelles : {str(e)}", exc_info=True)
        return jsonify({"erreur": "Erreur lors de la traduction des nouvelles"}), 500

def is_french(text):
    # Une fonction simple pour vérifier si le texte est déjà en français
    # Cette implémentation est basique et pourrait être améliorée
    french_words = set(['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'donc'])
    words = text.lower().split()
    return any(word in french_words for word in words)

@app.route('/submit_task', methods=['POST'])
def submit_task():
    data = request.json
    task_type = data.get('task_type')
    if task_type not in dir(Agents):
        return jsonify({"erreur": "Type de tâche invalide"}), 400

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
        return jsonify({"erreur": "Tâche non trouvée"}), 404

@app.route('/portfolio_analysis', methods=['POST'])
def portfolio_analysis():
    data = request.json
    task_sequence = ['analyze_documents', 'analyze_sentiment', 'model_financials', 
                     'optimize_portfolio', 'manage_risks', 'check_compliance', 'generate_report']

    results = {}
    for task in task_sequence:
        results[task] = getattr(Agents, task)(data)

    return jsonify(results)

@app.route('/chat', methods=['POST'])
@jwt_required()
def chat():
    global financial_data
    try:
        user_id = get_jwt_identity()
        data = request.json
        user_message = data.get('message')
        conversation_id = data.get('conversation_id')
        user_portfolio = data.get('portfolio')
        user_risk_profile = data.get('risk_profile')
        use_claude = request.json.get('use_claude', False)
        
        print(f"Début de la fonction chat pour l'utilisateur {user_id}")
        print(f"Message de l'utilisateur : {user_message}")
        print(f"Conversation ID : {conversation_id}")
        print(f"Utilisation de Claude : {use_claude}")
        
        save_chat_message(user_id, 'user', user_message)
        print("Message de l'utilisateur sauvegardé")

        if not conversation_id:
            conversation_id = conversation_manager.start_conversation()
            financial_data = {}
            print(f"Nouvelle conversation créée avec ID : {conversation_id}")

        messages = conversation_manager.get_messages(conversation_id)
        messages.append({"role": "user", "content": user_message})
        print(f"Messages de la conversation : {messages}")

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
        model = "claude-3-5-sonnet-20240620" if use_claude else "gpt-4o-2024-08-06"
        tools = claude_tools if use_claude else functions

        print(f"Modèle utilisé : {model}")
        print(f"Outils disponibles : {tools}")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=1024 if use_claude else None
        )
        print(f"Réponse brute du modèle : {response}")

        if use_claude:
            assistant_message = response.content[0]
            if assistant_message.type == 'text':
                message_content = assistant_message.text
                save_chat_message(user_id, 'assistant', message_content)
                return jsonify({"reply": message_content, "conversation_id": conversation_id})
        else:
            assistant_message = response.choices[0].message
            message_content = assistant_message.content

        if not getattr(assistant_message, 'tool_calls', None):
            conversation_manager.add_message(conversation_id, {"role": "assistant", "content": message_content})
            save_chat_message(user_id, 'assistant', message_content)
            print("Pas d'appel d'outil, retour de la réponse")
            return jsonify({"reply": message_content, "conversation_id": conversation_id})

        print("Traitement des appels d'outils")
        tool_messages = []
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"Appel de la fonction : {function_name}")
            print(f"Arguments de la fonction : {function_args}")
            
            function_info = get_function_info(function_name)
            structured_args = structure_data(function_args)
            if function_name in ['optimize_portfolio', 'manage_risks', 'check_compliance', 'get_investment_recommendation']:
                if user_portfolio:
                    function_args['portfolio'] = user_portfolio
                if user_risk_profile:
                    function_args['risk_profile'] = user_risk_profile
                else:
                    return jsonify({"reply": "Désolé, je n'ai pas pu accéder à votre portefeuille. Veuillez vérifier que vous en avez bien un.", "conversation_id": conversation_id})

            function_response = execute_function(function_name, function_args, user_message)
            
            print(f"Réponse de la fonction : {function_response}")
            
            if isinstance(function_response, dict) and function_response.get("erreur") == "missing_data":
                conversation_manager.add_message(conversation_id, {"role": "assistant", "content": function_response["message"]})
                print("Données manquantes, retour de la réponse d'erreur")
                return jsonify({"reply": function_response["message"], "conversation_id": conversation_id})
            
            verbose_response = generate_verbose_response(function_response, function_name)
            verbose_response += f"\n\nInformation importante : {function_info['description']}"

            tool_messages.append({
                "role": "tool",
                "content": verbose_response,
                "tool_call_id": tool_call.id
            })
            print(f"Message de l'outil ajouté : {tool_messages[-1]}")

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                } for tc in assistant_message.tool_calls
            ]
        })
        messages.extend(tool_messages)

        print("Génération de la réponse finale")
        final_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        final_message = final_response.choices[0].message
        conversation_manager.add_message(conversation_id, {"role": "assistant", "content": final_message.content})
        print(f"Message final : {final_message}")
        
        save_chat_message(user_id, 'assistant', final_message.content)
        print("Message final sauvegardé dans la base de données")
        
        return jsonify({"reply": final_message.content, "conversation_id": conversation_id})
    except Exception as e:
        app.logger.error(f"Erreur dans la route /chat: {str(e)}", exc_info=True)
        return jsonify({'erreur': 'Une erreur est survenue lors du traitement de votre demande'}), 500

@app.route('/agent/<agent_name>', methods=['POST'])
@jwt_required()
def call_agent(agent_name):
    data = request.json
    user_id = get_jwt_identity()
    portfolio = get_portfolio(user_id)
    try:
        if agent_name == "document":
            result = document_agent.analyser(data['text'])
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
            result = user_profile_agent.analyze_user_profile(user_id, portfolio, chat_history)
        else:
            return jsonify({"erreur": "Agent non trouvé"}), 404
        
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
        return jsonify({"erreur": f"Missing required data: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"Error in {agent_name} agent: {str(e)}")
        return jsonify({"erreur": str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"erreur": "Aucun fichier n'a été fourni"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"erreur": "Aucun fichier sélectionné"}), 400
    if file and file.filename.endswith('.pdf'):
        try:
            pdf_reader = PdfReader(io.BytesIO(file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Analyse du texte extrait avec l'agent d'analyse de documents
            result = analyze_financial_report(text)
            return jsonify(result)
        except Exception as e:
            app.logger.error(f"Erreur lors de l'analyse du PDF : {str(e)}")
            return jsonify({"erreur": "Erreur lors de l'analyse du PDF"}), 500
    else:
        return jsonify({"erreur": "Type de fichier invalide"}), 400

def analyze_financial_report(text):
    try:
        client, model = anthropic_client, "claude-3-5-sonnet-20240620"
        
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            system="Vous êtes un analyste financier expert. Extrayez les informations financières clés du texte donné et présentez-les de manière structurée.",
            messages=[
                {"role": "user", "content": f"Analysez ce rapport financier et extrayez les informations suivantes : chiffre d'affaires, bénéfice net, EBITDA et risques clés. Présentez les résultats au format JSON structuré comme suit : {{'chiffre_affaires': nombre, 'benefice_net': nombre, 'ebitda': nombre, 'risques': [liste de chaînes]}}. Si une information n'est pas disponible, utilisez null.\n\nRapport:\n{text}"}
            ]
        )
        
        response_text = response.content[0].text
        print(f"Réponse brute de l'API : {response_text}")
        
        # Essayez d'abord de parser le JSON directement
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Si le parsing JSON échoue, essayez d'extraire les informations manuellement
            result = extract_financial_info(response_text)
        
        print(f"Résultat de l'analyse : {result}")
        return result

    except json.JSONDecodeError as e:
        app.logger.error(f"Erreur lors de l'analyse JSON : {str(e)}")
        return {"erreur": "Erreur lors de l'analyse des données financières"}
    except Exception as e:
        app.logger.error(f"Erreur inattendue dans analyze_financial_report : {str(e)}")
        return {"erreur": "Une erreur inattendue s'est produite lors de l'analyse du rapport financier"}

def extract_financial_info(text):
    result = {
        "chiffre_affaires": None,
        "benefice_net": None,
        "ebitda": None,
        "risques": []
    }
    
    # Extraction du chiffre d'affaires
    ca_match = re.search(r"chiffre d'affaires.*?(\d+(?:,\d+)?(?:\.\d+)?)", text, re.IGNORECASE)
    if ca_match:
        result["chiffre_affaires"] = float(ca_match.group(1).replace(',', ''))
    
    # Extraction du bénéfice net
    bn_match = re.search(r"bénéfice net.*?(\d+(?:,\d+)?(?:\.\d+)?)", text, re.IGNORECASE)
    if bn_match:
        result["benefice_net"] = float(bn_match.group(1).replace(',', ''))
    
    # Extraction de l'EBITDA
    ebitda_match = re.search(r"ebitda.*?(\d+(?:,\d+)?(?:\.\d+)?)", text, re.IGNORECASE)
    if ebitda_match:
        result["ebitda"] = float(ebitda_match.group(1).replace(',', ''))
    
    # Extraction des risques
    risques_match = re.search(r"risques.*?:(.*?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if risques_match:
        result["risques"] = [risk.strip() for risk in risques_match.group(1).split(',') if risk.strip()]
    
    return result

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
        return jsonify({"erreur": "Aucun fichier n'a été fourni"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"erreur": "Aucun fichier sélectionné"}), 400
    if file and file.filename.endswith('.pdf'):
        pdf_reader = PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Analyse du texte extrait avec l'agent d'analyse de documents
        result = document_agent.analyser_rapport_financier(text)
        
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
        return jsonify({"erreur": "Type de fichier invalide"}), 400

@app.route('/settings', methods=['GET', 'POST'])
@jwt_required()
def settings():
    user_id = get_jwt_identity()
    app.logger.info(f"Requête reçue pour l'utilisateur {user_id}")
    app.logger.info(f"Méthode de la requête : {request.method}")
    app.logger.info(f"En-têtes de la requête : {request.headers}")
    
    if request.method == 'GET':
        settings = {
            'name': get_user_setting(user_id, 'name', ''),
            'defaultPortfolioValue': get_user_setting(user_id, 'default_portfolio_value', 100000),
            'riskProfile': get_user_setting(user_id, 'risk_profile', 'moderate'),
            'preferredSectors': get_user_setting(user_id, 'preferred_sectors', []),
            'theme': get_user_setting(user_id, 'theme', 'light')
        }
        app.logger.info(f"Paramètres récupérés : {settings}")
        return jsonify(settings), 200

    elif request.method == 'POST':
        if not request.is_json:
            app.logger.error("Requête POST reçue sans Content-Type application/json")
            return jsonify({"erreur": "Content-Type must be application/json"}), 415
        
        data = request.get_json()
        app.logger.info(f"Données reçues : {data}")
        
        if not data:
            return jsonify({"erreur": "No JSON data provided"}), 400
        
        set_user_setting(user_id, 'name', data.get('name', ''))
        set_user_setting(user_id, 'default_portfolio_value', data.get('defaultPortfolioValue', 100000))
        set_user_setting(user_id, 'risk_profile', data.get('riskProfile', 'moderate'))
        set_user_setting(user_id, 'preferred_sectors', data.get('preferredSectors', []))
        set_user_setting(user_id, 'theme', data.get('theme', 'light'))
        
        return jsonify({"message": "Settings saved successfully"}), 200

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
            return jsonify({"erreur": "Erreur lors de la sauvegarde du portfolio"}), 500
    elif request.method == 'GET':
        try:
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM portfolio WHERE user_id = ?", (current_user,))
                portfolio = cursor.fetchall()
                if not portfolio:
                    return jsonify({"name": "défaut", "stocks": []})
                return jsonify({"name": "défaut", "stocks": [{"symbol": row[3], "weight": row[4], "entry_price": row[5]} for row in portfolio]})
        except Exception as e:
            return jsonify({"erreur": str(e)}), 500

@app.route('/latest_price', methods=['GET'])
def latest_price():
    symbol = request.args.get('symbol')
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            latest_price = data['Close'].iloc[-1]
            return jsonify({"symbol": symbol, "prix": float(latest_price)})
        else:
            return jsonify({"erreur": "Aucune donnée disponible"}), 404
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération du dernier prix pour {symbol}: {str(e)}")
        return jsonify({"erreur": str(e)}), 500

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
        app.logger.error(f"Erreur lors de la récupération des prix historiques pour {symbol}: {str(e)}")
        return jsonify({"erreur": str(e)}), 500

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
        app.logger.error(f"Erreur lors de la récupération des nouvelles : {str(e)}")
        return jsonify({"erreur": "Service de nouvelles temporairement indisponible"}), 503

@app.route('/backtest', methods=['POST'])
@jwt_required()
def backtest():
    data = request.json
    portfolio = data.get('portfolio', {})
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    if not portfolio or 'stocks' not in portfolio or not portfolio['stocks']:
        return jsonify({"erreur": "Le portfolio est vide ou invalide"}), 400

    # Télécharger les données historiques
    stocks_data = {}
    weights = {}
    for stock in portfolio['stocks']:
        symbol = stock.get('symbol')
        weight = float(stock.get('weight', 0)) / 100  # Convertir en décimal
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        if hist.empty:
            return jsonify({"erreur": f"Aucune donnée historique disponible pour {symbol}"}), 400
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
        except sqlite3.Error as e:
            app.logger.error(f"Erreur de base de données lors de la récupération de l'historique de chat: {str(e)}")
            return jsonify({"erreur": "Échec de la récupération de l'historique de chat"}), 500
        except Exception as e:
            app.logger.error(f"Erreur inattendue lors de la récupération de l'historique de chat: {str(e)}")
            return jsonify({"erreur": "Une erreur inattendue s'est produite"}), 500

    elif request.method == 'POST':
        data = request.json
        if not data or not isinstance(data, dict) or 'role' not in data or 'content' not in data:
            return jsonify({"erreur": "Format de données invalide. Attendu 'role' et 'content'"}), 400
        
        try:
            print(data)
            save_chat_message(user_id, data['role'], data['content'])
            return jsonify({"message": "Message de chat enregistré avec succès"}), 200
        except sqlite3.Error as e:
            app.logger.error(f"Erreur de base de données lors de l'enregistrement du message de chat: {str(e)}")
            return jsonify({"erreur": "Échec de l'enregistrement du message de chat"}), 500
        except Exception as e:
            app.logger.error(f"Erreur inattendue lors de l'enregistrement du message de chat: {str(e)}")
            return jsonify({"erreur": "Une erreur inattendue s'est produite"}), 500

@app.route('/compare_portfolios', methods=['POST'])
@jwt_required()
def compare_portfolios():
    data = request.json
    print("Données reçues dans compare_portfolios:", data)
    
    portfolio = data['portfolio']
    benchmark = data['benchmark']
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    print("Type de portfolio:", type(portfolio))
    print("Contenu du portfolio:", portfolio)

    # Vérifier si le portfolio est vide
    if not portfolio:
        return jsonify({"erreur": "Le portfolio est vide"}), 400

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
        return jsonify({"erreur": "Structure de portfolio invalide"}), 400

    for stock in stocks:
        try:
            symbol = stock['symbol']
            weight = float(stock['weight']) / 100  # Convertir le poids en décimal
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            portfolio_data[symbol] = hist['Close']
            portfolio_weights[symbol] = weight
        except Exception as e:
            print(f"Erreur lors du traitement de l'action {stock}: {str(e)}")
            return jsonify({"erreur": f"Erreur lors du traitement de l'action {stock}: {str(e)}"}), 400

    # Vérifier si des données ont été récupérées
    if not portfolio_data:
        return jsonify({"erreur": "Aucune donnée n'a pu être récupérée pour le portfolio donné"}), 400

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
        return jsonify({"erreur": "Scénario invalide"}), 400
    
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
        return jsonify({"erreur": "La valeur du portfolio est requise"}), 400
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("INSERT OR REPLACE INTO user_settings (user_id, setting_name, setting_value) VALUES (?, 'portfolio_value', ?)",
                         (user_id, str(new_value)))
        return jsonify({"message": "Valeur du portfolio mise à jour avec succès"}), 200
    except Exception as e:
        return jsonify({"erreur": str(e)}), 500

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
        return jsonify({"erreur": str(e)}), 500
    
def update_db_structure():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            ALTER TABLE user_settings
            ADD COLUMN name TEXT
        ''')
        conn.commit()

if __name__ == '__main__':
    init_db()
    update_db_structure()
    app.run(debug=True)
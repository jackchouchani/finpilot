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

# Initialize OpenAI and Anthropic clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Configuration de la base de données SQLite
DATABASE = 'copilote.db'

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
def save_chat_history(user_id, message):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_history (user_id, message, timestamp)
            VALUES (?, ?, datetime('now'))
        """, (user_id, json.dumps(message)))
        conn.commit()

# Fonction pour récupérer l'historique des chats
def get_chat_history(user_id, limit=50):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT message FROM chat_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))
        messages = cursor.fetchall()
    return [json.loads(message[0]) for message in messages]

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\n', ' ')
    text = ''.join(char for char in text if ord(char) > 31 or char == ' ')
    return text

def get_chat_history(user_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT message FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50", (user_id,))
        messages = cursor.fetchall()
    return [json.loads(message[0]) for message in messages]

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
                        user_id INTEGER PRIMARY KEY,
                        setting_name TEXT,
                        setting_value TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        result TEXT
                    )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS chat_history
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         user_id INTEGER,
                         message TEXT,
                         timestamp DATETIME,
                         FOREIGN KEY (user_id) REFERENCES users(id))''')

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
    client = openai.OpenAI()
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
def chat():
    global financial_data
    
    user_id = get_jwt_identity()
    user_message = request.json['message']
    conversation_id = request.json.get('conversation_id')
    use_claude = request.json.get('use_claude', False)
    
    # Sauvegardez le message de l'utilisateur
    save_chat_history(user_id, {"role": "user", "content": user_message})

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

    client = anthropic.Anthropic() if use_claude else openai_client
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
            print(f"Claude response type: {response.content[0].type}")
            if response.content[0].type == 'text':
                conversation_manager.add_message(conversation_id, {"role": "assistant", "content": response.content[0].text})
                return jsonify({"reply": response.content[0].text, "conversation_id": conversation_id})
            assistant_message = response.content[0]
            # Sauvegardez la réponse de l'assistant
            save_chat_history(user_id, {"role": "assistant", "content": assistant_message.content})

        else:
            assistant_message = response.choices[0].message
            # Sauvegardez la réponse de l'assistant
            save_chat_history(user_id, {"role": "assistant", "content": assistant_message.content})

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
        elif agent_name == "compliance":
            result = compliance_agent.check_compliance(portfolio)
        elif agent_name == "market_sentiment":
            print(data)
            if 'summary' not in data:
                data['summary'] = "No summary provided"  # Utiliser une valeur par défaut si 'summary' est manquant
            result = market_sentiment_agent.analyze_sentiment(data['ticker'], data['summary'])
        elif agent_name == "investment_recommendation":
            portfolio = data.get('portfolio', [])
            if isinstance(portfolio, dict):
                portfolio = [portfolio]
            print(portfolio)
            portfolio = [stock['symbol'].upper() if isinstance(stock, dict) and 'symbol' in stock else stock.upper() for stock in portfolio]
            risk_profile = data.get('risk_profile', 'moderate')  # Utiliser 'moderate' par défaut
            result = investment_recommendation_agent.get_recommendation(portfolio, risk_profile)
        elif agent_name == "historical_data_analysis":
            result = historical_data_agent.analyze_previous_day(data['ticker'])
        elif agent_name == "user_profile_analysis":
            chat_history = get_chat_history(user_id)
            portfolio = get_portfolio(user_id)
            result = user_profile_agent.analyze_user_profile(portfolio, chat_history)
        else:
            return jsonify({"error": "Agent not found"}), 404
        
        return jsonify(result)
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
    print("Received data in backtest:", data)
    
    portfolio = data.get('portfolio', {})
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    print("Type of portfolio:", type(portfolio))
    print("Content of portfolio:", portfolio)

    # Vérifier si le portfolio est vide ou s'il n'y a pas de stocks
    if not portfolio or 'stocks' not in portfolio or not portfolio['stocks']:
        return jsonify({"error": "Portfolio is empty or invalid"}), 400

    # Vérifier et ajuster les dates
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Télécharger les données historiques
    stocks_data = {}
    weights = {}
    for stock in portfolio['stocks']:
        print(f"Processing stock:")
        print("Type:", type(stock))
        print("Content:", stock)
        
        try:
            symbol = stock.get('symbol')
            weight = stock.get('weight')
            if not symbol or weight is None:
                return jsonify({"error": f"Invalid stock data: {stock}"}), 400
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if hist.empty:
                return jsonify({"error": f"No historical data available for {symbol}"}), 400
            
            stocks_data[symbol] = hist['Close']
            weights[symbol] = float(weight)
        except Exception as e:
            print(f"Error processing stock {stock}: {str(e)}")
            return jsonify({"error": f"Error processing stock {stock}: {str(e)}"}), 400

    # Créer un DataFrame avec les prix de clôture de tous les stocks
    df = pd.DataFrame(stocks_data)

    # Vérifier si nous avons des données
    if df.empty:
        return jsonify({"error": "No data available for the given date range"}), 400

    # Calculer les rendements journaliers
    returns = df.pct_change().dropna()

    # Vérifier si nous avons des rendements à analyser
    if returns.empty:
        return jsonify({"error": "Insufficient data for analysis"}), 400

    # Initialiser le portefeuille
    initial_value = 10000  # Valeur initiale du portefeuille
    portfolio_value = initial_value

    # Simuler la performance du portefeuille
    portfolio_returns = []
    for date, row in returns.iterrows():
        daily_return = sum(row * pd.Series(weights))
        portfolio_value *= (1 + daily_return)
        portfolio_returns.append(float(portfolio_value))

    # Calculer les métriques de performance
    total_return = (portfolio_value - initial_value) / initial_value
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    portfolio_volatility = returns.mul(pd.Series(weights)).sum(axis=1).std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - 0.02) / portfolio_volatility if portfolio_volatility != 0 else 0

    results = {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "volatility": float(portfolio_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "portfolio_values": portfolio_returns
    }

    print("Backtest results:", results)
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
            save_chat_history(user_id, data)
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
    start_date = data['start_date']
    end_date = data['end_date']

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
        print(f"Processing stock:")
        print("Type:", type(stock))
        print("Content:", stock)
        
        try:
            symbol = stock['symbol']
            weight = float(stock['weight'])
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

    # Calculer les rendements
    portfolio_return = annualized_return

    # Télécharger les données historiques du benchmark
    benchmark_ticker = yf.Ticker(benchmark)
    benchmark_data = benchmark_ticker.history(start=start_date, end=end_date)
    benchmark_returns = benchmark_data['Close'].pct_change().dropna()

    # Assurez-vous que les deux séries de rendements ont la même longueur
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]

    # Calculer la performance cumulée
    portfolio_cumulative = (1 + (portfolio_returns * pd.Series(portfolio_weights)).sum(axis=1)).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()

    # Calculer les métriques de performance
    portfolio_total_return = portfolio_cumulative.iloc[-1] - 1 if len(portfolio_cumulative) > 0 else 0
    benchmark_total_return = benchmark_cumulative.iloc[-1] - 1 if len(benchmark_cumulative) > 0 else 0

    portfolio_volatility = portfolio_returns.std().mean() * np.sqrt(252) if len(portfolio_returns) > 0 else 0
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252) if len(benchmark_returns) > 0 else 0

    portfolio_sharpe = ((portfolio_total_return - risk_free_rate) / portfolio_volatility) if portfolio_volatility != 0 else 0
    benchmark_sharpe = ((benchmark_total_return - risk_free_rate) / benchmark_volatility) if benchmark_volatility != 0 else 0

    results = {
        "portfolio_return": float(portfolio_total_return),
        "benchmark_return": float(benchmark_total_return),
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
    
    # Définir les paramètres des scénarios
    scenarios = {
        "market_crash": {"mean": -0.3, "volatility": 0.4},
        "bull_market": {"mean": 0.2, "volatility": 0.15},
        "high_inflation": {"mean": 0.05, "volatility": 0.25},
    }
    
    if scenario not in scenarios:
        return jsonify({"error": "Invalid scenario"}), 400
    
    scenario_params = scenarios[scenario]
    
    # Simuler les rendements pour chaque action du portefeuille
    simulated_returns = {}
    for stock in portfolio['stocks']:
        returns = np.random.normal(scenario_params['mean'], scenario_params['volatility'], 252)  # 252 jours de trading
        simulated_returns[stock['symbol']] = returns
    
    # Calculer la performance du portefeuille
    portfolio_weights = {stock['symbol']: float(stock['weight']) for stock in portfolio['stocks']}
    portfolio_returns = sum(simulated_returns[symbol] * weight for symbol, weight in portfolio_weights.items())
    
    # Calculer la valeur finale du portefeuille
    initial_value = 10000  # Valeur initiale supposée
    final_value = initial_value * (1 + portfolio_returns.sum())
    
    results = {
        "scenario": scenario,
        "initial_value": initial_value,
        "final_value": final_value,
        "total_return": (final_value - initial_value) / initial_value,
        "daily_returns": portfolio_returns.tolist()
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
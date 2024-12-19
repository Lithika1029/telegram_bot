import os
import telebot
import pandas as pd
import joblib
import psutil
import validators

# Load the Telegram Bot Token
BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable not set.")
bot = telebot.TeleBot(BOT_TOKEN)

# Load the trained model and vectorizer
try:
    model = joblib.load('phishing_detection_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Model or vectorizer file not found. Ensure both 'phishing_detection_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")

# Function to validate and process URLs
def process_url(url):
    if not validators.url(url):
        raise ValueError("Invalid URL. Please send a valid URL starting with http:// or https://")
    url_tfidf = vectorizer.transform([url])
    return url_tfidf

# Command to handle URL detection
@bot.message_handler(func=lambda message: message.text.startswith("http"))
def detect_phishing_url(message):
    try:
        url_vectorized = process_url(message.text)
        prediction = model.predict(url_vectorized)
        if prediction[0] == 1:
            response = f"🚨 Warning: The URL appears to be a phishing link!\n{message.text}"
        else:
            response = f"✅ The URL seems safe.\n{message.text}"
    except ValueError as ve:
        response = f"Error: {str(ve)}"
    except Exception as e:
        response = f"Error: Unable to process the URL. Details: {str(e)}"
    bot.reply_to(message, response)

# Welcome command
@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Welcome to the Phishing URL Detection Bot! Send me a URL, and I will check if it's phishing or safe.")

# System Info command
@bot.message_handler(func=lambda message: message.text.lower() == "system info")
def send_system_info(message):
    memory = psutil.virtual_memory()
    total_memory = memory.total // (1024 * 1024)
    used_memory = memory.used // (1024 * 1024)
    cpu_usage = psutil.cpu_percent(interval=1)
    response = (
        f"System Info:\n"
        f"Total RAM: {total_memory} MB\n"
        f"Used RAM: {used_memory} MB\n"
        f"CPU Usage: {cpu_usage}%"
    )
    bot.reply_to(message, response)

# Start the bot
print("Bot is running...")
bot.infinity_polling()

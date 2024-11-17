from flask import Flask, render_template, request, jsonify
import random
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# Pre-load NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize Flask app
app = Flask(__name__)

# Global variables for the chatbot
lemmer = WordNetLemmatizer()
sent_tokens = []  # Dynamic corpus

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Chatbot response generation
def response(user_response):
    global sent_tokens
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.remove(user_response)
    return robo_response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/set_base_text", methods=["POST"])
def set_base_text():
    global sent_tokens
    base_text = request.json.get("base_text", "").lower()
    sent_tokens = nltk.sent_tokenize(base_text)  # Tokenize the base text
    return jsonify({"message": "Base text set successfully!"})

@app.route("/get_response", methods=["POST"])
def get_response():
    global sent_tokens
    user_message = request.json.get("message", "").lower()
    if not sent_tokens:
        return jsonify({"response": "Please provide the base text first!"})
    if user_message in ["bye", "exit"]:
        return jsonify({"response": "Bye! Take care."})
    if user_message in ["thanks", "thank you"]:
        return jsonify({"response": "You are welcome!"})
    if greeting(user_message) is not None:
        return jsonify({"response": greeting(user_message)})
    return jsonify({"response": response(user_message)})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import random
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Pre-load NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize Flask app
app = Flask(__name__)

# Global variables
lemmer = WordNetLemmatizer()
sent_tokens = []  # Dynamic corpus for responses

# Utility functions
def LemTokens(tokens):
    """Lemmatizes the given tokens."""
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    """Normalizes the input text by removing punctuation and lemmatizing."""
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """Checks if the input is a greeting and returns a suitable response."""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Chatbot response generation
def response(user_response):
    """Generates a response based on user input and the base text."""
    global sent_tokens
    robo_response = ''
    sent_tokens.append(user_response)  # Add user input to corpus
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
    sent_tokens.pop()  # Remove user input to avoid growing corpus indefinitely
    return robo_response

# Routes
@app.route("/")
def home():
    """Renders the home page."""
    return render_template("index.html")

@app.route("/set_base_text", methods=["POST"])
def set_base_text():
    """Sets the base text for the chatbot."""
    global sent_tokens
    base_text = request.json.get("base_text", "").lower()
    if not base_text.strip():
        return jsonify({"message": "Base text cannot be empty!"}), 400
    sent_tokens = nltk.sent_tokenize(base_text)  # Tokenize and store base text
    return jsonify({"message": "Base text set successfully!"})

@app.route("/get_response", methods=["POST"])
def get_response():
    """Handles user input and generates a response."""
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

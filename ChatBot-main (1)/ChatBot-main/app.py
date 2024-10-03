from flask import Flask, render_template, request, jsonify, session
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
from difflib import get_close_matches

# Load the tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Necessary for session management

# Load intents data
with open("intents.json") as file:
    intents_data = json.load(file)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = get_chat_response(msg)
    return jsonify(response=response)

def preprocess_text(text):
    """Preprocess the text for better matching by removing punctuation and lowercasing"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_chat_response(text):
    """Generate a response for the user's message"""
    # Preprocess user input
    text = preprocess_text(text)

    # Check if the text matches any known patterns from the intents data using fuzzy matching
    best_match = None
    highest_match_score = 0.0  # Track the highest match score for fuzzy matching

    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            pattern_clean = preprocess_text(pattern)

            # Compute fuzzy match score
            match_score = get_close_matches(text, [pattern_clean], n=1, cutoff=0.7)
            
            # If a match is found, track the best match and highest score
            if match_score:
                best_match = intent
                highest_match_score = 1.0  # Set match_score to 1 since `get_close_matches` gives a match
                break
        
        if best_match:
            break

    # If a match is found, return a response based on the best match
    if best_match and highest_match_score >= 0.7:  # Use a confidence threshold
        return best_match["responses"][0]

    # Fallback to DialoGPT for open-ended conversation only if no intent matches
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # Initialize or retrieve chat history from session
    if 'chat_history_ids' not in session:
        session['chat_history_ids'] = []  # Initialize as empty list if not in session

    # Convert session-stored chat history back to tensor
    chat_history_ids = torch.tensor(session['chat_history_ids'], dtype=torch.long) if session['chat_history_ids'] else torch.zeros((1, 0), dtype=torch.long)

    # Append new input to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    # Generate a response with controlled length
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Update session with the new chat history (convert tensor to list for session)
    session['chat_history_ids'] = chat_history_ids.tolist()

    # Decode and return the response from DialoGPT
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Fallback response if DialoGPT's output is not suitable
    if not response.strip() or "I’m not sure" in response:
        response = "I'm not sure how to respond to that. Can you please provide more details or ask something else?"

    return response

if __name__ == '__main__':
    app.run(debug=True)

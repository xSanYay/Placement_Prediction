import json
import os
import requests
from flask import Flask, request, jsonify, render_template
from google.api_core import retry
import datetime
import google.generativeai as genai

# Configure the Google Generative AI API key from environment variable.
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

app = Flask(__name__)

# Define the known domains in your dataset.
KNOWN_DOMAINS = [
    "Software", "IoT (Internet of Things)", "Signal Processing",
    "Embedded Systems", "VLSI Design", "Analog and Digital Electronics",
    "Communication Systems", "Power Electronics", "Semiconductor Industry"
]

# System prompt for the placement prediction agent.
PLACEMENT_BOT_PROMPT = f"""
You are a placement prediction chatbot. Your job is to interact with a candidate to gather their details:
- Ask for the candidate's CGPA.
- Ask for their skills (which you will map to one of these domains: {', '.join(KNOWN_DOMAINS)}).
- Ask for their expected salary.
Once you have the required details, call the prediction tool to get the most likely company for placement.
Your responses should guide the candidate to provide the necessary information in a friendly, conversational manner.
If the candidate gives skills (e.g., "python, java, html"), you must map them to one of the above domains (for example, map these skills to "Software").
When ready, call the tool function "get_placement_prediction" with the keys: cgpa (float), domain (one of the above domains), and expected_salary (float).
Keep the conversation engaging and short, dont drag and informative to help the candidate feel comfortable sharing their details and end with a motivational message.
Dont give formattingd and make sure not to mention terms like 'matching, etc ,etc' that you will do in backend or domains etc. Just keep the convo normal
Dont show them domain available, juts ask for skills and map them to one of the above domains, if you are unclear only then ask them which one they want from the ones you have doubt.
Give the output in this format : Give the company with the highest probability first, then after that list all the other possible companies.
"""

# Define a tool function that calls the placement prediction API endpoint.
def get_placement_prediction(cgpa: float, domain: str, expected_salary: float) -> dict:
    """
    Sends a POST request to the placement prediction API and returns the JSON response.
    """
    url = "http://localhost:4567/predict"  # Adjust the port/path if needed.
    payload = {
        "cgpa": cgpa,
        "domain": domain,
        "expected_salary": expected_salary
    }
    print(payload)
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
        else:
            data = {"error": f"API returned status code {response.status_code}"}
    except Exception as error:
        data = {"error": str(error)}
    return data

# Define the list of tool functions for the Gemini agent.
tools = [get_placement_prediction]

# Instantiate the Gemini generative model with the system prompt and tools.
model_name = "gemini-1.5-flash"
model = genai.GenerativeModel(model_name, tools=tools, system_instruction=PLACEMENT_BOT_PROMPT)
convo = model.start_chat(enable_automatic_function_calling=True)

# A retry wrapper for sending messages to Gemini.
@retry.Retry(initial=30)
def send_message_to_gemini(message: str):
    return convo.send_message(message)

# Flask endpoint to deliver an initial welcome message.
@app.route('/first_message', methods=['GET'])
def first_message():
    initial_response = "Welcome to the Placement Bot! Please provide your details (CGPA, skills, and expected salary)."
    return jsonify({"response": initial_response})

# Flask endpoint to accept user queries and return Gemini's response.
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get("query", "")
    try:
        gemini_response = send_message_to_gemini(user_query)
        bot_response = gemini_response.text
    except Exception as error:
        bot_response = str(error)
    return jsonify({"response": bot_response})

# Endpoint to render the chat interface.
@app.route('/')
def chatbot_view():
    return render_template('chat.html')

if __name__ == '__main__':
    # Run the Flask app on port 5678.
    app.run(host="0.0.0.0", port=5678, debug=False)
import logging
from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import re
import json

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

app = Flask(__name__)
CORS(app)

amadeus_api_key = os.getenv('AMADEUS_API_KEY')
amadeus_api_secret = os.getenv('AMADEUS_API_SECRET')

print("Current working directory:", os.getcwd()) 

tokenizer = AutoTokenizer.from_pretrained("fine_tuned_models")
model = AutoModelForCausalLM.from_pretrained("fine_tuned_models")

with open('travel_dataset.json', 'r') as f:
    travel_data = json.load(f)

city_to_iata = {
    "Jakarta": "CGK",
    "Tokyo": "NRT",
    "New York": "JFK",
    "Los Angeles": "LAX",
    "London": "LON",
    "Paris": "CDG",
    "Singapore": "SIN",
    "Sydney": "SYD",
    "Dubai": "DXB",
    "Bangkok": "BKK",
    "Hong Kong": "HKG",
    "Beijing": "PEK",
    "Seoul": "ICN",
    "Kuala Lumpur": "KUL",
    "Istanbul": "IST",
    "Berlin": "TXL",
    "Madrid": "MAD",
    "Rome": "FCO",
    "Mexico City": "MEX",
    "Amsterdam": "AMS",
    "Cairo": "CAI",
    "Moscow": "SVO",
    "Athens": "ATH",
    "Copenhagen": "CPH",
    "Zurich": "ZRH",
}

user_context = {}

def format_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        logging.debug(f"Invalid date format: {date_str}")
        return None

def extract_date(query):
    match = re.search(r"(on\s+)?(\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2}|\d{1,2}\s+\w+)", query, re.IGNORECASE)
    if match:
        date_str = match.group(0).replace("on ", "").strip()
        logging.debug(f"Extracted date: {date_str}")
        return format_date(date_str)
    logging.debug(f"Could not extract date from query: {query}")
    return None

def match_intent(query):
    for entry in travel_data:
        for input_pattern in entry["input"]:
            if re.search(input_pattern.lower(), query.lower()):
                logging.debug(f"Matched intent: {entry['intent']} for query: {query}")
                return entry
    logging.debug(f"No intent matched for query: {query}")
    return None

def search_cheapest_flight(origin, destination, departure_date):
    logging.debug(f"Searching for flights from {origin} to {destination} on {departure_date}")
    access_token = get_access_token()
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {access_token}"}

    origin_code = city_to_iata.get(origin)
    destination_code = city_to_iata.get(destination)

    if not all([origin_code, destination_code, departure_date]):
        logging.error("Missing origin, destination, or departure date")
        return {"error": "Origin, destination, and departure date must be provided."}

    params = {
        "originLocationCode": origin_code,
        "destinationLocationCode": destination_code,
        "departureDate": departure_date,
        "adults": 1,
        "currencyCode": "USD",
        "max": 5
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        logging.debug(f"Flight search response: {data}")

        if "data" in data and len(data["data"]) > 0:
            return data
        else:
            logging.debug("No flight data found for the given search criteria.")
            return {"error": "No flights found for the specified criteria."}
    except requests.exceptions.RequestException as e:
        logging.error(f"Error retrieving flight offers: {str(e)}")
        return {"error": "Could not retrieve flight offers from Amadeus API."}

def get_access_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": amadeus_api_key,
        "client_secret": amadeus_api_secret
    }

    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        access_token = response.json()["access_token"]
        logging.debug("Successfully obtained access token")
        return access_token
    except requests.exceptions.RequestException as e:
        logging.error(f"Error obtaining access token: {str(e)}")
        raise Exception("Could not obtain access token from Amadeus API.")

def extract_location(query):
    query = query.strip().lower()
    for city in city_to_iata.keys():
        if city.lower() in query:
            logging.debug(f"Extracted city: {city}")
            return city
    logging.debug(f"Could not extract city from query: {query}")
    return None

def extract_flight_details(query):
    normalized_query = query.lower()
    date = extract_date(normalized_query)

    origin = None
    destination = None

    if "from" in normalized_query and "to" in normalized_query:
        parts = normalized_query.split("from")
        if len(parts) > 1:
            origin_part = parts[1].strip()
            origin = extract_location(origin_part)
            destination_part = origin_part.split("to")[-1].strip()
            destination = extract_location(destination_part)

    return origin, destination, date

def gpt_simple_response(query):
    logging.debug(f"Generating simple response for query: {query}")
    inputs = tokenizer.encode(query, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = response.replace("{origin}", "your departure city") \
                       .replace("{destination}", "your destination city") \
                       .replace("{date}", "your travel date")
    logging.debug(f"Generated response: {response}")
    return response


def gpt_generate_response(query, user_id):
    global user_context

    if user_id not in user_context:
        user_context[user_id] = {"origin": None, "destination": None, "date": None}

    origin, destination, date = extract_flight_details(query)

    logging.debug(f"user_context before: {user_context}")
    logging.debug(f"Extracted: origin: {origin}, destination: {destination}, date: {date}")

    if origin:
        user_context[user_id]["origin"] = origin
    if destination:
        user_context[user_id]["destination"] = destination
    if date:
        user_context[user_id]["date"] = date

    logging.debug(f"user_context after: {user_context}")

    if all([user_context[user_id]["origin"], user_context[user_id]["destination"], user_context[user_id]["date"]]):
        flight_info = search_cheapest_flight(user_context[user_id]["origin"], user_context[user_id]["destination"], user_context[user_id]["date"])

        if "error" in flight_info:
            return {"message": flight_info["error"], "flight_info": []}

        flights = flight_info.get("data", [])
        if flights:
            return {
                "message": f"Great! I found {len(flights)} flights from {user_context[user_id]['origin']} to {user_context[user_id]['destination']} on {user_context[user_id]['date']}. Let's check them out!",
                "flight_info": flight_info
            }
        else:
            return {"message": "I'm sorry, but it looks like there are no flights available for that route.", "flight_info": []}

    return {
        "message": gpt_simple_response(query),
        "flight_info": []
    }


@app.route('/api/chatbot', methods=['POST'])
@cross_origin()
def chatbot_interface():
    try:
        data = request.json
        query = data.get('query')
        user_id = request.remote_addr
        logging.debug(f"Received query: {query} from user_id: {user_id}")
        response = gpt_generate_response(query, user_id)
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in chatbot interface: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

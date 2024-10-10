from flask import Flask, request, jsonify
import requests
import re
from flask_cors import CORS
from google.cloud import dialogflow_v2 as dialogflow
import logging
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./travelassistant-ugsp.json"

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app, origins=["http://localhost:4000"])

amadeus_api_key = os.getenv('AMADEUS_API_KEY', 'ynjYWaJMJPpyeq6DPZUTot4YbgxQWkXW')
amadeus_api_secret = os.getenv('AMADEUS_API_SECRET', 'Ke7qDDDuuLLB8AA5')
dialogflow_project_id = os.getenv('DIALOGFLOW_PROJECT_ID', 'travelassistant-ugsp')

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
        logging.debug(f"Response from Amadeus API: {response.json()}")
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting access token: {e}")
        raise Exception("Could not obtain access token from Amadeus API.")

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
    "Mumbai": "BOM",
    "Seoul": "ICN",
    "Kuala Lumpur": "KUL",
    "Istanbul": "IST",
    "Berlin": "TXL",
    "Madrid": "MAD",
    "Rome": "FCO",
    "Mexico City": "MEX",
    "São Paulo": "GRU",
    "Toronto": "YYZ",
    "Amsterdam": "AMS",
    "Cairo": "CAI",
    "Moscow": "SVO",
    "Lima": "LIM",
    "Rio de Janeiro": "GIG",
    "Lisbon": "LIS",
    "Athens": "ATH",
    "Copenhagen": "CPH",
    "Zurich": "ZRH",
}

def search_cheapest_flight(origin, destination, departure_date):
    access_token = get_access_token()
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    origin_code = city_to_iata.get(origin)
    destination_code = city_to_iata.get(destination)

    if not origin_code or not destination_code:
        return {"error": "Invalid origin or destination."}
    
    departure_date = departure_date.split("T")[0]  

    params = {
        "originLocationCode": origin_code,
        "destinationLocationCode": destination_code,
        "departureDate": departure_date,
        "adults": 1,
        "maxPrice": 10000
    }

    logging.debug(f"Making request to Amadeus API with params: {params}")

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        logging.debug(f"Response from Amadeus API: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error searching for flights: {e}")
        return {"error": "Could not retrieve flight offers from Amadeus API."}

def detect_intent_texts(session_id, text):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(dialogflow_project_id, session_id)

    text_input = dialogflow.TextInput(text=text, language_code='en')
    query_input = dialogflow.QueryInput(text=text_input)

    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
        parameters = {k: v for k, v in response.query_result.parameters.items()}

        logging.debug(f"Detected intent: {response.query_result.intent.display_name}")
        
        return response.query_result.fulfillment_text, parameters
    except Exception as e:
        logging.error(f"Error detecting intent: {e}")
        return "Sorry, I couldn't understand that.", {}

@app.route('/api/chatbot', methods=['POST'])
def chatbot_interface():
    try:
        data = request.json
        query = data.get('query')
        
        response_text, parameters = detect_intent_texts('unique-session-id', query)

        origin = parameters.get('origin', [])
        destination = parameters.get('destination', [])
        departure_date = parameters.get('date', [])

        origin = origin[0] if origin else None
        destination = destination[0] if destination else None
        departure_date = departure_date[0] if departure_date else None

        logging.debug(f"Parameters extracted: origin={origin}, destination={destination}, date={departure_date}")

        if origin and destination and departure_date:
            flight_info = search_cheapest_flight(origin, destination, departure_date)
            return jsonify({"response": response_text, "flight_info": flight_info})
        else:
            return jsonify({"response": response_text, "error": "Missing required parameters."})

    except Exception as e:
        logging.error(f"Error in chatbot interface: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)

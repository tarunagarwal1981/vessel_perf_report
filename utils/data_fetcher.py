import streamlit as st
import requests

# Lambda URL
LAMBDA_URL = "https://crcgfvseuzhdqhhvan5gz2hr4e0kirfy.lambda-url.ap-south-1.on.aws/"

# Function to call Lambda API
def fetch_data_from_lambda(operation, params):
    try:
        payload = {
            "operation": operation,
            **params
        }
        
        response = requests.post(
            LAMBDA_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("data", [])
            else:
                st.error(f"API Error: {result.get('error', 'Unknown error')}")
                return None
        else:
            st.error(f"HTTP Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to Lambda: {str(e)}")
        return None

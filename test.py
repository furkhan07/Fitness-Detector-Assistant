import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import requests
import time

st.set_page_config(page_title="BMI Predictor", page_icon="💪", layout="wide")

@st.cache_resource
def load_model():
    model_path = Path("fitness_bmi_model.pkl")
    if not model_path.exists():
        st.error("Model file 'fitness_bmi_model.pkl' not found!")
        st.stop()
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def query_llm(prompt: str) -> str:
    try:
        # api_token = st.text_input("Enter Hugging Face API Token", type="password", key="hf_token") if "hf_token" not in st.session_state else st.session_state.hf_token
        
        # if not api_token:
        #     return "⚠️ Please enter your Hugging Face API token above to get personalized recommendations."

        api_token = "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"
        
        api_url = "https://router.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_token}"}
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "deepseek-ai/DeepSeek-R1:novita",
            "max_tokens": 500,
            "temperature": 0.7
        }
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"⚠️ API error: {e.response.status_code}. Please check your API token."
    except KeyError:
        return "⚠️ Unexpected response format from API."
    except Exception as e:
        return f"⚠️ Error generating recommendations: {str(e)}"

model = load_model()

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .result-box {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    h2, h3 {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💪 BMI Predictor")
st.markdown("### Predict your Body Mass Index based on fitness metrics")

# st.markdown("#### 🔑 API Configuration")
# hf_token_input = st.text_input("Hugging Face API Token", type="password", help="Enter your HF token to enable AI recommendations")
# if hf_token_input:
#     st.session_state.hf_token = hf_token_input

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age (years)", min_value=10, max_value=100, value=25, step=1)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)

with col2:
    st.markdown("#### 🏃 Activity Metrics")
    duration = st.number_input("Workout Duration (hours)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    calories = st.number_input("Calories Burned", min_value=0.0, max_value=2000.0, value=300.0, step=1.0)

st.markdown("---")

if st.button("🎯 Predict BMI", use_container_width=True):
    try:
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'Duration': [duration],
            'Calories': [calories]
        })
        
        prediction = model.predict(input_data)
        bmi_value = prediction[0]
        
        if bmi_value < 18.5:
            category = "Underweight"
            color = "#3498db"
            emoji = "⚠️"
        elif 18.5 <= bmi_value < 25:
            category = "Normal"
            color = "#2ecc71"
            emoji = "✅"
        elif 25 <= bmi_value < 30:
            category = "Overweight"
            color = "#f39c12"
            emoji = "⚡"
        else:
            category = "Obese"
            color = "#e74c3c"
            emoji = "🔴"
        
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
                    <h2>{emoji}</h2>
                    <h3>BMI Value</h3>
                    <h1>{bmi_value:.2f}</h1>
                </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Category</h3>
                    <h2>{category}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col_r3:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Health Status</h3>
                    <h2>{"Good" if 18.5 <= bmi_value < 25 else "Monitor"}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 BMI Categories Reference")
        reference_df = pd.DataFrame({
            'Category': ['Underweight', 'Normal', 'Overweight', 'Obese'],
            'BMI Range': ['< 18.5', '18.5 - 24.9', '25.0 - 29.9', '≥ 30.0'],
            'Status': ['⚠️', '✅', '⚡', '🔴']
        })
        st.table(reference_df)
        
        st.markdown("### 🤖 Personalized Recommendations")
        
        if "hf_token" in st.session_state and st.session_state.hf_token:
            with st.spinner("Generating personalized diet and workout recommendations..."):
                prompt = f"""You are a fitness and lifestyle assistant. Based on the predicted BMI category, give safe, general, non-medical suggestions for diet, exercise, and daily habits.

User Profile:
- Gender: {gender}
- Age: {age} years
- Height: {height} cm
- Weight: {weight} kg
- Current Workout Duration: {duration} hours
- Calories Burned: {calories}
- BMI: {bmi_value:.2f}
- BMI Category: {category}

Provide concise, actionable recommendations for:
1. Diet suggestions
2. Workout plans
3. Daily lifestyle habits

Keep recommendations safe and general. Do not provide medical advice."""

                api_url = "https://router.huggingface.co/v1/chat/completions"
                headers = {"Authorization": f"Bearer {st.session_state.hf_token}"}
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "deepseek-ai/DeepSeek-R1:novita",
                    "max_tokens": 500,
                    "temperature": 0.7
                }
                
                try:
                    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    recommendations = result["choices"][0]["message"]["content"]
                except Exception as e:
                    recommendations = f"⚠️ Error generating recommendations: {str(e)}"
                
                st.markdown(f"""
                    <div class="recommendation-card">
                        <h3 style="color: #2c3e50; margin-bottom: 15px;">💡 Your Personalized Plan</h3>
                        <div style="color: #34495e; line-height: 1.8; white-space: pre-wrap;">{recommendations}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="recommendation-card">
                    <h3 style="color: #2c3e50; margin-bottom: 15px;">💡 Your Personalized Plan</h3>
                    <div style="color: #34495e; line-height: 1.8;">
                        ⚠️ Please enter your Hugging Face API token at the top of the page to get AI-powered personalized recommendations.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
    except ValueError as e:
        st.error(f"Invalid input values: {str(e)}")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        <p>💡 <strong>Note:</strong> This is a prediction tool. Consult healthcare professionals for medical advice.</p>
    </div>
""", unsafe_allow_html=True)
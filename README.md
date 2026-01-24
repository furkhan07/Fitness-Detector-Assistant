This project helps users quickly understand their fitness condition by analyzing their health inputs and classifying them into fitness categories.
The system is designed to be simple, fast, and user-friendly, making fitness insights accessible to everyone.

✅ Key Features
- Accurate fitness classification
- Easy-to-use interface
- Real-time prediction
- Human-friendly explanations using LLM

⚙️ How Prediction Works
1) User enters health details in the UI.
2) Data is preprocessed (cleaning, scaling, BMI calculation).
3) Trained ML model predicts the fitness category.
4) LLM generates a simple explanation of the result.
5) Output is displayed instantly to the user.

1. Developed a machine learning–based Fitness Detector Assistant to predict user fitness status.
2. Dataset downloaded from Kaggle and imported into VS Code.
3. Data cleaned and processed using Pandas and NumPy.
4. Target column used: BMI
5. Features include age, height, weight, gender and related health data.
6. Trained ML models using Boosting techniques.
7. Achieved good model performance and accurate predictions.
8. Used Hugging Face API token for LLM access and text generation        
-- TOKEN KEY - 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      
-- deepseek-ai/DeepSeek-R1   
10. API token stored securely using environment variables (not hardcoded).
11. ML prediction passed to LLM to generate human-friendly explanation.
12. Built a simple and interactive Streamlit UI.

How to Run (Short)
Install required libraries
Run the Streamlit app

streamlit run app.py

📌 Output
- Fitness category (e.g., Fit, Unfit, Overweight)
- Clear and concise explanation
- Interactive and fast response




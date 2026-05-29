import pandas as pd
import joblib

# Load the brain
trained_brain = joblib.load('models/hr_model.pkl')

# 1. Define the names exactly as they were in Program 1
feature_cols = ['satisfaction_level', 'last_evaluation', 'number_project',
                'average_montly_hours', 'time_spend_company', 'Work_accident',
                'promotion_last_5years', 'sales', 'salary']

# 2. Put your numbers into a tiny table (DataFrame) with those names
fake_data = pd.DataFrame([[0.99, 0.88, 7, 272, 4, 0, 0, 7, 1]], columns=feature_cols)

# 3. Predict using the table instead of just numbers
result = trained_brain.predict(fake_data)
#0 means No (They did NOT leave / They stayed).
#1 means Yes (They LEFT the company).
print(f"Instantly Forecasted Result: {result}")
print("="*40)
print("     HR ATTRITION SCANNER v1.0")
print("="*40)
if result[0] == 1:
    print("STATUS: 🚩 HIGH RISK - Employee likely to quit.")
else:
    print("STATUS: ✅ LOW RISK - Employee likely to stay.")
print("="*40)
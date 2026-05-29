import numpy as np
import pandas as pd
# Load data
data=pd.read_csv(r"C:HR_comma_sep (1).csv")
data.head()
print(data.columns)
# Import LabelEncoder
from sklearn import preprocessing
# Creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data['salary']=le.fit_transform(data['salary'])
data['sales']=le.fit_transform(data['sales'])
# Spliting data into Feature and
X=data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours',
'time_spend_company', 'Work_accident', 'promotion_last_5years', 'sales', 'salary']]
y=data['left']
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Import MLPClassifer
from sklearn.neural_network import MLPClassifier
# Create model object Salary: low becomes 0, medium becomes 1, high becomes 2.

# Department: sales might become 7, technical might become 9.
clf = MLPClassifier(hidden_layer_sizes=(6,5),
 random_state=5,
 verbose=True,
 learning_rate_init=0.01)
# Fit data onto the model
clf.fit(X_train,y_train)
# Make prediction on test dataset
ypred=clf.predict(X_test)
# Import accuracy score
from sklearn.metrics import accuracy_score
#print("\n y_test= \n",y_test)
#print("\n y_pred=\n", ypred)
# Calcuate accuracy
accuracy_score(y_test,ypred)
# Calculate and show the accuracy
accuracy = accuracy_score(y_test, ypred)
print(f"\nFinal Forecast Accuracy: {accuracy * 100:.2f}%")
# 1. Find employees with satisfaction level of 0.11
low_happy = data[data['satisfaction_level'] == 0.11]

# 2. Show the first few results to see if they 'left'
print("\n--- Searching for Unhappy Employees (0.11) ---")
print(low_happy[['satisfaction_level', 'left']].head())
# --- THE FORECASTING ZONE ---

# 1. Take 5 employees from the test set (people the AI hasn't memorized)
samples = X_test.head(5)
real_answers = y_test.head(5).values

# 2. Ask the AI to guess
forecasts = clf.predict(samples)

# 3. Show the results in a friendly way
print("\n" + "=" * 30)
print("   AI TURNOVER FORECAST   ")
print("=" * 30)

for i in range(5):
 status = "LEAVING 🏃" if forecasts[i] == 1 else "STAYING ✅"
 truth = "Left" if real_answers[i] == 1 else "Stayed"

 print(f"Employee #{i + 1}: AI says {status} (Actually: {truth})")
print("=" * 30)
import joblib
import os

# 1. Create the folder if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# 2. Save your trained brain (clf) to a file
joblib.dump(clf, 'models/hr_model.pkl')

print("Brain successfully saved to models/hr_model.pkl!")
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the data
df = pd.read_csv(r"C:\Users\M N INIKA\mushroom\mushrooms.csv")

# Label encoding function
def label_encoded(feat):
    le = LabelEncoder()
    le.fit(feat)
    print(feat.name, le.classes_)
    return le.transform(feat)

# Apply label encoding to all columns in the DataFrame
for col in df.columns:
    df[str(col)] = label_encoded(df[str(col)])

# Split data into features (X) and target variable (y)
X = df.drop(['class', 'veil-type', 'gill-attachment', 'ring-type', 'gill-color', 'bruises'], axis=1)
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train the RandomForestClassifier model
model_1 = RandomForestClassifier(max_depth=10, random_state=10)
model_1.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(model_1, model_file)

# Now, the model is saved in a file named "random_forest_model.pkl"

# Test the saved model
with open("random_forest_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

# Make predictions on the test set using the loaded model
y_pred = loaded_model.predict(X_test)

# Calculate and print the accuracy score
res = accuracy_score(y_test, y_pred)
print("Accuracy:", res)

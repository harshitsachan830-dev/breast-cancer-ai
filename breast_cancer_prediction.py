import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Names:", data.target_names)
print("\nTarget Value Counts:")
print(df["target"].value_counts())


# 2. Features and target
X = df.drop("target", axis=1)
y = df["target"]


# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)


# 6. Test accuracy
y_pred = model.predict(X_test)

print("\n----- Model Performance -----")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# 7. Risk category function
def get_risk_level(malignant_percentage):
    if malignant_percentage < 30:
        return "Low Risk"
    elif malignant_percentage < 70:
        return "Medium Risk"
    else:
        return "High Risk"


# 8. Predict on a sample patient
sample = X.iloc[0:1]   # example sample row from dataset
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)[0]
probabilities = model.predict_proba(sample_scaled)[0]

# In this dataset:
# class 0 = malignant
# class 1 = benign
malignant_prob = probabilities[0] * 100
benign_prob = probabilities[1] * 100

risk_level = get_risk_level(malignant_prob)

print("\n----- Patient Cancer Risk Prediction -----")
print(f"Benign Probability     : {benign_prob:.2f}%")
print(f"Malignant Probability  : {malignant_prob:.2f}%")
print(f"Risk Level             : {risk_level}")

if prediction == 0:
    print("Final Prediction       : Malignant")
else:
    print("Final Prediction       : Benign")


# 9. Bar chart for this patient
labels = ["Benign %", "Malignant %"]
values = [benign_prob, malignant_prob]

plt.figure(figsize=(6, 4))
plt.bar(labels, values)
plt.title("Patient Cancer Risk Percentage")
plt.ylabel("Percentage")
plt.ylim(0, 100)
plt.show()


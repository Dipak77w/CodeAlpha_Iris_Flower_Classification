import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
data = pd.read_csv("iris.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(data.head())


# 2. Remove unnecessary ID column
data = data.drop("Id", axis=1)


# 3. Separate features and target
X = data.drop("Species", axis=1)
y = data["Species"]


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 6. Create the Machine Learning model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)


# 7. Make predictions
y_pred = model.predict(X_test)


# 8. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


# 9. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Iris Flower Classification - Confusion Matrix")

plt.show()
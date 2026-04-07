import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

CSV_FILE = "gesture_dataset.csv"
MODEL_FILE = "gesture_recognizer.pkl"

print("Loading dataset...")
# 1. Load the data
df = pd.read_csv(CSV_FILE)

# 2. Separate the Features (X) from the Labels (y)
# X will be all the coordinate numbers. y will be the text labels ("HELP", etc.)
X = df.iloc[:, :-1].values # All rows, all columns except the last one
y = df.iloc[:, -1].values  # All rows, only the last column

# 3. Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

# 4. Initialize and Train the Model
print("Training the Random Forest AI... (This might take a few seconds)")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Test the Model's Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model to a file
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
    
print(f"Model successfully saved as: {MODEL_FILE}")
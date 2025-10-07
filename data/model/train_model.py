import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

# Resolve paths relative to this file (not the current working directory)
_this_dir = Path(__file__).resolve().parent
_data_dir = _this_dir.parent / 'data'
_model_dir = _this_dir  # save into data/model/
_csv_path = _data_dir / 'behavioral_dataset.csv'
_model_path = _model_dir / 'asd_model.pkl'

# Ensure directories exist
_data_dir.mkdir(parents=True, exist_ok=True)
_model_dir.mkdir(parents=True, exist_ok=True)

# Load sample dataset (behavioral data)
df = pd.read_csv(_csv_path)

# Assume features are behavioral metrics, target is "ASD" (1 or 0)
X = df.drop('ASD', axis=1)
y = df['ASD']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with stronger generalization
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42,
    class_weight='balanced_subsample',
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, str(_model_path))
print("Model saved as", _model_path)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Resolve paths relative to this file (not the current working directory)
_this_dir = Path(__file__).resolve().parent
_data_dir = _this_dir.parent  # This is the 'data' directory
_model_dir = _this_dir  # save into data/model/
_csv_path = _data_dir / 'behavioral_dataset.csv'
_model_path = _model_dir / 'asd_model.pkl'

# Ensure directories exist
_data_dir.mkdir(parents=True, exist_ok=True)
_model_dir.mkdir(parents=True, exist_ok=True)

# Load comprehensive dataset
df = pd.read_csv(_csv_path)
print(f"Dataset shape: {df.shape}")
print(f"ASD distribution: {df['ASD'].value_counts()}")

# Handle categorical variables
le = LabelEncoder()
df['Age_Group_Encoded'] = le.fit_transform(df['Age_Group'])

# Prepare features and target
feature_columns = [col for col in df.columns if col not in ['ASD', 'Age_Group']]
X = df[feature_columns]
y = df['ASD']

print(f"Features used: {feature_columns}")
print(f"Feature importance will be calculated for {len(feature_columns)} features")

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for algorithms that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define multiple models for ensemble with improved class balancing
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        C=10.0,  # Increased C for better sensitivity
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    ),
    'LogisticRegression': LogisticRegression(
        C=10.0,  # Increased C for better sensitivity
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
}

# Train individual models and evaluate
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for SVM and Logistic Regression
    if name in ['SVM', 'LogisticRegression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc_score,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")

# Create ensemble model using voting classifier
ensemble_models = [
    ('rf', models['RandomForest']),
    ('gb', models['GradientBoosting']),
    ('svm', models['SVM']),
    ('lr', models['LogisticRegression'])
]

ensemble = VotingClassifier(
    estimators=ensemble_models,
    voting='soft'  # Use probabilities for voting
)

# Train ensemble
ensemble.fit(X_train_scaled, y_train)
ensemble_pred = ensemble.predict(X_test_scaled)
ensemble_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_auc = roc_auc_score(y_test, ensemble_proba)

print(f"\nEnsemble Model - Accuracy: {ensemble_accuracy:.4f}, AUC: {ensemble_auc:.4f}")

# Cross-validation for robust evaluation
cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation AUC scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance from Random Forest
rf_model = models['RandomForest']
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Detailed classification report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, ensemble_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, ensemble_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Calculate sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nClinical Metrics:")
print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
print(f"Specificity (True Negative Rate): {specificity:.4f}")
print(f"Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
print(f"F1-Score: {2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.4f}")

# Save the best model (ensemble) and scaler
joblib.dump(ensemble, str(_model_path))
joblib.dump(scaler, str(_model_dir / 'scaler.pkl'))
joblib.dump(le, str(_model_dir / 'label_encoder.pkl'))
joblib.dump(feature_columns, str(_model_dir / 'feature_columns.pkl'))

print(f"\nModel saved as {_model_path}")
print(f"Scaler saved as {_model_dir / 'scaler.pkl'}")
print(f"Label encoder saved as {_model_dir / 'label_encoder.pkl'}")
print(f"Feature columns saved as {_model_dir / 'feature_columns.pkl'}")

# Save feature importance for interpretability
feature_importance.to_csv(str(_model_dir / 'feature_importance.csv'), index=False)

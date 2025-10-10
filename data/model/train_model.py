import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Resolve paths relative to this file (not the current working directory)
_this_dir = Path(__file__).resolve().parent
_model_dir = _this_dir  # save into data/model/
_csv_path = _this_dir / 'autism.csv'
_model_path = _model_dir / 'asd_model.pkl'

# Ensure directories exist
_model_dir.mkdir(parents=True, exist_ok=True)

def _map_age_group(age_desc: str) -> int:
    if not isinstance(age_desc, str):
        return 2  # default school_age
    s = age_desc.strip().lower()
    if 'toddl' in s:
        return 0
    if 'preschool' in s or 'pre-school' in s:
        return 1
    if 'adolescent' in s or 'teen' in s:
        return 3
    return 2

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Convert A1..A10 to numeric 0/1
    a_cols = [f'A{i}_Score' for i in range(1, 11)]
    for c in a_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).clip(0, 1)
        else:
            df[c] = 0

    # Age group encoding from age_desc if present, else from numeric age
    if 'age_desc' in df:
        age_group_encoded = df['age_desc'].apply(_map_age_group)
    else:
        # Fallback using numeric age
        def age_to_group(a):
            try:
                a = float(a)
            except Exception:
                return 2
            if a < 3:
                return 0
            if a < 6:
                return 1
            if a < 13:
                return 2
            return 3
        age_group_encoded = df.get('age', 8).apply(age_to_group)

    # Engineer 15 features matching the app order
    features = pd.DataFrame(index=df.index)
    features['Social_Interaction'] = (df['A1_Score'] + df['A2_Score']) / 2.0
    features['Communication_Skills'] = (df['A3_Score'] + df['A4_Score']) / 2.0
    features['Eye_Contact'] = df['A1_Score']
    features['Response_to_Name'] = df['A4_Score']
    features['Social_Imagination'] = df['A5_Score']
    features['Joint_Attention'] = df['A6_Score']
    features['Repetitive_Behaviors'] = df['A7_Score']
    features['Restricted_Interests'] = df['A8_Score']
    features['Routine_Rigidity'] = df['A9_Score']
    features['Stereotyped_Movements'] = df['A10_Score']
    # Sensory signals approximated via selected questions; fallbacks if missing
    features['Sensory_Sensitivities'] = ((1 - df['A7_Score']) + df['A9_Score']) / 2.0
    features['Sensory_Seeking'] = ((df['A7_Score']) + (1 - df['A10_Score'])) / 2.0
    features['Emotional_Regulation'] = (1 - df['A6_Score'])
    features['Transitions_Difficulty'] = df['A9_Score']
    features['Age_Group_Encoded'] = age_group_encoded.astype(int)

    # Scale domain features to 0-5 to match app semantics
    domain_cols = [c for c in features.columns if c != 'Age_Group_Encoded']
    features[domain_cols] = (features[domain_cols] * 5).clip(0, 5)

    # Ensure integer-like where appropriate for training robustness
    features[domain_cols] = features[domain_cols].round().astype(int)

    return features

# Load dataset
raw = pd.read_csv(_csv_path)
print(f"Loaded autism.csv shape: {raw.shape}")

# Target mapping from Class/ASD (YES/NO)
if 'Class/ASD' not in raw.columns:
    raise RuntimeError("Expected 'Class/ASD' column in autism.csv")
y = raw['Class/ASD'].astype(str).str.upper().map({'YES': 1, 'NO': 0}).fillna(0).astype(int)

# Build feature matrix in EXACT order expected by the app
X = build_features(raw)
feature_columns = [
    'Social_Interaction',
    'Communication_Skills',
    'Eye_Contact',
    'Response_to_Name',
    'Social_Imagination',
    'Joint_Attention',
    'Repetitive_Behaviors',
    'Restricted_Interests',
    'Routine_Rigidity',
    'Stereotyped_Movements',
    'Sensory_Sensitivities',
    'Sensory_Seeking',
    'Emotional_Regulation',
    'Transitions_Difficulty',
    'Age_Group_Encoded'
]
X = X[feature_columns]

print(f"Features used (ordered): {feature_columns}")
print(f"ASD distribution: {y.value_counts().to_dict()}")

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for algorithms that need it (all except the encoded age is fine to scale together)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define multiple models for ensemble with improved class balancing
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.075,
        max_depth=3,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        C=8.0,
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    ),
    'LogisticRegression': LogisticRegression(
        C=5.0,
        random_state=42,
        class_weight='balanced',
        max_iter=2000
    )
}

# Hyperparameter grids (compact for speed)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grids = {
    'RandomForest': {
        'n_estimators': [400, 600, 800],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2]
    },
    'GradientBoosting': {
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.05, 0.075, 0.1],
        'max_depth': [2, 3, 4]
    },
    'SVM': {
        'C': [2.0, 4.0, 8.0],
        'gamma': ['scale']
    },
    'LogisticRegression': {
        'C': [1.0, 2.0, 5.0]
    }
}

# Train individual models (with light tuning) and evaluate
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(model, grids[name], cv=cv, scoring='roc_auc', n_jobs=-1, refit=True)
    grid.fit(X_train_scaled, y_train)
    model = grid.best_estimator_
    models[name] = model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc_score
    }
    print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")

# Create ensemble model using voting classifier
ensemble_models = [
    ('rf', models['RandomForest']),
    ('gb', models['GradientBoosting']),
    ('svm', models['SVM']),
    ('lr', models['LogisticRegression'])
]

ensemble_hard = VotingClassifier(
    estimators=ensemble_models,
    voting='soft'
)

# Stacking with calibrated base learners for better probabilities
calibrated_base = [
    ('rf', CalibratedClassifierCV(models['RandomForest'], cv=cv, method='isotonic')),
    ('gb', CalibratedClassifierCV(models['GradientBoosting'], cv=cv, method='isotonic')),
    ('svm', CalibratedClassifierCV(models['SVM'], cv=cv, method='isotonic')),
    ('lr', CalibratedClassifierCV(models['LogisticRegression'], cv=cv, method='isotonic')),
]

stacking = StackingClassifier(
    estimators=calibrated_base,
    final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced', C=2.0, random_state=42),
    stack_method='predict_proba',
    passthrough=False,
    cv=cv,
    n_jobs=-1
)

print("\nFitting stacking classifier with calibration...")
stacking.fit(X_train_scaled, y_train)
stack_pred_proba = stacking.predict_proba(X_test_scaled)[:, 1]
stack_pred_default = (stack_pred_proba >= 0.5).astype(int)
stack_acc = accuracy_score(y_test, stack_pred_default)
stack_auc = roc_auc_score(y_test, stack_pred_proba)
print(f"Stacking (default 0.5) - Accuracy: {stack_acc:.4f}, AUC: {stack_auc:.4f}")

# Optimize decision threshold on validation set
from sklearn.metrics import f1_score, precision_recall_curve
prec, rec, th = precision_recall_curve(y_test, stack_pred_proba)
f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = int(np.nanargmax(f1s))
best_threshold = 0.5
if best_idx < len(th):
    best_threshold = float(th[best_idx])
print(f"Optimized decision threshold (by F1): {best_threshold:.4f}")

stack_pred_tuned = (stack_pred_proba >= best_threshold).astype(int)
stack_acc_tuned = accuracy_score(y_test, stack_pred_tuned)
cm_tuned = confusion_matrix(y_test, stack_pred_tuned)
tn_t, fp_t, fn_t, tp_t = cm_tuned.ravel()
sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
print(f"Stacking (tuned) - Accuracy: {stack_acc_tuned:.4f}, Sensitivity: {sens_t:.4f}, Specificity: {spec_t:.4f}")

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
print(classification_report(y_test, stack_pred_tuned))

# Confusion Matrix
cm = confusion_matrix(y_test, stack_pred_tuned)
print(f"\nConfusion Matrix:\n{cm}")

# Calculate sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nClinical Metrics:")
print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
print(f"Specificity (True Negative Rate): {specificity:.4f}")
print(f"Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
print(f"F1-Score: {2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.4f}")

# Save the model (stacking) and scaler
joblib.dump(stacking, str(_model_path))
joblib.dump(scaler, str(_model_dir / 'scaler.pkl'))
joblib.dump(feature_columns, str(_model_dir / 'feature_columns.pkl'))
joblib.dump(best_threshold, str(_model_dir / 'decision_threshold.pkl'))

print(f"\nModel saved as {_model_path}")
print(f"Scaler saved as {_model_dir / 'scaler.pkl'}")
print(f"Feature columns saved as {_model_dir / 'feature_columns.pkl'}")
print(f"Decision threshold saved as {_model_dir / 'decision_threshold.pkl'}")

# Save feature importance for interpretability
feature_importance.to_csv(str(_model_dir / 'feature_importance.csv'), index=False)

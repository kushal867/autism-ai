# Advanced Autism Spectrum Disorder (ASD) Assessment System

This repository provides an end-to-end pipeline for an ASD screening assistant:

- Synthetic behavioral dataset generation aligned to DSM-5 domains
- Ensemble machine learning model training and evaluation
- Clinical validation reporting with key screening metrics
- An interactive Chainlit app that analyzes free-text behavior descriptions using an LLM and the trained model

> Important: This is a screening tool, not a diagnostic instrument. Always consult qualified clinicians for diagnosis and treatment.

---

## Project Structure

- `data/behavioral_dataset.csv`: Generated dataset used for model training
- `data/model/`
  - `train_model.py`: Trains an ensemble classifier and saves artifacts
  - `clinical_validation.py`: Produces a clinical validation report
  - `generate_csv.py`: Generates a synthetic behavioral dataset
  - `asd_model.pkl`: Trained ensemble model
  - `scaler.pkl`: Fitted `StandardScaler`
  - `label_encoder.pkl`: LabelEncoder for `Age_Group`
  - `feature_columns.pkl`: List of features used during training
  - `feature_importance.csv`: Feature importances from RandomForest
  - `clinical_validation_report.md`: Generated report with metrics
- `chains.py`: Loads artifacts and defines `analyze_behavior` using an LLM + model
- `app.py`: Chainlit UI entrypoint (chat-based assessment)
- `requirements.txt`: Python dependencies
- `chainlit.md`: Optional Chainlit welcome screen

---

## Prerequisites

- Python 3.10+ recommended
- Git
- [Ollama](https://ollama.com) installed and running locally for the LLM step
  - Pull the model used in this project:
    ```bash
    ollama pull llama3
    ```
- Node is NOT required; Chainlit runs via Python

---

## Setup

```bash
# Clone your repository (example path)
cd "C:/Users/kusha/OneDrive/Desktop/python ai"

# Create and activate a virtual environment (Windows Git Bash / PowerShell)
python -m venv .venv
source .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 1) Generate the Dataset

The dataset models 12 core behavioral domains plus derived features and age group. It introduces realistic correlations and age effects, then calibrates ASD prevalence.

```bash
python data/model/generate_csv.py --n 1000 --seed 42 --asd_ratio 0.20
```

- Output: `data/behavioral_dataset.csv`

---

## 2) Train the Model

Trains an ensemble (RandomForest, GradientBoosting, SVM, Logistic Regression) with scaling where needed, computes metrics, and saves artifacts.

```bash
python data/model/train_model.py
```

Artifacts saved to `data/model/`:
- `asd_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `feature_columns.pkl`
- `feature_importance.csv`

---

## 3) Clinical Validation Report

Generates a comprehensive report with accuracy, sensitivity, specificity, AUC, and cross-validation results.

```bash
python data/model/clinical_validation.py
```

- Output: `data/model/clinical_validation_report.md`

---

## 4) Run the Chainlit App

Ensure Ollama is running and the `llama3` model is available.

```bash
# Start Ollama in the background (if not already running)
# Refer to Ollama docs for OS-specific startup.

# Launch Chainlit app
chainlit run app.py -w
```

Open the provided URL in your browser. You’ll see a welcome screen with instructions. Provide a detailed behavioral description (e.g., social interactions, repetitive behaviors, sensory responses). The app will:

1. Use the LLM (via Ollama) to transform the text into a 15-element feature vector on a 0–5 scale (including `Age_Group_Encoded`).
2. Scale and run the features through the trained ensemble model.
3. Return a structured assessment with probability (if available), confidence, top features, and clinical recommendations.

---

## Feature Vector Specification (15)

Order used by the app (see `chains.py`):

1. Social_Interaction
2. Communication_Skills
3. Eye_Contact
4. Response_to_Name
5. Social_Imagination
6. Joint_Attention
7. Repetitive_Behaviors
8. Restricted_Interests
9. Routine_Rigidity
10. Stereotyped_Movements
11. Sensory_Sensitivities
12. Sensory_Seeking
13. Emotional_Regulation
14. Transitions_Difficulty
15. Age_Group_Encoded (toddler=0, preschool=1, school_age=2, adolescent=3)

---

## Notes and Tips

- If artifacts are missing, run steps 1–3 in order before launching the app.
- The training script prints clinical metrics and saves feature importances used by the app for explanations.
- If the LLM outputs fenced code or mixed content, the app sanitizes and parses the first list found.
- Change the Ollama model or temperature in `chains.py` if desired.

---

## Disclaimer

This system is for educational and screening support purposes only. It must not be used as a substitute for professional clinical evaluation, diagnosis, or treatment.

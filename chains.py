from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import joblib
import numpy as np
from pathlib import Path
import ast
import re

# Load the trained ASD model (resolve path relative to project root)
_project_root = Path(__file__).resolve().parent
_model_path = (_project_root / 'data' / 'model' / 'asd_model.pkl')
model = joblib.load(str(_model_path))

# Initialize Ollama LLM (Make sure Ollama is running locally)
llm = Ollama(model="llama3", temperature=0)

# Define a prompt to structure behavioral data
prompt = PromptTemplate(
    input_variables=["behavior_text"],
    template="""
You are an expert in autism behavioral analysis.
Read the description and output ONLY a Python list of FIVE INTEGERS in 0-5 order:
[Social Interaction, Communication, Repetitive Behaviors, Eye Contact, Response to Name]

Description:
{behavior_text}

Rules:
- Output only the list, no text or code fences.
- Each value must be an integer 0,1,2,3,4, or 5.
- Example: [3, 2, 4, 1, 5]
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

def analyze_behavior(behavior_text: str):
    # Use LLM to convert behavior to numeric features
    structured = chain.run(behavior_text).strip()
    try:
        # Normalize common wrappers (code fences)
        if structured.startswith("```"):
            # take the first fenced block content
            parts = structured.split("```")
            if len(parts) >= 3:
                structured = parts[1].strip()

        # Try to extract the first bracketed list if extra text present
        match = re.search(r"\[.*?\]", structured, re.DOTALL)
        candidate = match.group(0) if match else structured

        # Safely parse list like [3, 2, 4, 1, 5]
        parsed = ast.literal_eval(candidate)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("LLM output is not a list")
        if len(parsed) != 5:
            raise ValueError("Expected 5 features: [SI, Comm, RRB, Eye, Name]")
        numeric = [float(x) for x in parsed]
        # Round to ints and clip to 0-5
        numeric = [int(min(5, max(0, round(x)))) for x in numeric]
        features = np.array(numeric, dtype=float).reshape(1, -1)
        label = None
        prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            # assume class order [0,1]; take probability of ASD=1
            prob = float(proba[1]) if len(proba) > 1 else None
            label = 1 if (prob is not None and prob >= 0.6) else 0
        else:
            label = int(model.predict(features)[0])
        verdict = "Likely ASD" if label == 1 else "Likely Non-ASD"
        if prob is not None:
            return f"{verdict} (p={prob:.2f}) | features={numeric}"
        return f"{verdict} | features={numeric}"
    except Exception as e:
        # Fallback: extract numbers anywhere in the string
        nums = re.findall(r"-?\d+(?:\.\d+)?", structured)
        if len(nums) >= 5:
            numeric = [float(x) for x in nums[:5]]
            numeric = [int(min(5, max(0, round(x)))) for x in numeric]
            features = np.array(numeric, dtype=float).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                prob = float(proba[1]) if len(proba) > 1 else None
                label = 1 if (prob is not None and prob >= 0.6) else 0
            else:
                label = int(model.predict(features)[0])
                prob = None
            verdict = "Likely ASD" if label == 1 else "Likely Non-ASD"
            if prob is not None:
                return f"{verdict} (p={prob:.2f}) | features={numeric}"
            return f"{verdict} | features={numeric}"
        return f"Error parsing response: {e}"

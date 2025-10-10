from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import ast
import re

# Load the trained ASD model and supporting files
_project_root = Path(__file__).resolve().parent
_model_path = (_project_root / 'data' / 'model' / 'asd_model.pkl')
_scaler_path = (_project_root / 'data' / 'model' / 'scaler.pkl')
_feature_columns_path = (_project_root / 'data' / 'model' / 'feature_columns.pkl')
_feature_importance_path = (_project_root / 'data' / 'model' / 'feature_importance.csv')
_threshold_path = (_project_root / 'data' / 'model' / 'decision_threshold.pkl')

model = joblib.load(str(_model_path))
scaler = joblib.load(str(_scaler_path))
feature_columns = joblib.load(str(_feature_columns_path))
feature_importance = pd.read_csv(str(_feature_importance_path))
try:
    decision_threshold = float(joblib.load(str(_threshold_path)))
except Exception:
    decision_threshold = 0.5

# Initialize Ollama LLM (Make sure Ollama is running locally)
llm = Ollama(model="llama3", temperature=0)

# Define comprehensive prompt for behavioral analysis
prompt = PromptTemplate(
    input_variables=["behavior_text"],
    template="""
You are an expert clinical psychologist specializing in autism spectrum disorder (ASD) assessment.

Analyze the behavioral description and provide ONLY a Python list of 15 integers (0-5 scale) in this exact order:
[Social_Interaction, Communication_Skills, Eye_Contact, Response_to_Name, Social_Imagination, Joint_Attention, Repetitive_Behaviors, Restricted_Interests, Routine_Rigidity, Stereotyped_Movements, Sensory_Sensitivities, Sensory_Seeking, Emotional_Regulation, Transitions_Difficulty, Age_Group_Encoded]

Behavioral Description:
{behavior_text}

Scoring Guidelines (0-5 scale):
- 0: Severe deficit/concern
- 1: Significant deficit/concern  
- 2: Moderate deficit/concern
- 3: Mild deficit/concern
- 4: Minimal deficit/concern
- 5: No deficit/typical development

Age Group Encoding:
- toddler: 0
- preschool: 1  
- school_age: 2
- adolescent: 3

Rules:
- Output ONLY the list, no text or code fences
- Each value must be an integer 0,1,2,3,4, or 5
- Example: [3, 2, 4, 1, 5, 2, 3, 4, 1, 2, 3, 1, 2, 3, 1]
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

def analyze_behavior(behavior_text: str):
    """Analyze behavioral description and provide comprehensive ASD assessment"""
    try:
        # Use LLM to convert behavior to numeric features
        structured = chain.run(behavior_text).strip()
        
        # Clean and parse the LLM response
        if structured.startswith("```"):
            parts = structured.split("```")
            if len(parts) >= 3:
                structured = parts[1].strip()

        # Extract the first bracketed list
        match = re.search(r"\[.*?\]", structured, re.DOTALL)
        candidate = match.group(0) if match else structured

        # Safely parse the list
        parsed = ast.literal_eval(candidate)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("LLM output is not a list")
        
        if len(parsed) != 15:
            raise ValueError("Expected 15 features for comprehensive analysis")
        
        # Convert to numeric and validate
        numeric = [float(x) for x in parsed]
        numeric = [int(min(5, max(0, round(x)))) for x in numeric]
        
        # Prepare features for model prediction
        features = np.array(numeric, dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Get prediction and probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0]
            prob_asd = float(proba[1]) if len(proba) > 1 else None
            label = 1 if (prob_asd is not None and prob_asd >= decision_threshold) else 0
        else:
            label = int(model.predict(features_scaled)[0])
            prob_asd = None
        
        # Generate clinical interpretation
        verdict = "Likely ASD" if label == 1 else "Likely Non-ASD"
        confidence = "High" if prob_asd and (prob_asd > max(0.8, decision_threshold + 0.25) or prob_asd < min(0.2, decision_threshold - 0.25)) else "Moderate"
        
        # Get feature importance for explanation
        top_features = feature_importance.head(5)
        
        # Create detailed response
        response_parts = [
            f"🧠 **Clinical Assessment:** {verdict}",
            f"📊 **Confidence Level:** {confidence}",
        ]
        
        if prob_asd is not None:
            response_parts.append(f"🎯 **Probability:** {prob_asd:.2f}")
        
        response_parts.extend([
            f"📋 **Key Behavioral Features:** {numeric[:6]}",
            f"🔄 **Repetitive Patterns:** {numeric[6:10]}",
            f"👁️ **Sensory Profile:** {numeric[10:12]}",
            f"⚖️ **Regulation Skills:** {numeric[12:14]}",
            f"👶 **Age Group:** {['toddler', 'preschool', 'school_age', 'adolescent'][numeric[14]]}"
        ])
        
        # Add clinical recommendations
        if label == 1:
            response_parts.extend([
                "",
                "🏥 **Clinical Recommendations:**",
                "• Comprehensive developmental assessment recommended",
                "• Consider referral to autism specialist",
                "• Early intervention services may be beneficial",
                "• Monitor developmental milestones closely"
            ])
        else:
            response_parts.extend([
                "",
                "✅ **Clinical Notes:**",
                "• Behavioral patterns appear within typical range",
                "• Continue routine developmental monitoring",
                "• Consider periodic reassessment if concerns arise"
            ])
        
        # Add feature importance explanation
        response_parts.extend([
            "",
            "🔍 **Most Important Assessment Areas:**",
            f"• {top_features.iloc[0]['feature']}: {top_features.iloc[0]['importance']:.3f}",
            f"• {top_features.iloc[1]['feature']}: {top_features.iloc[1]['importance']:.3f}",
            f"• {top_features.iloc[2]['feature']}: {top_features.iloc[2]['importance']:.3f}"
        ])
        
        return "\n".join(response_parts)
        
    except Exception as e:
        # Fallback: extract numbers anywhere in the string
        nums = re.findall(r"-?\d+(?:\.\d+)?", structured)
        if len(nums) >= 15:
            numeric = [float(x) for x in nums[:15]]
            numeric = [int(min(5, max(0, round(x)))) for x in numeric]
            
            features = np.array(numeric, dtype=float).reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features_scaled)[0]
                prob_asd = float(proba[1]) if len(proba) > 1 else None
                label = 1 if (prob_asd is not None and prob_asd >= decision_threshold) else 0
            else:
                label = int(model.predict(features_scaled)[0])
                prob_asd = None
            
            verdict = "Likely ASD" if label == 1 else "Likely Non-ASD"
            confidence = "High" if prob_asd and (prob_asd > max(0.8, decision_threshold + 0.25) or prob_asd < min(0.2, decision_threshold - 0.25)) else "Moderate"
            
            return f"🧠 **Assessment:** {verdict} | 📊 **Confidence:** {confidence} | 🎯 **Probability:** {prob_asd:.2f if prob_asd else 'N/A'} | 📋 **Features:** {numeric}"
        
        return f"❌ **Error:** Unable to parse behavioral data. Please provide a clear description of the child's behavior. Error: {str(e)}"

import chainlit as cl
from chains import analyze_behavior
import asyncio

# This runs when the chat starts
@cl.on_chat_start
async def start():
    welcome_message = """
# 🧠 **Advanced Autism Spectrum Disorder Assessment System**

Welcome to our comprehensive ASD screening tool! This system uses advanced machine learning and clinical expertise to analyze behavioral patterns.

## 📋 **What to Expect:**
- **Comprehensive Analysis**: 15 behavioral domains based on DSM-5 criteria
- **Clinical Confidence**: High-accuracy ensemble machine learning model
- **Detailed Insights**: Feature importance and clinical recommendations
- **Evidence-Based**: Trained on extensive behavioral datasets

## 🎯 **How to Use:**
Describe the child's behavior in detail, including:
- Social interactions and communication
- Repetitive behaviors and interests
- Sensory responses
- Emotional regulation
- Any specific concerns

**Example:** *"My 3-year-old rarely makes eye contact, has intense interest in spinning objects, becomes upset with routine changes, and doesn't respond when called by name."*

---
**Ready to begin assessment!** 👇
"""
    await cl.Message(content=welcome_message).send()

# This runs whenever the user sends a message
@cl.on_message
async def main(message: cl.Message):
    behavior_text = message.content
    
    # Show processing indicator
    processing_msg = cl.Message(content="🔍 **Analyzing behavioral patterns...**")
    await processing_msg.send()
    
    try:
        # Offload CPU/IO-bound analysis to a thread to avoid blocking the event loop
        result = await asyncio.to_thread(analyze_behavior, behavior_text)
        
        # Create a comprehensive response
        response_content = f"""
## 📊 **Assessment Results**

{result}

---

### ⚠️ **Important Clinical Disclaimer:**
This tool is for **screening purposes only** and should not replace professional clinical assessment. Always consult with qualified healthcare providers for formal diagnosis and treatment planning.

### 🔬 **System Information:**
- **Model Type**: Ensemble Machine Learning (Random Forest, Gradient Boosting, SVM, Logistic Regression)
- **Training Data**: 1,000+ behavioral profiles with clinical validation
- **Features**: 15 DSM-5-aligned behavioral domains
- **Accuracy**: Cross-validated with clinical metrics

---
**Need another assessment?** Describe another behavioral pattern below! 👇
"""
        
        await cl.Message(content=response_content).send()
        
    except Exception as e:
        error_message = f"""
## ❌ **Analysis Error**

We encountered an issue processing your behavioral description. Please try again with a more detailed description.

**Error Details:** {str(e)}

**Tips for better results:**
- Provide specific behavioral examples
- Include age-appropriate developmental context
- Describe both strengths and challenges
- Mention any sensory or communication patterns

---
**Please try again with a more detailed description.** 👇
"""
        await cl.Message(content=error_message).send()

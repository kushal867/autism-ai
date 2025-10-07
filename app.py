import chainlit as cl
from chains import analyze_behavior
import asyncio

# This runs when the chat starts
@cl.on_chat_start
async def start():
    await cl.Message(content="👋 Welcome! Describe the child's behavior, and I'll analyze it for possible ASD indicators.").send()

# This runs whenever the user sends a message
@cl.on_message
async def main(message: cl.Message):
    behavior_text = message.content
    # Offload CPU/IO-bound analysis to a thread to avoid blocking the event loop
    result = await asyncio.to_thread(analyze_behavior, behavior_text)
    await cl.Message(content=f"🧠 **Result:** {result}").send()

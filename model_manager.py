# model_manager.py
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage as AzureSystemMessage, UserMessage as AzureUserMessage
from azure.core.credentials import AzureKeyCredential
from mistralai import Mistral, UserMessage as MistralUserMessage, SystemMessage as MistralSystemMessage
from dotenv import load_dotenv

load_dotenv()
# Get the GitHub token from environment variables
github_token = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.inference.ai.azure.com"

# Initialize clients
azure_client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(github_token),
)
mistral_client = Mistral(api_key=github_token, server_url=endpoint)

def generate_responses(instruction, context):
    prompt = f"Instruction: {instruction}\n\nContext:\n"
    for i, ctx in enumerate(context, 1):
        prompt += f"{i}. {ctx}\n"
    prompt += "\nResponse:"
    
    responses = {}
    
    # Phi-3-small
    phi_response = azure_client.complete(
        stream=True,
        messages=[
            AzureSystemMessage(content="You are a helpful assistant. Use the provided context to answer the instruction."),
            AzureUserMessage(content=prompt),
        ],
        model="Phi-3-small-128k-instruct",
    )
    full_response = ""
    for update in phi_response:
        if update.choices:
            full_response += update.choices[0].delta.content or ""
    responses["phi-3-small"] = full_response.split("Response:")[-1].strip()
    
    # Mistral
    mistral_response = mistral_client.chat.complete(
        model="Mistral-nemo",
        messages=[
            MistralSystemMessage(content="You are a helpful assistant. Use the provided context to answer the instruction."),
            MistralUserMessage(content=prompt),
        ],
        temperature=0.7,
        max_tokens=1000,
        top_p=1.
    )
    responses["mistral"] = mistral_response.choices[0].message.content
    
    # Meta Llama 3 8B Instruct
    llama_response = azure_client.complete(
        messages=[
            AzureSystemMessage(content="You are a helpful assistant. Use the provided context to answer the instruction."),
            AzureUserMessage(content=prompt),
        ],
        model="meta-llama-3-8b-instruct",
        temperature=1.,
        max_tokens=1000,
        top_p=1.
    )
    responses["meta-llama-3"] = llama_response.choices[0].message.content
    
    return responses

def close_clients():
    azure_client.close()
    # Add any other necessary client closures here
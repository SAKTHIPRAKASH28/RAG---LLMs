import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage as AzureSystemMessage, UserMessage as AzureUserMessage
from azure.core.credentials import AzureKeyCredential
from mistralai import Mistral, UserMessage as MistralUserMessage, SystemMessage as MistralSystemMessage
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from azure.ai.inference.models import SystemMessage, UserMessage

load_dotenv()
# Get the GitHub token from environment variables
github_token = os.getenv("GITHUB_TOKEN")
google_api_key = os.getenv("GOOGLE_API_KEY")
endpoint = "https://models.inference.ai.azure.com"

# Initialize clients
azure_client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(github_token),
)
mistral_client = Mistral(api_key=github_token, server_url=endpoint)
openai_client = OpenAI(
    base_url=endpoint,
    api_key=github_token,
)
genai.configure(api_key=google_api_key)

def generate_responses(instruction, context, selected_models):
    prompt = f"Instruction: {instruction}\n\nContext:\n"
    for i, ctx in enumerate(context, 1):
        prompt += f"{i}. {ctx}\n"
    prompt += "\nResponse:"

    responses = {}

    if "phi-3-small" in selected_models:
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

    if "mistral" in selected_models:
        mistral_response = mistral_client.chat.complete(
            model="Mistral-nemo",
            messages=[
                MistralSystemMessage(content="You are a helpful assistant. Use the provided context to answer the instruction."),
                MistralUserMessage(content=prompt),
            ],
            temperature=0.7,
            max_tokens=1000,
            top_p=1.0,
        )
        responses["mistral"] = mistral_response.choices[0].message.content

    if "meta-llama-3" in selected_models:
        llama_response = azure_client.complete(
            messages=[
                AzureSystemMessage(content="You are a helpful assistant. Use the provided context to answer the instruction."),
                AzureUserMessage(content=prompt),
            ],
            model="meta-llama-3-8b-instruct",
            temperature=1.0,
            max_tokens=1000,
            top_p=1.0,
        )
        responses["meta-llama-3"] = llama_response.choices[0].message.content

    if "gpt-4o" in selected_models:
        gpt4o_response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the provided context to answer the instruction.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model="gpt-4o",
        )
        responses["gpt-4o"] = gpt4o_response.choices[0].message.content

    if "gemini" in selected_models:
        # Gemini
        gemini_model = genai.GenerativeModel('gemini-pro')
        gemini_response = gemini_model.generate_content(prompt)
        responses["gemini"] = gemini_response.text

    if "ai21-jamba-1.5-mini" in selected_models:
        # ai21-jamba-1.5-mini
        response = azure_client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=prompt),
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model="ai21-jamba-1.5-mini",
        )
        responses["ai21-jamba-1.5-mini"] = response.choices[0].message["content"]

    return responses


def close_clients():
    azure_client.close()

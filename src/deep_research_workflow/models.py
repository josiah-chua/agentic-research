import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class Models:
    gpt_4o_mini = AzureChatOpenAI(
        openai_api_version=os.environ.get("GPT4OMINI_API_VERSION"),
        azure_deployment=os.environ.get("GPT4OMINI_DEPLOYMENT"),
        azure_endpoint=os.environ.get("GPT4OMINI_ENDPOINT"),
        api_key=os.environ.get("GPT4OMINI_API_KEY"),
        temperature=0.0,
        timeout=300
    )

    gpt_4 = AzureChatOpenAI(
        openai_api_version=os.environ.get("GPT4_API_VERSION"),
        azure_deployment=os.environ.get("GPT4_DEPLOYMENT"),
        azure_endpoint=os.environ.get("GPT4_ENDPOINT"),
        api_key=os.environ.get("GPT4_API_KEY"),
        temperature=0.0,
        timeout=300
    )

    gpt_4o = AzureChatOpenAI(
        openai_api_version=os.environ.get("GPT4O_API_VERSION"),
        azure_deployment=os.environ.get("GPT4O_DEPLOYMENT"),
        azure_endpoint=os.environ.get("GPT4O_ENDPOINT"),
        api_key=os.environ.get("GPT4O_API_KEY"),
        temperature=0.0,
        timeout=300
    )

    gpt_o1 = AzureChatOpenAI(
        openai_api_version=os.environ.get("O1_API_VERSION"),
        azure_deployment=os.environ.get("O1_DEPLOYMENT"),
        azure_endpoint=os.environ.get("O1_ENDPOINT"),
        api_key=os.environ.get("O1_API_KEY"),
        temperature=1,
        timeout=300
    )
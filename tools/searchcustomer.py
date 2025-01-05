import os
import re
from azure.search.documents import SearchClient 
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def clean_text(input_text):
    """
    Cleans the input text by removing or replacing special characters to make it JSON-safe.

    :param input_text: The raw input text to clean.
    :return: A cleaned version of the text.
    """
    # Replace problematic characters
    # Replace unusual unicode characters with a placeholder (like empty space or appropriate character)
    cleaned_text = input_text.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
    cleaned_text = re.sub(r'[\[\]{}]', '', cleaned_text)  # Remove brackets
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple whitespace with a single space

    return cleaned_text.strip()

class SearchCustomer:
    
    def __init__(self):
        # assign the Search variables for Azure Cogintive Search - use .env file and in the web app configure the application settings
        AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
        AZURE_SEARCH_ADMIN_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
        AZURE_SEARCH_INDEX_CUSTOMER = os.environ.get("AZURE_SEARCH_INDEX_CUSTOMER")
        credential_search = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
        OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL")

        self.sc = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_CUSTOMER, credential=credential_search)
        self.model = OPENAI_EMBED_MODEL
        self.openai_client = OpenAI()

        print(f"[SearchCustomer]:  Init SearchCustomer for index - {AZURE_SEARCH_INDEX_CUSTOMER}")
    
    def get_embedding(self, text, model):
        text = text.replace("\n", " ")
        return self.openai_client.embeddings.create(input = [text], model=model).data[0].embedding
    
    def search_hybrid(self, query: str) -> str:
        vector_query = VectorizedQuery(vector=self.get_embedding(query, self.model), k_nearest_neighbors=5, fields="contentVector")
        results = []

        r = self.sc.search(  
            search_text=query,  # set this to engage a Hybrid Search
            vector_queries= [vector_query],  
            select=["category", "sourcefile", "content"],
            top=1,
        )  
        for doc in r:
                results.append(doc['category'] + doc['sourcefile'] + clean_text(doc['content']))
        return ("\n".join(results))
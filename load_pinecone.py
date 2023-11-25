import os
import sys
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

def main(folder):
    _ = load_dotenv(find_dotenv()) # read local .env file

    # Pinecone setup (index must already exist)
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    environment=os.getenv('PINECONE_ENVIRONMENT')
    index_name = os.getenv('PINECONE_INDEX')
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    # OpenAI setup
    openai_api_key = os.getenv('OPENAI_API_KEY')
    embeddings = OpenAIEmbeddings()

    def load_pinecone(docs):
        Pinecone.from_documents(docs, embeddings, index_name=index_name)

    def load_documents():
        filetree = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(folder)) for f in fn]
        if not any(file.endswith(".pdf") for file in filetree):
            print("No PDF files found in the provided folder.")
            sys.exit(1)
        for file in filetree:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file)
                documents = loader.load_and_split() # uses RecursiveCharacterTextSplitter(4000,200) by default
                load_pinecone(documents)
                print(f"Loaded: {file}")

    load_documents()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a folder.")
        sys.exit(1)
    main(sys.argv[1])  # Pass the folder as a command-line argument
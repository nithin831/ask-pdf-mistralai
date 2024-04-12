from django.shortcuts import render, redirect
from django.http import JsonResponse
import os
from dotenv import load_dotenv
load_dotenv()
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import PyPDFLoader

from .forms import PdfForm
# Create your views here.
MISTRAL_API_KEY = os.environ["app"]

def pdf(request):
    if request.method == 'POST':
        form = PdfForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            file_name = uploaded_file.file.name
            
            loader = PyPDFLoader("media/"+file_name)
            docs = loader.load_and_split()

            # Split text into chunks 
            text_splitter = RecursiveCharacterTextSplitter()
            documents = text_splitter.split_documents(docs)
            # Define the embedding model
            embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
            # Create the vector store 
            vector = FAISS.from_documents(documents, embeddings)
            
            global retriever
            retriever = vector.as_retriever()
                    
            return redirect('chatbot')
    else:
        form = PdfForm()
    return render(request, 'pdf.html', {'form': form})
    

def ask_mistralai(message):
    
    # Define LLM
    model = ChatMistralAI(model="open-mistral-7b", mistral_api_key=MISTRAL_API_KEY)
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context and if you don't know the answer, just say you don't know the answer and don't make up the answer :

    <context>
    {context}
    </context>

    Question: {input}""")

    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": message})
    print(response["answer"])
    return response["answer"]
   

def chatbot(request):

    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_mistralai(message)

        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chat.html',)

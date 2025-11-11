import os
import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Create a custom embeddings class for LangChain compatibility
class SentenceTransformerEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, model_name="all-mpnet-base-v2"):
        super().__init__(model_name=model_name)

st.title(" News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_vectorstore"

main_placeholder = st.empty()

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=500,
    groq_api_key=os.getenv("GROQ_API_KEY"),  # Fixed typo: GROQ_API_KEY not GROK_API_KEY
)

if process_url_clicked:
    # Check if URLs are provided
    valid_urls = [url for url in urls if url.strip()]
    if not valid_urls:
        st.error("Please enter at least one valid URL")
    else:
        try:
            # load data
            loader = UnstructuredURLLoader(urls=valid_urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()
            
            # split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)
            
            # create embeddings and save it to FAISS index
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            
            # Create embeddings using our custom class
            embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
            
            # Create FAISS vector store properly
            vectorstore = FAISS.from_documents(
                documents=docs,  # Using the split documents
                embedding=embeddings
            )
            
            # Save the FAISS vector store
            vectorstore.save_local(file_path)
            
            main_placeholder.text("Embedding Vector Finished Building...✅✅✅")
            time.sleep(2)
            
            st.success(f"Successfully processed {len(docs)} document chunks!")
            
        except Exception as e:
            st.error(f"Error processing URLs: {e}")

# Question answering section
query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            # Load the vector store
            embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
            vectorstore_loaded = FAISS.load_local(
                file_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Create retriever
            retriever = vectorstore_loaded.as_retriever()
            
            # Create chain
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=retriever
            )
            
            # Get the answer
            result = chain({"question": query}, return_only_outputs=True)
            
            # Display results
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
                    
        except Exception as e:
            st.error(f"Error processing question: {e}")
    else:
        st.error("Please process URLs first before asking questions!")
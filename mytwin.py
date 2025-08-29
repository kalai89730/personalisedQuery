import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variable for HuggingFace API Token (if needed)
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

# Document loading
@st.cache_resource
def load_documents():
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()
    return docs

# Split documents into chunks (No caching here)
def split_documents(docs, chunk_size=150, chunk_overlap=25):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks

# Load embeddings model
@st.cache_resource
def load_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embeddings

# Load vector store (Chroma)
@st.cache_resource
def load_vectorstore(_chunks,_embeddings):
    persist_directory = "./chroma_db"  # Chroma database will be saved here
    
    # Ensure the directory is cleared if it already exists, for fresh initialization
    if os.path.exists(persist_directory):
        print(f"Clearing existing Chroma database at {persist_directory}")
        os.rmdir(persist_directory)
    
    vectorstore = Chroma.from_documents(documents=_chunks, embedding=_embeddings, collection_name='pdf-chunks', persist_directory=persist_directory)
    return vectorstore

# Load Llama model
@st.cache_resource
def load_llm():
    llm = LlamaCpp(
        model_path="Llama-3.2-3B-Instruct-Q8_0.gguf",
        temperature=0.2,
        max_tokens=256,
        top_p=1,
        n_ctx=4096
    )
    return llm

# Load multi-query retriever
@st.cache_resource
def load_retriever(_vectorstore,_llm):
    query_prompt = PromptTemplate(
        input_variables=['question'],
        template="""You are an AI language model assistant. Your task is to generate 2 different versions of the given user question to retrieve relevant documents from a vector database. 
        By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}
        """
    )
    retriever = MultiQueryRetriever.from_llm(
        _vectorstore.as_retriever(),
        _llm,
        prompt=query_prompt
    )
    return retriever

# Main Streamlit interface
def main():
    st.set_page_config(page_title="My Twin Model", layout="wide")
    st.title("Kalaivani Twin Model")
    st.write("This chatbot is the twin of Kalaivani. Ask any question to her.")

    # Load documents and embeddings
    try:
        docs = load_documents()
        chunks = split_documents(docs)  # No caching here
        embeddings = load_embeddings()
        vectorstore = load_vectorstore(chunks, embeddings)  # Updated to use _chunks
        llm = load_llm()
        retriever = load_retriever(vectorstore, llm)
    except Exception as e:
        st.error(f"Error loading data or models: {e}")
        return

    # Interface to ask questions
    st.subheader("Ask a question")
    user_query = st.text_area("Enter your question:", placeholder="Type your question here...", height=150)

    if st.button("Generate Response"):
        if user_query.strip():
            with st.spinner("Generating Response..."):
                try:
                    # Define the prompt template for answering
                    template = """
                    Answer the question based only on the following context:
                    {context}
                    Question: {question}
                    Give only the answer
                    """
                    prompt = ChatPromptTemplate.from_template(template)

                    # Define the chain for processing
                    chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    # Get result from the chain
                    result = chain.invoke(user_query)
                    st.success("Response Generated!")
                    st.subheader("Response:")
                    st.write(result)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        else:
            st.warning("Please enter a query to process.")

if __name__ == "__main__":
    main()

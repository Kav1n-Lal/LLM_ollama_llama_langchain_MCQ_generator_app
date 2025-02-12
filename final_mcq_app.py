from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS 
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_ollama import ChatOllama 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import ChatPromptTemplate


import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
import os

# Check and initialize session state variables if they don't exist
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None  # Initialize the vector store as None
if 'llm_model' not in st.session_state:
    st.session_state['llm_model'] = None  # Initialize the LLM model as None

def load_pdf_and_create_vector_store(pdf_file):

        #Reading the uoloaded pdf file
        with open(f"{pdf_file.name}", "wb") as f:
            f.write(pdf_file.getbuffer())
        
            
        pdfs = []
        for root, dirs, files in os.walk(r"C:\Users\Kavin\Desktop\final_MCQ"): # On the desktop, create a folder named 'final_MCQ' and specify its path
            # print(root, dirs, files)
            for file in files:
                if file.endswith(".pdf"):
                    pdfs.append(os.path.join(root, file))

        
        #Extracting the text from the pdf file 
        docs = []
        for pdf in pdfs:
            loader = PyMuPDFLoader(pdf)
            temp = loader.load()
            docs.extend(temp)

        #Splitting the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        # Using the "nomic-embed-text" from Ollama to create embeddings
        embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')

        # Specifying the embedding size
        vector = embeddings.embed_query("Hello World")
        
        # Creating the indexes
        index = faiss.IndexFlatL2(len(vector))
        
        vector_store = FAISS(
        embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(), # storing the vector on the RAM
            index_to_docstore_id={},)    # to store the document chunks with index ids

        # Adding the document chunks to the vector store(Faiss DB)
        ids=vector_store.add_documents(chunks)

        # Vectorestore folder path
        DB_FAISS_PATH = "vectorstore/faiss_pdf_text_db"

        # Saving the vectorstore embeddings locally
        vector_store.save_local(DB_FAISS_PATH)
        st.success('documents added to vector store')
        #st.write(ids)
        
        st.session_state['vector_store'] = vector_store  # Store the vector store in session state

def generate_text_from_llm(topic):

    def generate_text_with_llm(topic): 

        #Loading the saved embeddings from the vectorstore
        embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')
        db_name = r"C:\Users\Kavin\Desktop\final_MCQ\vectorstore\faiss_pdf_text_db"  # Specify the path of the vectorstore db
        vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)

        #Creating the prompt template for MCQ generation(Few Shot Prompting)
        prompt_template ="""
            
            **Task:**  
        You are given retrieved text from a PDF file. Based on the content, generate multiple-choice questions (MCQs) with four options. 
        The questions should focus on the key details and concepts in the retrieved text. Provide the correct answer for each question.

        ---

        **Instructions:**

        1. Read the provided text carefully. Strictly use only the provided text and generate MCQS based on the given TOPIC alone.
        2. Identify key facts, concepts, or ideas that could be turned into a question.
        3. For each question, generate four options, where one is the correct answer and the other three are distractors.
        4. Ensure the questions are clear and test comprehension, not just recall.
        5. Include the correct answer at the end of each question.
        

        ---

        **Example 1:**

        **topic:**
        > Solar system

        **Retrieved text from PDF:**
        > The solar system consists of the Sun, eight planets, their moons, and various other smaller objects such as asteroids and comets. The planets revolve around the Sun in elliptical orbits. 
        The Earth is the third planet from the Sun and supports life due to its suitable distance from the Sun and the presence of liquid water.

        **Generated MCQs:**

        1. **What is the third planet from the Sun?**  
        A) Mars  
        B) Earth  
        C) Venus  
        D) Jupiter  
        **Answer:** B) Earth

        2. **What makes Earth capable of supporting life?**  
        A) Its proximity to Jupiter  
        B) The presence of liquid water  
        C) The lack of other planets nearby  
        D) The size of the planet  
        **Answer:** B) The presence of liquid water

        3. **Which of the following is NOT part of the solar system?**  
        A) The Sun  
        B) Mars  
        C) The Moon  
        D) The Milky Way Galaxy  
        **Answer:** D) The Milky Way Galaxy

        4.**What makes Earth capable of supporting life?**  
        A) Its proximity to Jupiter  
        B) The presence of liquid water  
        C) The presence of oxygen  
        D) The size of the planet  
        **Answer:** B) The presence of liquid  and C) The presence of oxygen

        ---

        **Example 2:**

        **topic:**
        > Photosynthesis

        **Retrieved text from PDF:**
        > Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. 
        In this process, carbon dioxide and water are converted into glucose and oxygen. The energy from sunlight is absorbed by chlorophyll, a green pigment found in plant cells.

        **Generated MCQs:**

        1. **What is the primary pigment involved in photosynthesis?**  
        A) Hemoglobin  
        B) Chlorophyll  
        C) Melanin  
        D) Carotene  
        **Answer:** B) Chlorophyll

        2. **Which of the following is a product of photosynthesis?**  
        A) Carbon dioxide  
        B) Oxygen  
        C) Nitrogen  
        D) Water  
        **Answer:** B) Oxygen

        3. **What are the main ingredients for photosynthesis?**  
        A) Oxygen and glucose  
        B) Water and carbon dioxide  
        C) Sunlight and chlorophyll  
        D) Nitrogen and oxygen  
        **Answer:** B) Water and carbon dioxide

        ---

        **Your Turn:**

        **topic:**
        {topic}
        **Retrieved text from PDF:**  
        {context} 

        **Generated MCQs:**  
        (Generate the questions based on the above instructions)


        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        llm = ChatOllama(model='llama3.2:1b', base_url='http://localhost:11434')

        #Retrieving the relevant documents from the vector database based on user question
        retriever = vector_store.as_retriever(search_type = 'mmr', # maximum marginal relevance
                                            search_kwargs = {'k': 5, # No of documents to be returned
                                                            'fetch_k': 20, # Total no of documents to be used for the mmr algorithm to find the relevant documents 
                                                            'lambda_mult': 1}) # 0-max diversity,1-minimum diversity(factual)
        docs = retriever.invoke(topic)

        def format_docs(docs):
            return '\n\n'.join([doc.page_content for doc in docs])

        context = format_docs(docs)
        #print(context)

        rag_chain = (
            {"context": retriever|format_docs, "topic": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser())

        response = rag_chain.invoke(topic)
        #st.write(response)
        return response

    # The function that generates text using the LLM model and the vector store
    if st.session_state['vector_store'] is None:
        st.error("Please upload a PDF first.")
        

    # Generate text using the LLM model and the vector store
    generated_text = generate_text_with_llm( topic)
    st.write(generated_text)
    
    #Download button to download the generated response as ".txt" file
    st.download_button("Download generated MCQs", generated_text)


# Streamlit UI
st.title("LLM Multiple Choice Questions Generator App")

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    # If a PDF is uploaded, process it and create the vector store
    load_pdf_and_create_vector_store(pdf_file)

# Generate Text from LLM
topic = st.text_input("Enter the topic for the LLM to generate MCQs")

if topic:
    result = generate_text_from_llm(topic)
    if result:
        st.write(result)

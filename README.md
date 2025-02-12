# LLM_ollama_llama_langchain_MCQ_generator_app
## LLM Used-🦙 llama3.2:1b 🤖

## Overview
The **MCQ Generator App** is a MCQ generator app powered by a fine-tuned large language model (LLM) known as *Llama3.2:1b* pulled from *OLLAMA*. This app takes a pdf file from the user and generates MCQs based on the **text extracted** from the pdf  and the **topic** entered by the user.

## Process Flowchart
- The uploaded pdf is split into chunks , converted to embeddings using the *nomic-embed-text* embedding pulled from OLLAMA and stored to the **FAISS DB**(Vector database) for semantic search.
![data_ingestion](https://github.com/user-attachments/assets/45f4d42b-74ae-4ee1-847d-eeb07fbb5fac)
- **RAG-Retrieval Augmented Generation**: The question or topic from the user is converted into embeddings, fed to the vectorstore for semantic search, after that relevant chunks of documents are retrieved and fed to the LLM along with the user question or topic, to finally generate MCQS.
![retrieval_and_generation](https://github.com/user-attachments/assets/783ce298-1f05-4126-b91d-417931dbda10)

## 🚀 Features

- **MCQ Generation:** Runs locally and allows users to generate MCQs based on PDF data.
- **Download:** Can download the generated MCQs as **.txt** file.

## Development Specs
- Utilizes [llama3.2:1b](https://ollama.com/library/llama3.2:1b) and [embeddings](https://ollama.com/library/nomic-embed-text) for robust functionality.
- Developed using [Langchain](https://github.com/langchain-ai/langchain) and [Streamlit](https://github.com/streamlit/streamlit) technologies for enhanced performance.


## 🛠️ Installation
1. **Clone This Repository:**
 ```bash
   git clone [https://github.com/Kav1n-Lal/LLM_ollama_llama_langchain_MCQ_generator_app.git]
   ```
2. **Create a conda environment and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Download the Ollama and llama3.2:1b Model:

- Download Ollama for Windows and install it-[https://ollama.com/download]
- **On the command prompt type**:
- **ollama** to see whether it is running.
- Then type **ollama serve**, to view the localhost id.
- Now type **ollama run llama3.2:1b** to pull the LLM. You can use it like **chatGPT** on the terminal itself. To quit type **/bye**.
- Then type **ollama pull nomic-embed-text** to pull the embeddings respectively.
- **ollama list** to check whether the LLM and the embedding is pulled successfully. It displays the information.

## 📝 Usage

1. **Run the Application:**
   ```bash
   streamlit run final_mcq_app.py
   ```
2. **Access the Application:**
   - Once the application is running, access it through the provided URL.
     
## System Requirements
- **CPU:** Intel® Core™ i5 or equivalent.
- **RAM:** 8 GB.
- **Disk Space:** 7 GB.
- **Hardware:** Operates on CPU; no GPU required.

## 🤖 How to Use
- Copy the cloned repository path and on lines 35 and 87 in final_mcq_app.py enter the **file paths of the cloned folder and the vectorstore**  before running the code.
- Upon running the application, you'll be presented with a box to upload your pdf file, upload the file and after successfull storing of the text into the vector database, enter the topic from the pdf file to generate MCQs.
- After successfull MCQ generation, check the generated MCQs and click on the **Download generated MCQs** button to download the MCQs as **.txt** file.

## 📷 Screenshots
- screenshot 1![mcq_1](https://github.com/user-attachments/assets/ba80a96d-b3f1-4906-bf5b-48a62701ba7f)
- screenshot 2 ![mcq_2](https://github.com/user-attachments/assets/9f210a3b-2e58-410f-9aec-83cc1bc76045)
- screenshot 3 ![mcq_3](https://github.com/user-attachments/assets/369f94f9-cea1-4a8f-9241-dc8d82de352a)
- screenshot 4![mcq_4](https://github.com/user-attachments/assets/7cac003a-6213-4e0a-84e2-3e6afa67ee97)




import os
import numpy as np
import faiss
import openai
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langdetect import detect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API Key
#os.environ["OPENAI_API_KEY"] = "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"  # Replace with your key
openai.api_key = os.getenv("OPENAI_API_KEY")
# Function to transcribe audio file
def transcribe_audio(audio_file):
    try:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        # If the API returns a string, return it directly.
        return transcript  
    except Exception as e:
        return f"An error occurred: {e}"

# Function to process transcript and prepare the FAISS index
def process_transcript(transcript):
    # Save transcript to file
    transcript_path = "transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as file:
        file.write(transcript)
    
    # Load the text file
    loader = TextLoader(transcript_path, encoding="utf-8")
    documents = loader.load()

    # Preprocess and clean the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    processed_docs = []

    for doc in documents:
        cleaned_text = doc.page_content.strip().replace("\n", " ")
        processed_docs.extend(text_splitter.split_text(cleaned_text))

    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS index
    doc_embeddings = embeddings.embed_documents(processed_docs)
    embedding_matrix = np.array(doc_embeddings).astype('float32')
    dim = embedding_matrix.shape[1]

    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embedding_matrix)

    return faiss_index, processed_docs, embeddings

# Set up the language model (LLM)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Streamlit app layout
st.title("Audio Transcription and Q&A")
st.write("Upload an audio file (.mp3), and I'll transcribe it and help answer questions based on the transcription.")

# Upload audio file
audio_file = st.file_uploader("Choose an audio file", type=["mp3"])

if audio_file:
    transcript = transcribe_audio(audio_file)
    
    # Ensure we have a valid transcript (not an error message)
    if transcript and not transcript.startswith("An error occurred"):
        # Process transcript for embedding and indexing
        faiss_index, processed_docs, embeddings = process_transcript(transcript)
        
        # User input for query
        query = st.text_input("Ask a question:")
        
        if query:
            # Check for greetings
            if query.lower() in ["hi", "hlo", "hii", "hello"]:
                st.write("Hello there! How can I help you today?")
            else:
                # Detect the language of the input query
                try:
                    query_lang = detect(query)
                except:
                    query_lang = "en"

                # Query processing
                query_embedding = np.array([embeddings.embed_query(query)]).astype('float32')
                k = min(5, len(processed_docs))
                distances, indices = faiss_index.search(query_embedding, k)

                retrieved_docs = [processed_docs[idx] for idx in indices[0] if idx < len(processed_docs)]

                min_distance = np.min(distances[0]) if distances[0].size > 0 else float('inf')
                threshold = 0.75

                if min_distance > threshold:
                    st.write("Data not present in the provided documents.")
                else:
                    language_instruction = (
                        "Respond concisely and accurately in English."
                        if query_lang == "en" else
                        "日本語で簡潔かつ正確に回答してください。提供された文書からのみ情報を抽出し、"
                        "余分な情報や外部の知識を使用せず、質問に関連する最も重要な情報だけを提供してください。"
                        "冗長な表現は避けてください。"
                    )

                    prompt = (
                        "You are an AI language model that provides responses strictly based on the provided document. "
                        "Do not generate information beyond the given excerpts. If the answer is not found in the provided text, "
                        "respond with: 'The requested information is not available in the provided document.'\n\n"
                        f"I retrieved the following relevant information:\n\n"
                        f"Query: {query}\n"
                        f"Relevant Excerpts:\n" + "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs)]) +
                        "\n\nYour task is to:\n"
                        "1. Carefully analyze the provided excerpts and extract only the most relevant single-line response.\n"
                        "2. Answer concisely in *one sentence only* using only the retrieved excerpts.\n"
                        "3. If the information is unclear or missing, explicitly state: 'The requested information is not available in the provided document.'\n\n"
                        f"{language_instruction}\n"
                        "Provide a single-sentence response based strictly on the retrieved excerpts."
                    )

                    final_answer = llm.invoke(prompt)
                    st.write(f"Answer: {final_answer.content.strip()}")
    else:
        st.write(transcript)

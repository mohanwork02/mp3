import os
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langdetect import detect
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your key

# Step 1: Accept the audio file path
audio_path = input("Human: Provide audio file path (.mp3): ").strip()

# Step 2: Transcribe the audio using OpenAI Whisper
print("Machine: Transcribing audio with OpenAI Whisper...")

try:
    with open(audio_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    # Save the transcript to a file
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

    # Set up the language model (LLM)
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Chatbot loop
    while True:
        query = input("\nAsk me anything or type 'exit' to quit: ").strip().lower()
        if query in ["hi", "hlo", "hii","hello"]:
            print("\nHow can I assist today?")
            continue

        if query == 'exit':
            print("\nGoodbye!")
            break

        # Detect the language of the input query
        try:
            query_lang = detect(query)
        except:
            query_lang = "en"

        query_embedding = np.array([embeddings.embed_query(query)]).astype('float32')
        k = min(5, len(processed_docs))
        distances, indices = faiss_index.search(query_embedding, k)

        retrieved_docs = [processed_docs[idx] for idx in indices[0] if idx < len(processed_docs)]

        min_distance = np.min(distances[0]) if distances[0].size > 0 else float('inf')
        threshold = 0.75

        if min_distance > threshold:
            print("\nData not present in the provided documents.")
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
            print("\nAnswer:", final_answer.content.strip())

except Exception as e:
    print(f"An error occurred: {e}")

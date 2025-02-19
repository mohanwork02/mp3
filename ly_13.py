import os
import numpy as np
import faiss
import openai
import ffmpeg
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langdetect import detect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = ""  # Replace with your actual key

# Step 1: Accept video file path
video_path = input("Human: Provide video file path (.mp4): ").strip()

# Step 2: Extract audio from the video
print("Machine: Extracting audio from the video...")
audio_path = "temp_audio.wav"

try:
    ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run(overwrite_output=True)
    print(f"Machine: Audio saved as {audio_path}")
except Exception as e:
    print(f"Error extracting audio: {e}")
    exit()

# Step 3: Transcribe the audio using OpenAI Whisper
print("Machine: Transcribing audio with OpenAI Whisper...")

try:
    with open(audio_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    
    # Save transcript
    transcript_path = "transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as file:
        file.write(transcript)

    print(f"Machine: Transcript saved as {transcript_path}")

    # Load text file
    loader = TextLoader(transcript_path, encoding="utf-8")
    documents = loader.load()

    # Store full transcript for full-text queries
    full_transcript = " ".join([doc.page_content.strip().replace("\n", " ") for doc in documents])

    # Preprocess text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    processed_docs = text_splitter.split_text(full_transcript)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS index
    doc_embeddings = embeddings.embed_documents(processed_docs)
    embedding_matrix = np.array(doc_embeddings).astype('float32')
    dim = embedding_matrix.shape[1]

    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embedding_matrix)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4")

    # Full-text keywords
    full_text_keywords = [
        "all lyrics", "full text", "complete lyrics", "entire text", "all transcript text", "full lyrics","full transcript","full video","video content"
        "total speech", "total voice", "total lyrics", "total transcript text", "total transcript", "all text","print lyrics","full content"
        "full speech", "full song", "complete song", "entire song", "full voice", "entire lyrics", "entire voice","complete transcript",
        "total song", "total transcript", "すべての歌詞", "全文", "完全な歌詞", "全体のテキスト", "すべてのトランスクリプトテキスト", "完全な歌詞",
        "全スピーチ", "全音声", "全歌詞", "全トランスクリプトテキスト", "全トランスクリプト", "すべてのテキスト",
        "全スピーチ", "完全な曲", "完全な曲", "全体の曲", "完全な音声", "全体の歌詞", "全体の音声", "全曲",
        "全トランスクリプト"
    ]

    # Chatbot loop
    while True:
        query = input("\nAsk me anything or type 'exit' to quit: ").strip().lower()

        if query in ["hi", "hlo", "hii", "hello"]:
            print("\nHow can I assist today?")
            continue

        if query == "exit":
            print("\nGoodbye!")
            break

        # Detect language
        try:
            query_lang = detect(query)
        except:
            query_lang = "en"

        # Full-text query check
        if any(keyword in query for keyword in full_text_keywords):
            print("\nHere is the full text from the transcription:\n")
            print(full_transcript)
            continue

        # Perform semantic search
        query_embedding = np.array([embeddings.embed_query(query)]).astype("float32")
        k = min(5, len(processed_docs))
        distances, indices = faiss_index.search(query_embedding, k)

        retrieved_docs = [processed_docs[idx] for idx in indices[0] if idx < len(processed_docs)]

        # Check if relevant data is found
        min_distance = np.min(distances[0]) if distances[0].size > 0 else float("inf")
        threshold = 0.75  # Adjust if needed

        if min_distance > threshold or not retrieved_docs:
            print("\nThe requested information is not available in the provided document.")
            continue  # Skip to the next query

        # Set language response mode
        language_instruction = (
            "Respond concisely and accurately in English."
            if query_lang == "en" else
            "日本語で簡潔かつ正確に回答してください。提供された文書からのみ情報を抽出し、"
            "余分な情報や外部の知識を使用せず、質問に関連する最も重要な情報だけを提供してください。"
            "冗長な表現は避けてください。"
        )

        # Create prompt
        prompt = (
            "You are an AI assistant providing responses strictly from the provided document. "
            "Do not generate information beyond the given excerpts. If the answer is not found, "
            "respond with: 'The requested information is not available in the provided document.'\n\n"
            f"Query: {query}\n"
            f"Relevant Excerpts:\n" + "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs)]) +
            "\n\nYour task is to:\n"
            "1. Extract the most relevant single-line response from the excerpts.\n"
            "2. Answer concisely in *one sentence only* using only the retrieved excerpts.\n"
            "3. If unclear or missing, state: 'The requested information is not available in the provided document.'\n\n"
            f"{language_instruction}\n"
            "Provide a single-sentence response based strictly on the retrieved excerpts."
        )

        final_answer = llm.invoke(prompt).content.strip()

        # Prevent hallucinated responses
        if "The requested information is not available" in final_answer:
            print("\nThe requested information is not available in the provided document.")
        else:
            print("\nAnswer:", final_answer)

except Exception as e:
    print(f"An error occurred: {e}")

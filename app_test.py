import http.client
import json
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
import re
import asyncio
import nest_asyncio
import warnings
from langchain_groq import ChatGroq
from gtts import gTTS
import base64
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder

warnings.filterwarnings('ignore', category=UserWarning, module='torch')
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nest_asyncio.apply()

def init_async():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

def init_groq_model():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return ChatGroq(
        groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.2
    )

llm_groq = init_groq_model()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vectorstore

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    llm = llm_groq
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain

def extract_job_features(text):
    skills = re.findall(r'\b(Java|Python|Data Science|Machine Learning|Deep Learning|Software Engineer|Data Engineer|AI|NLP|C\+\+|SQL|TensorFlow|Keras)\b', text, re.IGNORECASE)
    titles = re.findall(r'\b(Engineer|Data Scientist|Developer|Manager|Analyst|Consultant)\b', text, re.IGNORECASE)
    features = list(set(skills + titles))
    return features if features else ["General"]

def get_job_recommendations(features):
    host = "jooble.org"
    jooble_api_key = os.getenv("JOOBLE_API_KEY")

    connection = http.client.HTTPConnection(host)
    headers = {"Content-type": "application/json"}
    keywords = ", ".join(features)
    body = json.dumps({"keywords": keywords, "location": "Remote"})
    
    try:
        connection.request("POST", f"/api/{jooble_api_key}", body, headers)
        response = connection.getresponse()
        data = response.read()
        jobs = json.loads(data).get("jobs", [])
        
        job_listings = []
        for job in jobs:
            job_listings.append({
                "title": job.get("title", "Job Title"),
                "company": job.get("company", "Company Name"),
                "link": job.get("link", "#"),
                "description": clean_job_description(job.get("snippet", "Job description not available."))
            })
        return job_listings
    except Exception as e:
        st.error(f"Error fetching job data: {e}")
        return []

def clean_job_description(description):
    description = re.sub(r'&nbsp;|&#39;|<[^>]+>', '', description)
    relevant_info = re.findall(r'\b(?:Python|Java|TensorFlow|Keras|Machine Learning|AI|NLP|Deep Learning|Engineer|Data Scientist|Developer|Analyst)\b', description, re.IGNORECASE)
    for word in relevant_info:
        description = re.sub(r'\b' + re.escape(word) + r'\b', f"**{word}**", description)
    return description

def handle_userinput(user_question):
    if user_question:
        try:
            response = st.session_state.conversation.invoke({
                "question": user_question
            })
            answer = response.get('answer')
            st.write(answer)

            # Convert response to speech
            tts = gTTS(text=answer, lang='en')
            tts.save("response.mp3")
            with open("response.mp3", "rb") as audio_file:
                audio_bytes = audio_file.read()
                b64 = base64.b64encode(audio_bytes).decode()

            st.markdown(
                f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")

def main():
    st.set_page_config(page_title="Job Assistant Chatbot", page_icon=":briefcase:")
    st.header("Job Assistant Chatbot 💬")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    try:
        init_async()
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "job_recommendations" not in st.session_state:
            st.session_state.job_recommendations = []

        tab_choice = st.sidebar.radio("Choose a tab", ["Chatbot", "Job Recommendations"])

        if tab_choice == "Chatbot":
            st.header("Chat with Job Assistant 🤖")

            user_question = st.text_input("Type your question about your resume:")

            # Voice input
            st.write("🎤 Or use your voice:")
            audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🛑 Stop", key="voice")

            if audio:
                st.audio(audio['bytes'], format='audio/wav')
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio['filename']) as source:
                    audio_data = recognizer.record(source)
                    try:
                        voice_question = recognizer.recognize_google(audio_data)
                        st.success(f"You said: {voice_question}")
                        user_question = voice_question
                    except sr.UnknownValueError:
                        st.error("Sorry, could not understand your voice.")
                    except sr.RequestError:
                        st.error("Speech Recognition API error.")

            if user_question:
                handle_userinput(user_question)

            st.sidebar.subheader("Your documents")
            pdf_docs = st.sidebar.file_uploader("Upload your resumes in PDF form and click 'Process'", accept_multiple_files=True)
            if st.sidebar.button("Process"):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        try:
                            raw_text = get_pdf_text(pdf_docs)
                            text_chunks = get_text_chunks(raw_text)
                            vectorstore = get_vectorstore(text_chunks)
                            st.session_state.conversation = get_conversation_chain(vectorstore)

                            job_features = extract_job_features(raw_text)
                            st.session_state.job_recommendations = get_job_recommendations(job_features)
                            st.success("Documents processed and job recommendations updated.")
                        except Exception as e:
                            st.error(f"Error processing documents: {e}")
                else:
                    st.warning("Please upload PDFs before processing.")

        elif tab_choice == "Job Recommendations":
            st.header("Recommended Jobs 💼")
            if st.session_state.job_recommendations:
                for job in st.session_state.job_recommendations:
                    st.markdown(f"**[{job['title']}]({job['link']})** at **{job['company']}**")
                    st.markdown(f"**Description:** {job['description']}", unsafe_allow_html=True)
            else:
                st.info("Upload and process your resume in the Chatbot tab to view job recommendations.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == '__main__':
    main()

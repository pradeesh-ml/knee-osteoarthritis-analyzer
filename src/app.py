import streamlit as st
import torch
from torchvision import models, transforms
from pytorch_grad_cam import GradCAMPlusPlus 
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
import numpy as np
import plotly.express as px 
from PIL import Image
import os

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Knee Osteoarthritis Analyzer", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = 'models\best_model.pth'
DB_DIR =  'db'
PERSISTENT_DIRECTORY = os.path.join(DB_DIR, 'chroma_db_koa')

CLASSES = ['Normal (Grade 0)', 'Doubtful (Grade 1)', 'Mild (Grade 2)', 'Moderate (Grade 3)', 'Severe (Grade 4)']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource 
def load_pytorch_model():
    model = models.resnet18(pretrained=False) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure 'best_model.pth' is in the same directory as the app.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    model = model.to(DEVICE)
    model.eval()
    return model

PYTORCH_MODEL = load_pytorch_model()

def preprocess_image(image_pil):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image_pil).unsqueeze(0).to(DEVICE)

def get_gradcam_plusplus(model, input_tensor, original_pil_image):
    if model is None:
        return None, "Model not loaded"

    rgb_img_resized = np.array(original_pil_image.resize((224, 224))) / 255.0
    target_layers = [model.layer4[-1]]

    try:
        gradcam_pp = GradCAMPlusPlus(model=model, target_layers=target_layers)
        grayscale_cam_pp = gradcam_pp(input_tensor=input_tensor, targets=None)[0]
        cam_image_pp = show_cam_on_image(rgb_img_resized, grayscale_cam_pp, use_rgb=True)
        return cam_image_pp, None
    except Exception as e:
        return None, f"Error generating GradCAM++: {e}"

@st.cache_resource 
def get_llm_and_embeddings():
    llm = OllamaLLM(model='llama3.2:3b')
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    return llm, embeddings

LLM, EMBEDDINGS = get_llm_and_embeddings()

@st.cache_resource 
def get_retriever(_embeddings, _persistent_dir):
    if not os.path.exists(_persistent_dir) or not os.listdir(_persistent_dir):
        st.warning("Chatbot retriever cannot be initialized: Vector store is not ready or empty.")
        return None
    try:
        db = Chroma(persist_directory=_persistent_dir, embedding_function=_embeddings)
        return db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error initializing retriever: {e}")
        return None

RETRIEVER = get_retriever(EMBEDDINGS, PERSISTENT_DIRECTORY)

def get_qa_chain(llm, retriever):
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
    
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    If the follow up question is already a standalone question or if the chat history is empty, DO NOT rephrase the question and just return the original follow up question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    QA_PROMPT_TEMPLATE = """You are a helpful AI assistant specializing in Knee Osteoarthritis (KOA).
    Your goal is to answer questions about KOA using *only* the provided context.
    If the answer is not in the context, simply say that you don't know.
    Do not mention the context explicitly in your responses.
    Avoid phrases like 'Based on the provided context' or similar.
    Keep your answers concise, clear, and informative.

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""
    QA_PROMPT = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)

    if retriever is None:
        return None

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.chat_memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False
    )

st.title("🔬 Knee Osteoarthritis Analysis & Chatbot 🤖")
st.markdown("""
Upload a knee X-ray image to get a predicted osteoarthritis grade and see the GradCAM++ visualization.
You can also ask questions about Knee Osteoarthritis to our specialized chatbot.
""")

st.header("X-Ray Image Analysis")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.session_state["uploaded_pil_image"] = pil_image  

    col1_img, col2_img = st.columns(2)


    if PYTORCH_MODEL is None:
        st.error("Image analysis model could not be loaded. Please check the logs.")
    else:
        if st.button("Analyze X-Ray Image", key="analyze_button"):
            with st.spinner("Analyzing image..."):
                input_tensor = preprocess_image(pil_image)
                st.session_state["input_tensor"] = input_tensor  


                with torch.no_grad():
                    outputs = PYTORCH_MODEL(input_tensor)
                    _, predicted_idx = torch.max(outputs, 1)
                    prediction_class = CLASSES[predicted_idx.item()]
                    st.session_state["prediction_class"] = prediction_class
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()

                    st.session_state["class_probabilities"] = probabilities

                    fig = px.bar(
                        x=CLASSES,
                        y=probabilities,
                        labels={'x': "KOA Grade", 'y': "Probability"},
                        title="Grade Classification Probabilities",
                        color=probabilities,
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                cam_image_pp, error_msg_cam = get_gradcam_plusplus(PYTORCH_MODEL, input_tensor, pil_image)
                if error_msg_cam:
                    st.error(error_msg_cam)
                elif cam_image_pp is not None:
                    st.session_state["gradcam_image"] = cam_image_pp
                else:
                    st.warning("GradCAM++ could not be generated.")

if "uploaded_pil_image" in st.session_state:
    col1_img, col2_img = st.columns(2)
    
    with col1_img:
        st.image(st.session_state["uploaded_pil_image"], caption="Uploaded X-Ray", use_container_width=True)

    with col2_img:
        if "gradcam_image" in st.session_state:
            st.image(st.session_state["gradcam_image"], caption="GradCAM++ Visualization", use_container_width=True)
        else:
            st.info("GradCAM++ not available. Please click 'Analyze X-Ray Image'.")

if "prediction_class" in st.session_state:
    st.subheader(f"Predicted Grade: {st.session_state['prediction_class']}")

st.header("KOA Chatbot")

if RETRIEVER is None:
    st.warning("Chatbot is not fully initialized. Context might be missing or functionality limited. Check for errors above.")
else:

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with Knee Osteoarthritis today?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about Knee Osteoarthritis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        qa_chain = get_qa_chain(LLM, RETRIEVER) 
        
        if qa_chain is None:
            st.error("Chatbot chain could not be initialized. Cannot process query.")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I'm having trouble initializing. Please check the application setup."})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = qa_chain.invoke({"question": prompt}) 
                        response = result["answer"]
                    except Exception as e:
                        response = f"Sorry, an error occurred: {e}"
                        st.error(response)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. How can I help you?"}]
        if "chat_memory" in st.session_state:
            st.session_state.chat_memory.clear()
        st.rerun()

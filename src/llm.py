import os
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

curr_dir=os.getcwd()
db_dir=os.path.join(curr_dir,'db')
persistent_directory=os.path.join(db_dir,'chroma_db_koa')

model=OllamaLLM(model='llama3.2:3b')
embedding=OllamaEmbeddings(model='nomic-embed-text')

def create_vector_embedding(db_dir,persistent_directory):
    if not os.path.exists(persistent_directory):
        text_files=[f for f in os.listdir(db_dir) if f.endswith('.txt')]
        all_loaded_documents=[]
        for files in text_files:
            file_path=os.path.join(db_dir,files)
            loader=TextLoader(file_path,encoding="utf-8")
            docs=loader.load()
            for doc in docs:
                doc.metadata={'source': files}
                all_loaded_documents.append(doc)
            print(f'Loaded {len(docs)} documents from {files}')
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        docs=text_splitter.split_documents(all_loaded_documents)
        print('-----Document chunk information-----\n')
        print(f'Number of documnet chunks: {len(docs)}')
        print(f"Creating and persisting Chroma DB to '{persistent_directory}'...")
        try:
            db=Chroma.from_documents(documents=docs,embedding=embedding,persist_directory=persistent_directory)
            print('Successfully created and stored vector embedding')
        except Exception as e:
            print(f'Error while creating vector store: {e}')
    else:
        print('Vector embedding is already exists')

create_vector_embedding(db_dir,persistent_directory)

db=Chroma(persist_directory=persistent_directory,embedding_function=embedding)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

memory = ConversationBufferMemory(
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
Your goal is to answer questions about KOA based *only* on the provided context.
If the context does not contain the answer to the question, state clearly that you cannot answer based on the provided information.
Do not make up information or answer from your general knowledge if it's not in the context.
Be concise and informative.

Context:
{context}

Question: {question}

Helpful Answer:"""
QA_PROMPT = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=CONDENSE_QUESTION_PROMPT, 
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},   
    return_source_documents=True, 
    verbose=False 
)

print("\n--- Knee Osteoarthritis Chatbot ---")
print("Ask me anything about Knee Osteoarthritis.")
print("Type 'quit', 'exit', or 'bye' to end the conversation.")

while True:
    try:
        user_query = input("\nYou: ")
        if user_query.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye! Stay healthy.")
            break
        if not user_query.strip():
            continue

        print("Bot: Thinking...")
    
        result = qa_chain.invoke({"question": user_query})
        answer = result["answer"]

        print(f"Bot: {answer}")

        # Optional: Print source documents for debugging or transparency
        # print("\n--- Sources ---")
        # for doc in result.get("source_documents", []):
        #     print(f"- {doc.metadata.get('source', 'Unknown')}, content snippet: {doc.page_content[:100]}...")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Optionally, you might want to break or offer to reset
        # For now, we'll just let the loop continue
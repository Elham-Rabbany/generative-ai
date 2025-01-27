
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# llm
hf_model = 'mistralai/Mistral-7B-Instruct-v0.3' # 'microsoft/Phi-3.5-mini-instruct'
llm = HuggingFaceEndpoint(repo_id=hf_model)

# embeddings
embedding_model = 'sentence-transformers/all-MiniLM-l6-v2'
embeddings_folder = 'data/cache/'

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model
    , cache_folder=embeddings_folder)

# load Vector Database
# allow_dangerous_deserialization is needed. Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on your machine
vector_db = FAISS.load_local('data/CIA_faiss_index', embeddings, allow_dangerous_deserialization=True)

# retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# prompt
<<<<<<< HEAD
template = template = """You are a nice chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. 
Keep your answers short and succinct.

=======
template = """You are a nice chatbot having a conversation with a human.
Answer the question based only on the following context and previous conversation.
Keep your answers short, succinct, informative, and clear, so that the couterpart can learn from you.
>>>>>>> 978ddff722f231e19c0020d6978b7f418c31ae74

Previous conversation:
{chat_history}

Context:
{context}

Question: {input}
Response:"""

prompt = ChatPromptTemplate.from_messages([
    ('system', template),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}'),
])

# bot with memory
@st.cache_resource
def init_bot():
    doc_retriever = create_history_aware_retriever(llm, retriever, prompt)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(doc_retriever, doc_chain)

rag_bot = init_bot()


##### streamlit #####

import streamlit as st

# Title
st.title("CIA World Factbook 2023-2024")

# Subtitle
st.subheader("Your Gateway to Global Information")

# Introduction Text
st.markdown("""
Welcome to the **CIA World Factbook Explorer**! This app allows you to explore detailed information about countries worldwide, 
including demographics, geography, economy, and more.


**Start exploring now and unlock the power of global knowledge!**
""")

# Add an Image or Logo
st.image("Photo/CIA World Factbook-2023-2024.jpg", caption="CIA World Factbook 2023-2024", use_column_width=True)


# Footer with a Call to Action
st.markdown("""
---
**Explore the World. Discover the Data. Shape the Future.**
# """)
# # Resize and Enhance Image
# def process_image(input_path, output_path, width):
#     # Open the image
#     image = Image.open(input_path)
    
#     # Resize the image
#     aspect_ratio = image.height / image.width
#     new_height = int(width * aspect_ratio)
#     resized_image = image.resize((width, new_height))
    
#     # Save the resized image
#     resized_image.save(output_path)
#     return output_path

# # Paths for the image
# input_path = "Photo/CIA World Factbook.jpg"
# output_path = "Photo/CIA_World_Factbook_Resized.jpg"
# processed_image_path = process_image(input_path, output_path, width=800)  # Set width to 800 pixels

# # Streamlit App
# st.title("🌍 CIA World Factbook 2023-2024")
# st.image(processed_image_path, caption="CIA World Factbook 2023-2024")




# Initialise chat history
# Chat history saves the previous messages to be displayed
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input('Curious minds wanted!'):

    # Display user message in chat message container
    st.chat_message('human').markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({'role': 'human', 'content': prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner('Thinking...'):

        # send question to chain to get answer
        answer = rag_bot.invoke({'input': prompt, 'chat_history': st.session_state.messages, 'context': retriever})

        # extract answer from dictionary returned by chain
        response = answer.get('answer', 'Sorry, I couldn’t find an answer.')

        # Display chatbot response in chat message container
        with st.chat_message('assistant'):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({'role': 'assistant', 'content':  response})

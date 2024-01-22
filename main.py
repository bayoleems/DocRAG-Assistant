from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import streamlit as st
import os
import time

load_dotenv()
st.set_page_config(page_title="DocRAG Assistant💬")
openai_api_key = os.environ["OPENAI_API_KEY"]
def document_preprocess(file, doc_type):
    file_path = os.path.join(os.getcwd(), file.name)

    # Save the uploaded file to disk
    with open(file_path, "wb") as f:
        f.write(file.getvalue())
    
    if doc_type == '.DOCX':
        loader = Docx2txtLoader(file_path)
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=150, length_function=len
        )
        documents = text_splitter.split_documents(raw_documents)
        embeddings_model = OpenAIEmbeddings()
        os.remove(file_path)
        return documents, embeddings_model
    if doc_type == '.PDF':
        loader = PyPDFLoader(file_path)
        raw_documents = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=150, length_function=len
        )
        documents = text_splitter.split_documents(raw_documents)
        embeddings_model = OpenAIEmbeddings()
        os.remove(file_path)
        return documents, embeddings_model
    


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today ?🙂"}]

st.title('DocRAG Assistant💬')
with st.sidebar:
    st.markdown("""**DocRAG Assistant** 
                engages users in intuitive conversations to gather information about document content, milestones, and potential challenges.\nWhether you're managing project reports, compliance documents, or any other textual information""")          
    st.markdown("👨🏾‍💻 About the author [here](https://www.linkedin.com/in/saleem-adebayo-82b512138/)!")
    
    st.subheader('Document parameters')
    doc_type =st.selectbox('*Document Extension*', ['.DOCX','.PDF'])
    file_path = st.file_uploader('*Upload document*')
    if file_path and doc_type:
        documents, embeddings_model = document_preprocess(file_path, doc_type)
        db = FAISS.from_documents(documents, embeddings_model)
        retriever = db.as_retriever()
        llm = ChatOpenAI(temperature=1, model="gpt-4")
        qa_chain = create_qa_with_sources_chain(llm)
        retrieval_qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever,
            return_source_documents=True,
        )
    

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today ?🙂"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def run_query(prompt):
    output = retrieval_qa({
        "question": f"{prompt}",
        "chat_history": []
    })
    return output


if prompt := st.chat_input('Type here..'):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner('loading...'):
            response = run_query(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response['answer']:
                full_response += item
                placeholder.markdown(full_response + "|")
            placeholder.markdown(response['answer'])
            # full_response = ''
            # stream_response = []
            # for item in response['answer']:
            #     stream_response.append(item)
            #     full_response += item
            #     result = "".join(stream_response)
            #     time.sleep(0.001)
            #     placeholder.markdown(f'{result} ▌')
            # placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

# print(f"Source: {output['source_documents'][0].metadata['source']}")
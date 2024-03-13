from openai import OpenAI
import streamlit as st
import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import langchain
langchain.verbose = True

st.title("RAG - Product Reviews")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def pretty_print(text, words_per_line = 150):
  words = text.split()
  for i in range(0, len(words), words_per_line):
    line = ' '.join(words[i:i+words_per_line])
    st.write(line)
    return [line]

# RAG Setup
# Importing the dataset
pd.set_option("display.max_colwidth", None)
file_name = "./documents/customer_review.csv"
df = pd.read_csv(file_name)

loader = CSVLoader(file_path=file_name)
docs = loader.load()

chunk_size = 128
chunk_overlap = 32

r_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap,
    length_function = len,
    add_start_index = True
)
pages = r_text_splitter.split_documents(docs)

# Creating the Vector DB
embedding = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
persist_directory = 'persist_chroma'

vectordb = Chroma.from_documents(
    documents = pages,
    embedding = embedding,
    persist_directory = persist_directory
)

llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model=llm_name, temperature=0)

qa_chain_default = RetrievalQA.from_chain_type(
    llm,
    retriever = vectordb.as_retriever(search_kwargs={"k":3}),
    chain_type="stuff",
    return_source_documents=True,
)

# Streamlit Application
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = llm_name

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = qa_chain_default({"query": prompt})
        response = pretty_print(stream.get("result"))
    st.session_state.messages.append({"role": "assistant", "content": response})
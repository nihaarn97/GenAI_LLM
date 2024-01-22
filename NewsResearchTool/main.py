import streamlit as st
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

st.title("News Research Tool 📰📈")

st.sidebar.title("Enter News Article URLs")
links = []

for i in range(4):
    url = st.sidebar.text_input(f"News URL : {i+1}")
    links.append(url)

process_click = st.sidebar.button("Process Articles")

main_placeholder = st.empty()

gpt_llm = OpenAI(temperature=0.0, max_tokens=700)
embeddings = OpenAIEmbeddings()

if process_click:
    loaders = UnstructuredURLLoader(urls=links)
    main_placeholder.text("Data Loading Started... ✅")
    data = loaders.load()
    text_splitter = RecursiveCharacterTextSplitter(separators = ['\n\n', '\n', '.', ','], chunk_size=1000)
    main_placeholder.text("Text Splitter Started... ✅")
    docs = text_splitter.split_documents(data)
    main_placeholder.text("Building Embedding Vector Store... ✅")
    vector_index = FAISS.from_documents(docs, embeddings)
    vector_index.save_local("nrt_index")
    main_placeholder.text("Processing Done... ✅✅✅")

search_query = main_placeholder.text_input("What would you like to know?")

if search_query:
    vector_index = FAISS.load_local("nrt_index", embeddings)
    question_chain = RetrievalQAWithSourcesChain.from_llm(llm = gpt_llm,
                                                          retriever = vector_index.as_retriever())
    res = question_chain({"question": search_query}, return_only_outputs = True)
    st.header("Answer")
    st.write(res["answer"])

    # Display the source if available
    sources = res.get("sources", "")
    if sources:
        st.subheader("Answer Source...")
        sources_list = sources.split("\n")
        for val in sources_list:
            st.write(val)






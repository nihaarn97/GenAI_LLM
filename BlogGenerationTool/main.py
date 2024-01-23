import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


def get_llm_response(text, words, style):
    llm_model = CTransformers(model = "models/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type = "llama",
                        config = {"max_new_tokens": 512,
                                  "temperature": 0.05})

    blog_template = f"Write a {words} word blog for an audience with {style} understanding of the topic {text}. Adhere to the word limit strictly."

    blog_prompt = PromptTemplate(input_variables = ['words', 'style', 'text'],
                                 template = blog_template)

    response = llm_model(blog_prompt.format(words = words, style = style, text = text))
    return response


st.set_page_config(page_title = "Blog Generator",
                   page_icon = "📖",
                   layout = "centered",
                   initial_sidebar_state = "collapsed")

st.header("📖 Generate Natural Blogs For Your Website")

input_text = st.text_input("Enter your blog topic")

col1, col2 = st.columns([5,5])

with col1:
    words = st.text_input("Words in blog post")
with col2:
    style = st.selectbox("Type of audience",
                         ('General', 'Intermediate', 'Advanced'),
                         index = 0)

submit = st.button("Generate Blog Post")

if submit:
    st.write(get_llm_response(input_text, words, style))


import streamlit as st
import langchain_helper

st.title("Restaurant Idea Generator")

cuisine = st.sidebar.selectbox(
                            "Pick Cuisine For Your Restaurant",
                            ("Indian", "American", "Mexican", "Italian", "French", "German", "Greek", "Turkish",
                             "Thai", "Japanese", "Chinese"))

total = st.sidebar.selectbox(
                            "Pick Number Of Dishes",
                            ("5", "10", "15", "20"))

course = st.sidebar.selectbox(
                            "Pick The Course",
                            ("Starter", "Main Course", "Dessert"))

if cuisine and total and course:
    response = langchain_helper.generate_idea(cuisine, total, course)
    st.header(response["restaurant_name"].strip())
    menu_items = response["menu_items"].strip().split("\n")
    st.write(f"**{course} menu items for restaurant idea**")
    for item in menu_items:
        st.write("-", item)

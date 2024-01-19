import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# read local .env file
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

llm_model = OpenAI(temperature = 0.0)
# temperature determines how creative the model can get with 0 being the least and 1 being the most
# Setting a high temperature also leads to wrong answer generation

def generate_idea(cuisine, total, course):
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this. Only one name will do.")

    name_chain = LLMChain(llm=llm_model, prompt=prompt_template_name, output_key="restaurant_name")

    prompt_template_menu = PromptTemplate(
        input_variables=['total', 'course', 'restaurant_name'],
        template="Suggest {total} {course} menu items for {restaurant_name}. Return it as a comma separated list.")

    menu_chain = LLMChain(llm=llm_model, prompt=prompt_template_menu, output_key="menu_items")

    final_chain = SequentialChain(chains=[name_chain, menu_chain],
                                  input_variables=['cuisine', 'total', 'course'],
                                  output_variables=['restaurant_name', 'menu_items'])

    response = final_chain({'cuisine': cuisine, 'total': total, 'course': course})

    return response

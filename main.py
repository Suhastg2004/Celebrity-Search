import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain
import streamlit as st
from constants import huggingface_key

# Set Hugging Face API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_key

# Initialize the Hugging Face pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-large")

# Load tokenizer and model for Hugging Face
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Wrap the Hugging Face pipeline in LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Streamlit app setup
st.title('Celebrity Search Results')

# Input from user
input_text = st.text_input("Search the topic you want")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events that happened around {dob} in the world"
)

# Memory buffers for conversation history
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# LLM Chains
chain = LLMChain(
    llm=llm,
    prompt=first_input_prompt,
    verbose=True,
    output_key='person',
    memory=person_memory
)

chain2 = LLMChain(
    llm=llm,
    prompt=second_input_prompt,
    verbose=True,
    output_key='dob',
    memory=dob_memory
)

chain3 = LLMChain(
    llm=llm,
    prompt=third_input_prompt,
    verbose=True,
    output_key='description',
    memory=descr_memory
)

# Sequential Chain combining the individual chains
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description'],
    verbose=True
)

# Running the chain and showing results
if input_text:
    response = parent_chain({'name': input_text})

    st.write("**Person Information:**")
    st.write(response['person'])

    st.write("**Date of Birth:**")
    st.write(response['dob'])

    st.write("**Historical Events:**")
    st.write(response['description'])

    with st.expander('Person Name History'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)

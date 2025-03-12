##########################################################################################################################################################
#importing the dependencies
##########################################################################################################################################################

import streamlit as st
from openai import OpenAI

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
import numpy as np
from sklearn.cluster import KMeans
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text
import json
import requests
import html2text

##########################################################################################################################################################
#declaring the variables
##########################################################################################################################################################

openai_api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=openai_api_key)

map_prompt = """
You will be given a single passage of a content on a website. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a full understanding of what the website is about.
Your response should be at least three paragraphs and fully encompass what was said in the passage.

```{text}```
FULL SUMMARY:
"""

combine_prompt = """
You will be given a series of summaries about a website. The summaries will be enclosed in triple backticks (```)
Your goal is to give a verbose summary of what is the website about with bullet points if necessay.
The reader should be able to grasp what is the website about.

```{text}```
VERBOSE SUMMARY:
"""

llm3 = ChatOpenAI(temperature=0,
                 openai_api_key=openai_api_key,
                 max_tokens=1000,
                 model='gpt-3.5-turbo'
                )

llm4 = ChatOpenAI(temperature=0,
                 openai_api_key=openai_api_key,
                 max_tokens=3000,
                 model='gpt-4',
                 request_timeout=120
                )

url_schema = Object(
    id="links",
    description="Summarizing a URL",
    attributes=[
        Text(
            id="url",
            description="The URL mentioned",
        )
    ],
    examples=[
        ("can you summarise this - https://streamlit.io", [{"url": "https://streamlit.io"}])
    ]
)



##########################################################################################################################################################
#defining functions
##########################################################################################################################################################

def get_link(llm3, url_schema, message):
    chain = create_extraction_chain(llm3, url_schema)
    output = chain.invoke((message))["data"]
    return output['links']

def get_response(url):
    response = requests.get(url)
    if response.status_code == 200:
        return {'response': response.text, 'status_code': 200}
    else:
        return {'response': f"Failed to retrieve content. Status code: {response.status_code}", 'status_code': 400}
    
def html_to_markdown(html_content):
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    markdown_content = converter.handle(html_content)
    return markdown_content

def chunk_text(markdown_content):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=1000, chunk_overlap=300)
    docs = text_splitter.create_documents([markdown_content])
    return docs

def get_best_chunks(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    num_clusters = min(10, len(docs))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    closest_indices = []
    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    return selected_indices

def llm_summarise(docs, selected_indices):
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    map_chain = load_summarize_chain(llm=llm3,
                                chain_type="stuff",
                                prompt=map_prompt_template)
    selected_docs = [docs[doc] for doc in selected_indices]

    summary_list = []
    for i, doc in enumerate(selected_docs):
        chunk_summary = map_chain.invoke([doc])
        summary_list.append(chunk_summary['output_text'])

    summaries = "\n".join(summary_list)
    summaries = Document(page_content=summaries)

    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    reduce_chain = load_summarize_chain(llm=llm4,
                                chain_type="stuff",
                                prompt=combine_prompt_template,
                                    )
    output = reduce_chain.invoke([summaries])
    return output['output_text']

def contains_url(llm3, url_schema, message):
    links = get_link(llm3, url_schema, message)
    if len(links) == 0:
        return False
    else:
        return links[0]['url']

def process_url(url):
    response = get_response(url)
    if response.get('status_code') == 200:
        html_content = response.get('response')
        print('html_content extracted')
        markdown_content = html_to_markdown(html_content)
        print('markdown_content converted')
        docs = chunk_text(markdown_content)
        selected_indices = get_best_chunks(docs)
        print('chunked')
        return llm_summarise(docs, selected_indices)
    return "Failed to fetch the content."



##########################################################################################################################################################


# Show title and description.
st.title("Summariser ⚡")
st.write(
    "This is a simple chatbot that uses OpenAI's models to generate summaries for any website. "
    "To use this feature, please provide any website link that you wanna summarise."
)



# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is up?"):

    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if the message contains a URL
    if contains_url(llm3, url_schema, prompt):
        url = contains_url(llm3, url_schema, prompt)  # Extract the first URL
        output = process_url(url)  # Process the URL
        response = output
    else:
        # Normal chat response
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream response
        with st.chat_message("assistant"):
            response = st.write_stream(stream)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

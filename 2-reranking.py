#!/usr/bin/env python
# coding: utf-8

# ### Reranking Hybrid Search Statergies

# Re-ranking is a second-stage filtering process in retrieval systems, especially in RAG pipelines, where we:
# 
# 1. First use a fast retriever (like BM25, FAISS, hybrid) to fetch top-k documents quickly.
# 
# 2. Then use a more accurate but slower model (like a cross-encoder or LLM) to re-score and reorder those documents by relevance to the query.
# 
# 👉 It ensures that the most relevant documents appear at the top, improving the final answer from the LLM.

# In[1]:


from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser 


# In[18]:


## load text file
loader=TextLoader("langchain_sample.txt")
raw_docs=loader.load()

# Split text into document chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)
docs


# In[3]:


## user query
query="How can i use langchain to build an application with memory and tools?"


# In[19]:


### FAISS and Huggingface model Embeddings

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore=FAISS.from_documents(docs,embedding_model)
retriever=vectorstore.as_retriever(search_kwargs={"k":8})


# In[20]:


## OpenAI Embedding
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
from langchain_openai import OpenAIEmbeddings

embeddings=OpenAIEmbeddings()
vectorstore_openai=FAISS.from_documents(docs,embeddings)
retriever_openai=vectorstore_openai.as_retriever(search_kwargs={"k":8})




# In[21]:


retriever


# In[22]:


retriever_openai


# In[23]:


## prompt and use the llm
from langchain.chat_models import init_chat_model
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm=init_chat_model("groq:gemma2-9b-it")
llm


# In[24]:


# Prompt Template
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Your task is to rank the following documents from most to least relevant to the user's question.

User Question: "{question}"

Documents:
{documents}

Instructions:
- Think about the relevance of each document to the user's question.
- Return a list of document indices in ranked order, starting from the most relevant.

Output format: comma-separated document indices (e.g., 2,1,3,0,...)
""")


# In[25]:


retrieved_docs=retriever.invoke(query)
retrieved_docs


# In[26]:


chain=prompt| llm | StrOutputParser()
chain


# In[27]:


doc_lines = [f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
formatted_docs = "\n".join(doc_lines)


# In[28]:


doc_lines


# In[29]:


formatted_docs


# In[30]:


response=chain.invoke({"question":query,"documents":formatted_docs})
response


# In[31]:


# Step 5: Parse and rerank
indices = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
indices


# In[32]:


retrieved_docs


# In[33]:


reranked_docs = [retrieved_docs[i] for i in indices if 0 <= i < len(retrieved_docs)]
reranked_docs


# In[34]:


# Step 6: Show results
print("\n📊 Final Reranked Results:")
for i, doc in enumerate(reranked_docs, 1):
    print(f"\nRank {i}:\n{doc.page_content}")


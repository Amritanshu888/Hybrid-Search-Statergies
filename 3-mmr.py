#!/usr/bin/env python
# coding: utf-8

# ### Maximal Marginal Relevance
# MMR (Maximal Marginal Relevance) is a powerful diversity-aware retrieval technique used in information retrieval and RAG pipelines to balance relevance and novelty when selecting documents.

# In[7]:


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# In[8]:


import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


# In[9]:


# Step 1: Load and chunk the document
loader = TextLoader("langchain_rag_dataset.txt")
raw_docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(raw_docs)
chunks


# In[10]:


# Step 2: FAISS Vector Store with HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)


# In[11]:


### Step 3: Create MMR Retirever
retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)


# In[ ]:





# In[6]:


# Step 4: Prompt and LLM
prompt = PromptTemplate.from_template("""
Answer the question based on the context provided.

Context:
{context}

Question: {input}
""")
llm=init_chat_model("groq:gemma2-9b-it")


# In[12]:


# Step 5: RAG Pipeline
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)


# In[13]:


# Step 6: Query
query = {"input": "How does LangChain support agents and memory?"}
response = rag_chain.invoke(query)

print("✅ Answer:\n", response["answer"])


# In[14]:


response


# In[ ]:





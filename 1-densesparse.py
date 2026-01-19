#!/usr/bin/env python
# coding: utf-8

# ## Hybrid Retriever- Combining Dense And Sparse Retriever

# In[1]:


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document



# In[2]:


# Step 1: Sample documents
docs = [
    Document(page_content="LangChain helps build LLM applications."),
    Document(page_content="Pinecone is a vector database for semantic search."),
    Document(page_content="The Eiffel Tower is located in Paris."),
    Document(page_content="Langchain can be used to develop agentic ai application."),
    Document(page_content="Langchain has many types of retrievers.")
]

# Step 2: Dense Retriever (FAISS + HuggingFace)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
dense_vectorstore = FAISS.from_documents(docs, embedding_model)
dense_retriever = dense_vectorstore.as_retriever()


# In[4]:


### Sparse Retriever(BM25)
sparse_retriever=BM25Retriever.from_documents(docs)
sparse_retriever.k=3 ##top- k documents to retriever

## step 4 : Combine with Ensemble Retriever
hybrid_retriever=EnsembleRetriever(
    retrievers=[dense_retriever,sparse_retriever],
    weight=[0.7,0.3]
)


# In[5]:


hybrid_retriever


# In[6]:


# Step 5: Query and get results
query = "How can I build an application using LLMs?"
results = hybrid_retriever.invoke(query)

# Step 6: Print results
for i, doc in enumerate(results):
    print(f"\n🔹 Document {i+1}:\n{doc.page_content}")


# ### RAG Pipeline with hybrid retriever

# In[8]:


from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# In[12]:


# Step 5: Prompt Template
prompt = PromptTemplate.from_template("""
Answer the question based on the context below.

Context:
{context}

Question: {input}
""")

## step 6-llm
llm=init_chat_model("openai:gpt-3.5-turbo",temperature=0.2)
llm


# In[13]:


### Create stuff Docuemnt Chain
document_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)

## create Full rAg chain
rag_chain=create_retrieval_chain(retriever=hybrid_retriever,combine_docs_chain=document_chain)
rag_chain


# In[14]:


# Step 9: Ask a question
query = {"input": "How can I build an app using LLMs?"}
response = rag_chain.invoke(query)

# Step 10: Output
print("✅ Answer:\n", response["answer"])

print("\n📄 Source Documents:")
for i, doc in enumerate(response["context"]):
    print(f"\nDoc {i+1}: {doc.page_content}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





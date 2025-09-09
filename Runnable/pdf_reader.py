from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
# load document
loader = TextLoader('Runnable/DSA.txt')
documents = loader.load()

# split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# converts text into embeddings (HuggingFace)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(docs, embeddings)

# create a retriever
retriever = vector_store.as_retriever()

# manually retrieve relevant documents
query = "What is LangChain?"
relevant_docs = retriever.get_relevant_documents(query)

# combine retrieval text into a single string
retrieve_text = "\n".join([doc.page_content for doc in relevant_docs])

# initialize the llm
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# properly format the prompt
prompt = f"Based on the following text, answer the question:\n\nQuestion: {query}\n\nContext:\n{retrieve_text}"

# get the answer
answer = llm.invoke(prompt)
print(answer.content)

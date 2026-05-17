from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorestore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

retriver = vectorestore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)
llm = ChatMistralAI(model='mistral-small-2603', temperature=0.7)


#prompt_template

prompt = ChatPromptTemplate.from_messages([
    ("system","""You are a helful assistant.
    use only the provided context to answer the question. If you don't know the answer, say you don't know. Always use all the provided context to answer the question."""
    ),
    ("human","""context: {context}
    question: {question}""")
])
print("===Rag Chain Output===")
print("press 0 to exit")
while True:
    query = input("Enter your question: ")
    if query == "0":
        print("Exiting...")
        break
    docs = retriver.invoke(query)
    context = "\n\n".join([f"{doc.page_content} (Source: {doc.metadata['source']})" for doc in docs])
    final_prompt = prompt.invoke({
        "context":context,
        "question":query
    })
    response = llm.invoke(final_prompt)
    print(f"\n AI:{response.content}")
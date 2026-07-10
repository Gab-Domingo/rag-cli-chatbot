from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

CHAT_MODEL = "gemini-3.5-flash"
N_RESULTS = 3

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
    "If the answer cannot be found in the context, clearly state that you do not know."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTION),
    ("human", "Retrieved Context:\n{context}\n\nUser Question: {input}"),
])


def format_docs(documents):
    if not documents:
        return "No relevant context found."

    blocks = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[{source}]\n{doc.page_content}")
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": N_RESULTS})
        self.llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "input": RunnablePassthrough(),
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

    def answer(self, query):
        return self.chain.invoke(query)

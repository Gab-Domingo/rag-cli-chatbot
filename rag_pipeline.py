from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

CHAT_MODEL = "gemini-3.5-flash"

SYSTEM_INSTRUCTION = ("""
    You are PoBot, an AI assistant that answers questions about Hong Kong labour regulations.

Use only the retrieved context to answer.

Guidelines:
- Answer clearly and concisely.
- Use bullet points when listing rights or requirements.
- Do not mention "based on the provided context."
- If the answer is not in the retrieved context, say:
  "I couldn't find that information in the indexed documents."
- Do not invent or assume facts.
"""
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
        blocks.append(
            f"Source: {source}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10})
        self.llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)

    def answer(self, query):
        #Retrieve relevant documents
        docs = self.retriever.invoke(query)

        #Format context
        context = format_docs(docs)

        #Generate answer
        answer = (
            PROMPT
            | self.llm
            | StrOutputParser()
        ).invoke({
            "context": context,
            "input": query,
        })

        #Collecting citations
        citations = []
        seen = set()

        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source not in seen:
                seen.add(source)
                citations.append(source)

        return {
            "answer": answer,
            "citations": citations,
        }

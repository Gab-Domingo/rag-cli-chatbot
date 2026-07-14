from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

CHAT_MODEL = "gemini-3.5-flash"

SYSTEM_INSTRUCTION = ("""
You are PoBot, an AI assistant that answers questions about Hong Kong labour regulations.

Answer the user's question using ONLY the retrieved information provided to you.

Rules:
- Use only the retrieved information. Do not use outside knowledge or make assumptions.
- If the retrieved information does not contain enough information to answer the question, respond exactly:
  "I couldn't find that information in the indexed documents."
- Begin your answer immediately with the relevant information. Do not include introductions, disclaimers, or phrases describing where the information came from.
- Answer naturally, as if you already know the information.
- NEVER mention the retrieval process, retrieved context, provided documents, indexed documents, or source material in your answer.
- NEVER begin your answer with phrases such as:
  - "Based on the provided..."
  - "According to the retrieved..."
  - "The context states..."
  - "The documents indicate..."
- Do not explain how you obtained the information.
- Use clear and professional language.
- Preserve important legal terms and conditions exactly as written in the retrieved information.
- Include all relevant conditions, exceptions, and eligibility requirements found in the retrieved information.
- Do not generate citations in the response. Citations are added separately by the application.

Your highest priority is factual accuracy. If the answer is unsupported by the retrieved information, do not speculate.
"""
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTION),
    ("human", "Reference Information:\n{context}\n\nUser Question: {input}"),
])

class RAGPipeline:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10})
        self.llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)

    def _format_context(self,docs):
        """
        Formats the retrieved documents for the RAG pipeline.

        It concatenates the documents and returns the formatted text.

        Args:
            docs: List of documents to format.

        Returns:
            str: Formatted context.
        """

        blocks = []

        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page_number")
            if page:
                header = f"Source: {source}, (Page {page})"
            else:
                header = f"Source: {source}"

            blocks.append(
                f"{header}\n{doc.page_content}"
            )

        return "\n\n".join(blocks)


    def _extract_citations(self, docs):
        """
        Extracts citations from the retrieved documents.

        Args:
            docs: List of documents to extract citations from.

        Returns:
            list: List of citations.
        """

        citations = []
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source")
            page = doc.metadata.get("page_number")

            key = (source,page)

            
            if key not in seen:
                seen.add(key)
                citations.append({
                    "source": source,
                    "page": page,
                })
        return citations


    def answer(self, query):
        """
        Answers a query using the RAG pipeline.

        Args:
            query: The query to answer.

        Returns:
            dict: A dictionary containing the answer and citations.
        """
        #Retrieve relevant documents
        docs = self.retriever.invoke(query)

        #Format context and citations
        context = self._format_context(docs)

        answer = (
            PROMPT
            | self.llm
            | StrOutputParser()
        ).invoke({
            "context": context,
            "input": query,
        })


        citations = self._extract_citations(docs)
        return {
            "answer": answer,
            "citations": citations,
        }

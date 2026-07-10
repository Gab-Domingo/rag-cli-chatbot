EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"
N_RESULTS = 3

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
    "If the answer cannot be found in the context, clearly state that you do not know."
)


class RAGPipeline:
    def __init__(self, client, collection):
        self.client = client
        self.collection = collection
        self.chat = client.chats.create(model=CHAT_MODEL)

    def embed_query(self, query):
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
        )
        return response.embeddings[0].values

    def retrieve_context(self, query):
        query_vector = self.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=N_RESULTS,
            include=["documents", "metadatas"],
        )

        if not results["documents"] or not results["documents"][0]:
            return "No relevant context found."

        context_blocks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            source = meta.get("source", "unknown") if meta else "unknown"
            context_blocks.append(f"[{source}]\n{doc}")

        return "\n\n".join(context_blocks)

    def build_prompt(self, query, retrieved_context):
        return (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"Retrieved Context:\n{retrieved_context}\n\n"
            f"User Question: {query}"
        )

    def answer(self, query):
        retrieved_context = self.retrieve_context(query)
        prompt = self.build_prompt(query, retrieved_context)
        response = self.chat.send_message(prompt)
        return response.text

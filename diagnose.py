from rag_eval import RAGEval
import os
from dotenv import load_dotenv
from ingest import get_vectorstore, get_embeddings
from rag_pipeline import RAGPipeline


load_dotenv()
diagnostic_pipeline = RAGPipeline(vectorstore=get_vectorstore())

def test():
    
    doctor = RAGEval(
        retriever=diagnostic_pipeline.retriever,
        embedding_model=get_embeddings().embed_documents
    )
    report = doctor.inspect(
        query = "What are the termination provisions of the Employment Ordinance?"
    )
    report.print()
   
if __name__ == "__main__":
    test()  
    


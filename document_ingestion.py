import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import docx
import markdown
import chromadb
from chromadb.config import Settings

class DocumentIngester:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', persist_directory="./chroma_db"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name="hr_documents")

    def extract_text_from_pdf(self, file_path):
        reader = PdfReader(file_path)
        chunks = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            # Split page text into paragraphs
            paragraphs = page_text.split('\n\n')
            for para_num, para in enumerate(paragraphs):
                if para.strip():
                    chunks.append({
                        'text': para.strip(),
                        'page': page_num + 1,
                        'lines': f"{para_num*5 + 1}-{para_num*5 + len(para.split())}"  # Approximate lines
                    })
        return chunks

    def extract_text_from_docx(self, file_path):
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    def extract_text_from_md(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def ingest_document(self, file_path, doc_id=None, metadata=None):
        if doc_id is None:
            doc_id = os.path.basename(file_path)

        if metadata is None:
            metadata = {"source": file_path, "version": "1.0"}

        if file_path.endswith('.pdf'):
            chunks = self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            text = self.extract_text_from_docx(file_path)
            # Split text into chunks (simple paragraph-based splitting)
            paragraphs = text.split('\n\n')
            chunks = [chunk.strip() for chunk in paragraphs if chunk.strip()]
        elif file_path.endswith('.md'):
            text = self.extract_text_from_md(file_path)
            # Split text into chunks (simple paragraph-based splitting)
            paragraphs = text.split('\n\n')
            chunks = [chunk.strip() for chunk in paragraphs if chunk.strip()]
        else:
            raise ValueError("Unsupported file type")

        # Generate embeddings
        if file_path.endswith('.pdf'):
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_model.encode(chunk_texts)
        else:
            embeddings = self.embedding_model.encode(chunks)

        # Add to ChromaDB
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            if file_path.endswith('.pdf'):
                chunk_metadata["page"] = chunk['page']
                chunk_metadata["lines"] = chunk['lines']
                chunk_text = chunk['text']
            else:
                chunk_text = chunk
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding.tolist()],
                metadatas=[chunk_metadata],
                documents=[chunk_text]
            )

        print(f"Ingested document: {doc_id} with {len(chunks)} chunks")

    def search_documents(self, query, n_results=5):
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        return results

    def update_document_version(self, file_path, new_version):
        # For simplicity, we'll re-ingest with a new version
        doc_id = os.path.basename(file_path).replace('.pdf', '').replace('.docx', '').replace('.md', '')
        new_doc_id = f"{doc_id}_v{new_version}"
        metadata = {"source": file_path, "version": new_version}
        self.ingest_document(file_path, doc_id=new_doc_id, metadata=metadata)

if __name__ == "__main__":
    ingester = DocumentIngester()

    # Example usage
    # ingester.ingest_document("HR_Policy_v3.1.pdf")
    # results = ingester.search_documents("What is the leave policy?")
    # print(results)

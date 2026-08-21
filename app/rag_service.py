import os
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from sklearn.feature_extraction.text import TfidfVectorizer
from .models import GuidanceEmbedding

# Global TF-IDF Vectorizer fitted on startup
vectorizer = TfidfVectorizer(max_features=384)
is_fitted = False
global_docs = []

def fit_vectorizer_on_guidelines(filepath: str):
    """
    Reads the text guidelines corpus and fits the TF-IDF vectorizer.
    """
    global vectorizer, is_fitted, global_docs
    if not os.path.exists(filepath):
        print(f"Guidelines file not found at {filepath}")
        return []

    documents = []
    current_doc = {}
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.startswith("=== ") and line.endswith(" ==="):
            if current_doc:
                documents.append(current_doc)
            current_doc = {"title": "", "content": ""}
        elif line.startswith("TITLE:"):
            current_doc["title"] = line.replace("TITLE:", "").strip()
        elif line.startswith("CONTENT:"):
            current_doc["content"] = ""
        elif current_doc and line:
            current_doc["content"] += line + "\n"
            
    if current_doc:
        documents.append(current_doc)
        
    # Fit the vectorizer on document content
    corpus = [doc["content"] for doc in documents if doc["content"]]
    if corpus:
        vectorizer.fit(corpus)
        is_fitted = True
        global_docs = documents
        print(f"TF-IDF Vectorizer successfully fitted on {len(corpus)} documents.")
        
    return documents

def initialize_rag(db: Session):
    """
    Populates the pgvector table with guidelines if it's empty.
    """
    global is_fitted
    guidelines_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "emergency_guidance.txt"))
        
    docs = fit_vectorizer_on_guidelines(guidelines_path)
    
    # Check if table already populated
    count = db.query(GuidanceEmbedding).count()
    if count > 0:
        print("RAG database is already populated.")
        return

    if not is_fitted or not docs:
        print("No documents found to initialize RAG database.")
        return

    print("Populating RAG database with emergency guidelines...")
    for doc in docs:
        if not doc["content"]:
            continue
        # Get embedding vector
        vec = vectorizer.transform([doc["content"]]).toarray()[0]
        # Pad with zeros if less than 384 (though max_features is 384, vocabulary could be smaller for small corpus)
        if len(vec) < 384:
            vec = np.pad(vec, (0, 384 - len(vec)), 'constant')
            
        emb = GuidanceEmbedding(
            title=doc["title"],
            content=doc["content"].strip(),
            embedding=vec.tolist()
        )
        db.add(emb)
    
    db.commit()
    print("RAG database population complete.")

def query_rag(db: Session, query: str) -> str:
    """
    Vector search query calculating cosine similarity in Python (SQLite compatible)
    """
    global vectorizer, is_fitted, global_docs
    
    # Fit vectorizer if not already fitted
    if not is_fitted:
        guidelines_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "emergency_guidance.txt"))
        fit_vectorizer_on_guidelines(guidelines_path)
        
    if not is_fitted:
        return "Guidance: Remain calm. Help is on the way."
        
    # Transform query to vector
    query_vec = vectorizer.transform([query]).toarray()[0]
    if len(query_vec) < 384:
        query_vec = np.pad(query_vec, (0, 384 - len(query_vec)), 'constant')
        
    try:
        # Load all guidance from db or fallback to in-memory global_docs
        doc_list = []
        if db is not None:
            all_embeddings = db.query(GuidanceEmbedding).all()
            for emb in all_embeddings:
                if emb.embedding:
                    emb_vec = np.array(emb.embedding)
                    if len(emb_vec) < 384:
                        emb_vec = np.pad(emb_vec, (0, 384 - len(emb_vec)), 'constant')
                    doc_list.append({
                        "title": emb.title,
                        "content": emb.content,
                        "embedding": emb_vec
                    })
        else:
            # Fallback to local global_docs embeddings calculated on the fly
            for doc in global_docs:
                doc_vec = vectorizer.transform([doc["content"]]).toarray()[0]
                if len(doc_vec) < 384:
                    doc_vec = np.pad(doc_vec, (0, 384 - len(doc_vec)), 'constant')
                doc_list.append({
                    "title": doc["title"],
                    "content": doc["content"],
                    "embedding": doc_vec
                })

        if not doc_list:
            return "Guidance: Remain calm. Help is on the way."
            
        best_doc = None
        min_distance = float('inf')
        
        # Calculate cosine distance in python
        # cosine distance = 1 - (A . B) / (||A|| * ||B||)
        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0:
            norm_q = 1e-9
            
        for emb in doc_list:
            emb_vec = emb["embedding"]
            norm_e = np.linalg.norm(emb_vec)
            if norm_e == 0:
                norm_e = 1e-9
                
            dot_product = np.dot(query_vec, emb_vec)
            distance = 1.0 - (dot_product / (norm_q * norm_e))
            
            if distance < min_distance:
                min_distance = distance
                best_doc = emb
                
        if best_doc:
            return f"Official First Aid Instructions ({best_doc['title']}):\n{best_doc['content']}"
    except Exception as e:
        print(f"RAG query execution failed: {e}")
        
    # Fallback keyword matching if SQL vector query fails
    # Standard keyword searches
    query_lower = query.lower()
    best_doc = None
    if "fire" in query_lower or "burn" in query_lower:
        best_doc = "Cool the burn with cool running water for 10-20 minutes. Cover loosely."
    elif "heart" in query_lower or "cpr" in query_lower or "breathing" in query_lower or "choking" in query_lower:
        best_doc = "Begin CPR immediately. Push hard and fast at 100-120 compressions per minute."
    elif "car" in query_lower or "accident" in query_lower or "collision" in query_lower:
        best_doc = "Do not move injured victims unless there is immediate danger. Apply direct pressure to bleeding."
    elif "shoot" in query_lower or "wound" in query_lower or "bleed" in query_lower:
        best_doc = "Apply tourniquet 2-3 inches above bleeding extremity. For chest wounds, pack and apply pressure."
    else:
        best_doc = "Keep the patient warm and dry, monitor breathing, and reassure them that emergency responders are en route."
        
    return f"Emergency Guidance:\n{best_doc}"

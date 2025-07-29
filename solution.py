import os
import json
import fitz
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

EMBEDDING_MODEL = 'embedding_model'
# CLASSIFIER_MODEL = "distilbert-base-uncased"
SECTION_CHUNK_SIZE = 1024 
# TOP_SECTIONS = total_pdfs
# TOP_SUBSECTIONS = total_pdfs

embedder = SentenceTransformer(EMBEDDING_MODEL)
# classifier = pipeline("text-classification", model=CLASSIFIER_MODEL)

def count_pdfs(input_dir):
    return sum(1 for f in os.listdir(input_dir) if f.endswith(".pdf"))

def normalize_text(text):
    replacements = {
        '\ufb00': 'ff',  
        '\ufb01': 'fi',   
        '\ufb02': 'fl',   
        '\ufb03': 'ffi', 
        '\ufb04': 'ffl', 
        '\u2022': '•',    
        '\ufb00': 'ff',  
    }
    for orig_char, replacement in replacements.items():
        text = text.replace(orig_char, replacement)
    return text

def load_input_data(input_dir):
    input_path = os.path.join(input_dir, 'input.json')
    if not os.path.exists(input_path):
        raise FileNotFoundError("input.json not found")
    
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)
    
    persona = data['persona']['role']
    job = data['job_to_be_done']['task']
    return persona, job

def extract_structured_content(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        font_stats = {}
        
        
        for b in blocks:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    font = s["font"].lower()
                    size = s["size"]
                    font_stats[font] = font_stats.get(font, []) + [size]
        
        
        heading_sizes = set()
        for sizes in font_stats.values():
            if sizes:
                avg_size = sum(sizes) / len(sizes)
                heading_sizes.update([s for s in sizes if s >= 1.3 * avg_size])
        
        
        current_heading = ""
        current_content = []
        
        for b in blocks:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    raw_text = s["text"].strip()
                    if not raw_text:
                        continue
                    
                    
                    text = normalize_text(raw_text)
                    
                    
                    if s["size"] in heading_sizes:
                        if current_heading or current_content:
                            sections.append({
                                "heading": current_heading,
                                "content": " ".join(current_content),
                                "page": page_num,
                                "document": os.path.basename(pdf_path)
                            })
                        current_heading = text
                        current_content = []
                    else:
                        current_content.append(text)
        

        if current_heading or current_content:
            sections.append({
                "heading": current_heading,
                "content": " ".join(current_content),
                "page": page_num,
                "document": os.path.basename(pdf_path)
            })
    
    return sections

def chunk_section(section):
    words = section["content"].split()
    for i in range(0, len(words), SECTION_CHUNK_SIZE):
        yield {
            "text": " ".join(words[i:i+SECTION_CHUNK_SIZE]),
            "heading": section["heading"],
            "document": section["document"],
            "page": section["page"]
        }

def process_documents(input_dir, persona, job,TOP_SECTIONS):
    context_embedding = embedder.encode(f"{persona}: {job}")
    all_chunks = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            sections = extract_structured_content(pdf_path)
            for section in sections:
                for chunk in chunk_section(section):
                    all_chunks.append(chunk)
                
    chunk_texts = [f"{c['heading']} {c['text']}" for c in all_chunks]
    chunk_embeddings = embedder.encode(chunk_texts)
    similarities = cosine_similarity([context_embedding], chunk_embeddings)[0]
    for i, chunk in enumerate(all_chunks):
        chunk["similarity"] = float(similarities[i])
        
    ranked_chunks = sorted(all_chunks, key=lambda x: x["similarity"], reverse=True)

    extracted_sections = []
    seen_sections = set()
    
    for chunk in ranked_chunks:
        section_key = (chunk["document"], chunk["heading"])
        if section_key not in seen_sections and len(extracted_sections) < TOP_SECTIONS:
            extracted_sections.append({
                "document": chunk["document"],
                "section_title": chunk["heading"] or chunk["text"],
                "page_number": chunk["page"],
                "importance_rank": len(extracted_sections) + 1
            })
            seen_sections.add(section_key)
    
    subsection_analysis = [
        {
            "document": c["document"],
            "refined_text": c["text"],
            "page_number": c["page"]
        } for c in ranked_chunks[:TOP_SECTIONS]
    ]
    
    return {
        "metadata": {
            "input_documents": [f for f in os.listdir(input_dir) if f.endswith(".pdf")],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

if __name__ == "__main__":
    input_dir = "/app/input"
    output_dir = "/app/output"
    # input_dir='input/'
    # output_dir='output/'
    os.makedirs(output_dir, exist_ok=True)
    total_pdfs=count_pdfs(input_dir)

    persona, job= load_input_data(input_dir)
    result = process_documents(input_dir, persona, job,total_pdfs)
    with open(os.path.join(output_dir, "output.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
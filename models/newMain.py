import fitz  
import json
import os


def extract_pdf_structure(pdf_path):
    doc = fitz.open(pdf_path)
    json_result = {"title": "", "outline": []}

    candidates = []
    for page_num, page in enumerate(doc, start=1): 
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] != 0:
                continue  # Not text
            for line in block["lines"]:
                line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                if not line_text:
                    continue
                
                for span in line["spans"]:
                    font_size = span["size"]
                    
                    if font_size > 15:
                        candidates.append({
                            "text": line_text,
                            "font_size": font_size,
                            "page": page_num
                        })
            
                    if font_size > (json_result.get('max_size', 0)):
                        json_result['max_size'] = font_size
                        json_result['title'] = line_text

    seen = set()
    for c in candidates:
        key = (c['text'], c['page'])
        if key not in seen:
            json_result["outline"].append({
                "level": "H1",
                "text": c["text"],
                "page": c["page"]
            })
            seen.add(key)

    json_result.pop('max_size', None)
    return json_result



if __name__ == "__main__":
    import sys
    pdf_file = sys.argv[1]
    result = extract_pdf_structure(pdf_file)

    os.makedirs("output", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_file))[0]
    output_path = os.path.join("output", base_name + ".json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Extracted JSON saved to {output_path}")

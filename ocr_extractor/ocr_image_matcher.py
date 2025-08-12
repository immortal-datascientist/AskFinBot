
import pytesseract
from PIL import Image
import spacy
from sentence_transformers import SentenceTransformer, util

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path).convert("L")  # grayscale
        text = pytesseract.image_to_string(image)
        return text.strip().lower()
    except Exception:
        return ""

def extract_keywords(text):
    doc = nlp(text)
    return [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]

def find_relevant_images_with_keywords(image_map, user_question, matched_pages, top_k=5):
    keywords = extract_keywords(user_question)
    image_matches = []

    nearby_pages = set()
    for p in matched_pages:
        for offset in range(0, 3):  # only future pages
            if (p + offset) > 0:
                nearby_pages.add(p + offset)

    checked = 0
    for page in sorted(nearby_pages):
        if page in image_map:
            for img_path in image_map[page]:
                ocr_text = extract_text_from_image(img_path)
                if not ocr_text:
                    continue

                hits = [kw for kw in keywords if kw in ocr_text]
                match_score = len(hits)

                if match_score >= 1:
                    image_matches.append((img_path, match_score, page))

                checked += 1
                if checked >= 20:
                    break

    sorted_images = sorted(image_matches, key=lambda x: x[1], reverse=True)
    return sorted_images[:top_k]

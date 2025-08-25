

############################################ This Is with Sentence Transformers ############################################
# import pytesseract
# from PIL import Image
# import spacy
# from sentence_transformers import SentenceTransformer, util

# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# nlp = spacy.load("en_core_web_sm")

# def extract_text_from_image(image_path):
#     try:
#         image = Image.open(image_path).convert("L")  # grayscale
#         text = pytesseract.image_to_string(image)
#         return text.strip().lower()
#     except Exception:
#         return ""

# def extract_keywords(text):
#     doc = nlp(text)
#     return [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]

# def find_relevant_images_with_keywords(image_map, user_question, matched_pages, top_k=5):
#     keywords = extract_keywords(user_question)
#     image_matches = []

#     nearby_pages = set()
#     for p in matched_pages:
#         for offset in range(0, 3):  # only future pages
#             if (p + offset) > 0:
#                 nearby_pages.add(p + offset)

#     checked = 0
#     for page in sorted(nearby_pages):
#         if page in image_map:
#             for img_path in image_map[page]:
#                 ocr_text = extract_text_from_image(img_path)
#                 if not ocr_text:
#                     continue

#                 hits = [kw for kw in keywords if kw in ocr_text]
#                 match_score = len(hits)

#                 if match_score >= 1:
#                     image_matches.append((img_path, match_score, page))

#                 checked += 1
#                 if checked >= 20:
#                     break

#     sorted_images = sorted(image_matches, key=lambda x: x[1], reverse=True)
#     return sorted_images[:top_k]
#################################################################################################



# # ocr_image_matcher.py
# import spacy
# from sentence_transformers import SentenceTransformer
# from typing import List, Tuple

# nlp = spacy.load("en_core_web_sm")
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # kept for potential semantic use

# def extract_keywords(text: str) -> List[str]:
#     doc = nlp(text.lower())
#     return [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]

# def find_relevant_images_with_keywords(
#     image_map_with_ocr: dict,  # {page_number: [(img_path, ocr_text), ...]}
#     user_question: str,
#     matched_chunks,
#     matched_pages: List[int],
#     top_k: int = 5
# ) -> List[Tuple[str, int, int]]:
#     """
#     Faster image search:
#     - Uses OCR text cached at PDF ingestion
#     - Combines keywords from user question + retrieved chunks
#     - Restricts to same page ±1
#     - Requires match_score >= 2
#     """

#     # CHANGE ✅ Combined keywords from question and context text
#     context_text = " ".join([doc.page_content for doc in matched_chunks])
#     keywords = extract_keywords(user_question + " " + context_text)

#     image_matches = []
#     nearby_pages = set()

#     # CHANGE ✅ Restrict to ±1 page instead of future 3 pages
#     for p in matched_pages:
#         for offset in range(-1, 2):
#             if (p + offset) > 0:
#                 nearby_pages.add(p + offset)

#     # CHANGE ✅ No OCR here — use cached text
#     for page in sorted(nearby_pages):
#         if page in image_map_with_ocr:
#             for img_path, ocr_text in image_map_with_ocr[page]:
#                 if not ocr_text:
#                     continue

#                 hits = [kw for kw in keywords if kw in ocr_text]
#                 match_score = len(hits)

#                 # CHANGE ✅ Require at least 2 keyword matches
#                 if match_score >= 2:
#                     image_matches.append((img_path, match_score, page))

#     sorted_images = sorted(image_matches, key=lambda x: x[1], reverse=True)
#     return sorted_images[:top_k]





############################################ This Is without Sentence Transformers ############################################
# import pytesseract
# from PIL import Image
# import spacy

# # Removed: from sentence_transformers import SentenceTransformer, util   ## Open This
# # Removed: embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# nlp = spacy.load("en_core_web_sm")

# def extract_text_from_image(image_path):
#     try:
#         image = Image.open(image_path).convert("L")  # grayscale
#         text = pytesseract.image_to_string(image)
#         return text.strip().lower()
#     except Exception:
#         return ""

# def extract_keywords(text):
#     doc = nlp(text)
#     return [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]

# def find_relevant_images_with_keywords(image_map, user_question, matched_pages, top_k=5):
#     keywords = extract_keywords(user_question)
#     image_matches = []

#     nearby_pages = set()
#     for p in matched_pages:
#         for offset in range(0, 3):  # only future pages
#             if (p + offset) > 0:
#                 nearby_pages.add(p + offset)

#     checked = 0
#     for page in sorted(nearby_pages):
#         if page in image_map:
#             for img_path in image_map[page]:
#                 ocr_text = extract_text_from_image(img_path)
#                 if not ocr_text:
#                     continue

#                 hits = [kw for kw in keywords if kw in ocr_text]
#                 match_score = len(hits)

#                 if match_score >= 1:
#                     image_matches.append((img_path, match_score, page))

#                 checked += 1
#                 if checked >= 20:
#                     break

#     sorted_images = sorted(image_matches, key=lambda x: x[1], reverse=True)
#     return sorted_images[:top_k]
##################################################################################################




# ocr_image_matcher.py
import pytesseract
from PIL import Image
import spacy
from typing import List, Tuple, Dict, Any

nlp = spacy.load("en_core_web_sm")

# Cache for OCR results to avoid re-running on the same image
_OCR_CACHE: Dict[str, str] = {}

def extract_text_from_image(image_path: str) -> str:
    """Run OCR on an image (grayscale) and cache result."""
    if image_path in _OCR_CACHE:
        return _OCR_CACHE[image_path]
    try:
        image = Image.open(image_path).convert("L")  # grayscale
        text = pytesseract.image_to_string(image)
        text = text.strip().lower()
    except Exception:
        text = ""
    _OCR_CACHE[image_path] = text
    return text

def extract_keywords(text: str) -> List[str]:
    """Extract nouns, proper nouns, and verbs (lowercased)."""
    doc = nlp(text.lower())
    return [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"] and len(token.text) > 2]

def _normalize_image_map(image_map: Dict[int, Any]) -> Dict[int, List[Tuple[str, str]]]:
    """
    Convert {page: [paths]} or {page: [(path, ocr_text)]}
    into {page: [(path, ocr_text)]} form.
    """
    normalized: Dict[int, List[Tuple[str, str]]] = {}
    for page, items in (image_map or {}).items():
        normalized[page] = []
        for item in items:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                img_path = item[0]
                ocr_text = (item[1] or "").lower()
            else:
                img_path = item
                ocr_text = extract_text_from_image(img_path)
            normalized[page].append((img_path, ocr_text))
    return normalized

def find_relevant_images_with_keywords(
    image_map: Dict[int, Any],
    user_question: str,
    matched_chunks,
    matched_pages: List[int],
    top_k: int = 5
) -> List[Tuple[str, int, int]]:
    """
    Improved image search:
    - Uses OCR text (cached if available)
    - Combines keywords from question + matched text chunks
    - Restricts to same page ±1
    - Requires >= 2 keyword matches
    """
    # Combine keywords from question and retrieved chunks
    context_text = " ".join([getattr(doc, "page_content", "") for doc in (matched_chunks or [])])
    keywords = extract_keywords((user_question or "") + " " + context_text)
    if not keywords:
        return []

    img_map_with_ocr = _normalize_image_map(image_map)

    nearby_pages = set()
    for p in matched_pages:
        for offset in (-1, 0, 1):  # previous, same, next
            if (p + offset) > 0:
                nearby_pages.add(p + offset)

    image_matches: List[Tuple[str, int, int]] = []
    for page in sorted(nearby_pages):
        if page not in img_map_with_ocr:
            continue
        for img_path, ocr_text in img_map_with_ocr[page]:
            if not ocr_text:
                continue
            hits = [kw for kw in keywords if kw in ocr_text]
            match_score = len(hits)
            if match_score >= 2:  # require at least 2 matches
                image_matches.append((img_path, match_score, page))

    return sorted(image_matches, key=lambda x: x[1], reverse=True)[:top_k]

# ocr_image_matcher.py
import pytesseract
from PIL import Image, ImageOps, ImageFilter
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


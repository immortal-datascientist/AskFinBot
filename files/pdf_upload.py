

##################################  Original code ######################################
import streamlit as st
import os
import re
import pickle
import fitz  # PyMuPDF
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from user_interface.chat_ui import extract_table_images_from_pdf
from typing import List, Optional, Dict
from langchain.schema import Document
import re as _re

# =====================================================================================
def _find_footer_cut_y_fitz(page: fitz.Page) -> Optional[float]:
    """
    Detect the bottom horizontal border line and return its Y coordinate.
    Heuristic:
      - Look in page drawings for horizontal lines / thin rects
      - Must be in the bottom 25–30% of the page
      - Must span >=60% of page width
    Returns:
      y (float) if found, else None.
    """
    W, H = float(page.rect.width), float(page.rect.height)
    min_width = 0.60 * W
    min_y = 0.70 * H  # bottom 30%

    best_y = None
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []

    def _pt_xy(p):
        """Return (x, y) from a tuple/list or fitz.Point."""
        if isinstance(p, (tuple, list)) and len(p) >= 2:
            return float(p[0]), float(p[1])
        # fitz uses Point sometimes
        if hasattr(p, "x") and hasattr(p, "y"):
            return float(p.x), float(p.y)
        # Fallback
        return float(p[0]), float(p[1])

    for d in drawings:
        for it in d.get("items", []):
            op = it[0]
            # Horizontal line: ('l', p1, p2, ...)
            if op == "l":
                p1, p2 = it[1], it[2]
                x0, y0 = _pt_xy(p1)
                x1, y1 = _pt_xy(p2)
                if abs(y0 - y1) <= 1.5:  # ~horizontal
                    width = abs(x1 - x0)
                    y = (y0 + y1) / 2.0
                    if width >= min_width and y >= min_y:
                        if best_y is None or y > best_y:
                            best_y = y
            # Very thin horizontal rect: can be ('re', Rect) OR ('re', x0, y0, x1, y1, ...)
            elif op == "re":
                rect = it[1]
                if isinstance(rect, fitz.Rect):
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                else:
                    # Some builds pack coords directly
                    try:
                        x0, y0, x1, y1 = float(it[1]), float(it[2]), float(it[3]), float(it[4])
                    except Exception:
                        continue
                height = abs(y1 - y0)
                width = abs(x1 - x0)
                y_mid = (y0 + y1) / 2.0
                if height <= 1.5 and width >= min_width and y_mid >= min_y:
                    if best_y is None or y_mid > best_y:
                        best_y = y_mid

    return best_y  # None if not found


# ----------------------- NEW HELPERS (table-aware) -----------------------------------
def _normalize_line_endings(text: str) -> str:
    """
    Preserve structure:
    - Convert CRLF/CR to LF
    - Strip trailing spaces per line
    - Do NOT collapse multiple newlines
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n"))

def _rect_intersects(a: fitz.Rect, b: fitz.Rect) -> bool:
    # be a bit generous; any overlap or containment should count as table overlap
    return a.intersects(b) or a in b or b in a

def _get_table_rects(page: fitz.Page) -> List[fitz.Rect]:
    """
    Use PyMuPDF's table finder if available to get bounding boxes for tables.
    """
    rects: List[fitz.Rect] = []
    try:
        tf = page.find_tables()
        if tf and getattr(tf, "tables", None):
            for t in tf.tables:
                # t.bbox is (x0, y0, x1, y1)
                rects.append(fitz.Rect(*t.bbox))
    except Exception:
        # Some builds of PyMuPDF may not have find_tables; fall back later
        pass
    return rects

def _looks_like_table_text(txt: str) -> bool:
    """
    Heuristic to filter table-like text if structural detection is unavailable:
      - many '|' or tabs
      - multiple columns created by runs of >=3 spaces on several lines
      - lots of very short tokens (IDs, codes) typical of tables
    """
    if not txt:
        return False
    if txt.count("|") >= 2 or txt.count("\t") >= 2:
        return True
    spaced_cols = sum(1 for line in txt.splitlines() if len(_re.findall(r"( {3,})", line)) >= 2)
    if spaced_cols >= 2:
        return True
    tokens = _re.findall(r"\w+", txt)
    if tokens and (sum(1 for t in tokens if len(t) <= 3) / max(1, len(tokens))) > 0.6:
        return True
    return False
# -------------------------------------------------------------------------------------

def _extract_page_text_with_layout(page: fitz.Page, cut_y: Optional[float]) -> str:
    """
    Build page text from layout blocks (top->bottom, left->right),
    preserving internal newlines, but **excluding**:
      - any blocks that overlap detected table regions
      - any blocks that fall below the footer cut line (if provided)
    If no tables are detected by the engine, use a heuristic to skip table-like blocks.
    """
    # Find table rectangles (if supported)
    table_rects = _get_table_rects(page)

    blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text, ...)
    blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
    kept_parts: List[str] = []

    for b in blocks:
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        t = b[4] if len(b) > 4 else ""
        if not t or not str(t).strip():
            continue

        # Drop anything below the detected border line
        if cut_y is not None and y0 >= cut_y:
            continue

        rect = fitz.Rect(x0, y0, x1, y1)

        # Exclude if this block intersects any detected table
        if table_rects and any(_rect_intersects(rect, tr) for tr in table_rects):
            continue

        # If we didn't detect tables, use a heuristic to filter table-like text
        if (not table_rects) and _looks_like_table_text(t):
            continue

        t = _normalize_line_endings(t)
        kept_parts.append(t)

    page_text = "\n".join(kept_parts)

    # Optional: collapse 3+ blank lines to double blanks (visual spacing only)
    page_text = _re.sub(r"\n{3,}", "\n\n", page_text)
    return page_text

def build_structured_documents(pdf_path: str) -> List[Document]:
    """
    Return a list of LangChain Documents (one per page initially),
    with page_content preserving the exact line breaks from the PDF,
    and with:
      - table text removed
      - all content BELOW the footer border line removed (text only)
    """
    doc = fitz.open(pdf_path)
    docs: List[Document] = []
    for pno in range(len(doc)):
        page = doc[pno]
        cut_y = _find_footer_cut_y_fitz(page)  # None if not found
        text = _extract_page_text_with_layout(page, cut_y)
        # Strict fallback: if nothing survived (page with only tables), keep it empty
        docs.append(Document(page_content=text, metadata={"page": pno + 1}))
    doc.close()
    return docs
# =====================================================================================


def extract_images_from_pdf(pdf_path, output_folder):
    """
    Extract ONLY images that are positioned **above** the detected footer border line.
    Uses page.get_text('rawdict') to access image blocks with bounding boxes.
    """
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_map: Dict[int, List[str]] = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        cut_y = _find_footer_cut_y_fitz(page)
        # if not found, allow entire page (use page height)
        if cut_y is None:
            cut_y = float(page.rect.height)

        # Get image blocks with bbox and xref
        try:
            raw = page.get_text("rawdict")
            blocks = raw.get("blocks", [])
        except Exception:
            blocks = []

        for b in blocks:
            # type 1 == image block
            if b.get("type") != 1:
                continue

            bbox = b.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = map(float, bbox)

            # Keep only images fully above the footer line
            if y1 >= cut_y:
                continue

            # xref is available as 'image' or 'xref' depending on PyMuPDF version
            xref = b.get("image") or b.get("xref")
            if not xref:
                continue

            try:
                base_image = doc.extract_image(int(xref))
            except Exception:
                continue

            image_bytes = base_image.get("image")
            image_ext = base_image.get("ext", "png")
            if not image_bytes:
                continue

            image_path = os.path.join(output_folder, f"page{page_index+1}_img{xref}.{image_ext}")
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_map.setdefault(page_index + 1, []).append(image_path)

    doc.close()
    return image_map

# === Function to handle PDF upload ===
def handle_pdf_upload(uploaded_file):
    pdf_name = uploaded_file.name
    base_name = re.sub(r'[^\w\-]', '_', os.path.splitext(pdf_name)[0])

    storage_dir = os.path.join("stored_pdfs", base_name)
    images_path = os.path.join(storage_dir, "images.pkl")
    tables_path = os.path.join(storage_dir, "tables.pkl")
    pdf_path = os.path.join(storage_dir, "original.pdf")
    extracted_images_dir = os.path.join(storage_dir, "images")
    extracted_tables_dir = os.path.join(storage_dir, "tables")
    collection_name = base_name.lower()

    os.makedirs(storage_dir, exist_ok=True)

    # Use same embedding model in all scripts
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(url="http://localhost:6333")  # Docker Qdrant

    # Check if collection already exists
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in existing_collections:
        # ✅ Load cached data from Qdrant
        st.session_state.vectors = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=st.session_state.embeddings,
        )
        with open(images_path, "rb") as f:
            st.session_state.image_map = pickle.load(f)
        with open(tables_path, "rb") as f:
            st.session_state.table_image_map = pickle.load(f)
        st.success(f"✅ Loaded cached data for: {pdf_name}")
        return

    # 🆕 Process and upload new PDF
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- TEXT: structured, table-free, and cropped above footer line ---
    page_docs = build_structured_documents(pdf_path)  # one Document per page, footer-cropped

    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=300)
    documents = splitter.split_documents(page_docs)
    for i, d in enumerate(documents):
        # keep page number from original metadata
        d.metadata["chunk_index"] = i
        d.metadata["page"] = d.metadata.get("page", 0)

    vector_size = len(st.session_state.embeddings.embed_query("test"))

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Upload vectors
    st.session_state.vectors = Qdrant.from_documents(
        documents,
        st.session_state.embeddings,
        url="http://localhost:6333",
        collection_name=collection_name
    )

    # --- IMAGES: only those ABOVE the footer line ---
    st.session_state.image_map = extract_images_from_pdf(pdf_path, extracted_images_dir)
    with open(images_path, "wb") as f:
        pickle.dump(st.session_state.image_map, f)

    # --- TABLES (as images): rely on your UI function; make sure that version crops above footer too ---
    st.session_state.table_image_map = extract_table_images_from_pdf(pdf_path, extracted_tables_dir)
    with open(tables_path, "wb") as f:
        pickle.dump(st.session_state.table_image_map, f)

    st.success(f"✅ Processed and cached new PDF: {pdf_name}")



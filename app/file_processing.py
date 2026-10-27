import os
from io import BytesIO

import pandas as pd
from pypdf import PdfReader
from docx import Document


def read_bytes(filename: str, content: bytes):
    ext = os.path.splitext(filename)[1].lower()
    if not content:
        raise ValueError(f"Empty file: {filename}")

    if ext == ".pdf":
        return [
            (f"Page {i}", page.extract_text() or "")
            for i, page in enumerate(PdfReader(BytesIO(content)).pages, 1)
        ]
    if ext == ".docx":
        doc = Document(BytesIO(content))
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                lines.append(" | ".join(cell.text for cell in row.cells))
        return [("Document", "\n".join(lines))]
    if ext in (".txt", ".md"):
        return [("Document", content.decode("utf-8", errors="replace"))]
    if ext == ".csv":
        return [("CSV", pd.read_csv(BytesIO(content)).fillna("").to_string(index=False))]
    raise ValueError(f"Unsupported file type: {ext}")


def read_file(path: str):
    with open(path, "rb") as file:
        return read_bytes(path, file.read())

import re

def clean_medical_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = "".join(ch for ch in text if ch.isprintable() or ch in ['\n', '\t'])
    return text.strip()

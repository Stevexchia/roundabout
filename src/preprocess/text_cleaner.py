"""Used for cleaning text data."""

import re
from typing import Optional

try:
    from ftfy import fix_text
except ImportError:
    fix_text = None


def basic_clean(text: str) -> str:
    """Lowercase, trim, collapse spaces, normalize repeated punctuation."""
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    # Fix encoding issues if ftfy is available
    if fix_text:
        text = fix_text(text)
    
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    text = re.sub(r"([!?.,]){2,}", r"\1", text)  # Normalize punctuation
    
    return text.strip()


def advanced_clean(text: str) -> str:
    """More aggressive cleaning for feature extraction."""
    text = basic_clean(text)
    
    #remove links & urls
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    #remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    #punctuations removal
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text.strip()
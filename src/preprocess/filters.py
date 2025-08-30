try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # For consistent results
except ImportError:
    detect = None

def is_english(text: str) -> bool:
    # check if text is in english, else if langdetect is unavailable, return True
    if not text or not isinstance(text, str):
        return False
    if detect is None:
        return True
    
    try:
        return detect(text) == 'en'
    except:
        return True
    
def is_valid_review(text: str, min_length: int = 10) -> bool:
    #checks if the review is valid based on length
    if not isinstance(text, str):
        return False
    if len(text.strip()) < min_length:
        return False
    
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count < min_length / 2: #at least half the characters should be alphabetic
        return False
    
    return True

def should_keep_review(text: str) -> bool:
    return is_english(text) and is_valid_review(text)
import pandas as pd

# Read the file with the encoding that works (try 'latin1' or 'utf-8-sig' if utf-8 fails)
df = pd.read_csv('data/labeled/reviews_validation.csv', encoding='latin1')

# Save it back as UTF-8
df.to_csv('data/labeled/reviews_validation_utf8.csv', index=False, encoding='utf-8')
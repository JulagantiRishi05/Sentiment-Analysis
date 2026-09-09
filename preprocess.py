import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources silently
for resource in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        pass

class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            nltk.download('wordnet', quiet=True)
            self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Punctuation & Special Characters Removal (keep letters and spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        
        # 3. Tokenization
        tokens = text.split()  # robust whitespace tokenization
        
        # 4. Stopword removal & Lemmatization
        clean_tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 1
        ]
        
        return " ".join(clean_tokens)

    def preprocess_series(self, series):
        return series.apply(self.preprocess)

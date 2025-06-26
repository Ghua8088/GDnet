from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
class TextManager:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = list(stopwords.words("english"))
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize,
            preprocessor=self.preprocess,
            ngram_range=(1, 2),
            stop_words='english',  
            max_features=1000,
        )
    def fit(self, texts):
        self.vectorizer.fit(texts)
    def preprocess(self, text_string):
        space_pattern = r'\s+'
        giant_url_regex = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            r'[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = r'@[\w\-]+'
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = re.sub(giant_url_regex, '', parsed_text)
        parsed_text = re.sub(mention_regex, '', parsed_text)
        return parsed_text
    def tokenize(self, word):
        tokens = re.split(r'[^a-zA-Z]+',word.lower())
        return [
            self.stemmer.stem(t) for t in tokens
            if t and t not in self.stopwords
        ]
    def transform(self, texts):
        return self.vectorizer.transform(texts)
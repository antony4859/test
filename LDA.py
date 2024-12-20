# Step 1: Load the dataset
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))  # Strip metadata
documents = newsgroups.data     # raw text strings

# Step 2: Preprocess the text using NLTK
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize     # a function that does this: "Hello, world!" -> ['hello', ',', 'world', '!']
from nltk.stem import WordNetLemmatizer     # convert words to base form, like all verbs to infinitives
    
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
print("hello")
def preprocess_text(text):
    """
    Clean, tokenize, remove stopwords, and lemmatize a document.
    """
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)      # removes numbers and punctuation etc from text
    # Lowercase and tokenize
    words = word_tokenize(text.lower())
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()        
    
    return [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]



# Preprocess all documents
processed_docs = [preprocess_text(doc) for doc in documents]

# Step 3: Create Dictionary and Corpus
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents
# example: [["cat", "dog"], ["dog", "mouse", "cat"]] -> {0: "cat", 1: "dog", 2: "mouse"}
dictionary = Dictionary(processed_docs)

# Filter out extreme tokens
dictionary.filter_extremes(no_below=15, no_above=0.5)  # Remove rare and very frequent words

# Create the Bag-of-Words (BoW) corpus
# ["cat", "dog", "cat"] -> [(0, 2), (1, 1)]. 0 is cat. 1 is dog. and so on. Same ints as the dictionary.
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Step 4: Train the LDA model
from gensim.models import LdaModel

# Train the LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, passes=10, random_state=42)

# Step 5: Display the topics
for idx, topic in lda_model.print_topics(num_topics=10, num_words=10):
    print(f"Topic {idx}: {topic}")

import nltk,json
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
stop_words = stopwords.words('english')
wn = nltk.WordNetLemmatizer()

def basic_preprocess(txt):
    """
    basic version of pre-processing
    1. simple_preprocess the original text
    2. stopwords removal
    3. lemmatize each word
    
    :params[in]: txt, a string
    :params[in]: cleaned_txt, string
    """

    l0=simple_preprocess(txt, deacc=True)
    l1 = [word for word in l0 if word not in stop_words]
    l2 = [wn.lemmatize(w) for w in l1]
    cleaned_txt = ' '.join(l2)
    return cleaned_txt

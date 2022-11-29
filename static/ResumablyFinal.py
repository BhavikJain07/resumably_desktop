import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer
from onnxruntime import InferenceSession
import onnx
import fitz
from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs
import string, nltk, re, pickle
from gramformer import Gramformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.special import expit, softmax

CONFIG = "static/model"
WEIGHTS = "static/model_v2.onnx"
vect = pickle.load(open("static/tfidf_vect.pkl", 'rb'))

##Text Preprocessing
def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end

def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

def pdf_extractor(filename):
  with fitz.open(filename) as doc:
    x = ""
    for page in doc:
      x += page.get_text()
  return x

def clean_text(x):
    stopwords = nltk.corpus.stopwords.words('english')
    # x = re.sub("s+"," ", x)
    # x = re.sub("[^-9A-Za-z ]", "" , x)
#     x = " ".join([i.lower() for i in x if (i not in string.punctuation and i not in stopwords and len(i) > 2)])
    return resolve_encodings_and_normalize(x)

def preprocess(resume_file, job_desc, grammar = False):
  x = pdf_extractor(resume_file)
  if grammar:
    corrected_sentences = gf.correct(x, max_candidates=1)
    return clean_text(x), clean_text(job_desc),corrected_sentences[0] 
  return clean_text(x), clean_text(job_desc),None 

##Model Setup, Forward Propagation, Tokenizer
def tokenize(text):
  x = tokenizer(text,return_tensors="np")
  x["input_ids"] = x["input_ids"].astype('int64')
  x["attention_mask"] = x["attention_mask"].astype('int64')
  return x

def detokenize(tokens):
  return tokenizer.decode(tokens['input_ids'][0], True)

def token2vocab(tokens):
  return tokenizer.convert_ids_to_tokens(tokens['input_ids'][0], True)

def model_setup(weights, config):
  tokenizer = AutoTokenizer.from_pretrained(config)
  session = InferenceSession(weights)
  return tokenizer, session

def model_forward(inputs):
  return session.run(output_names=["weighted_hs",'output', "labels", "weights", "last_hs"], input_feed=dict(tokenize(inputs)))

def scale_score(score):
  return (score +1)/2

##Algorithms
# This give the Similarity score between the resume and job description. It might go beyond 1.0 and below 0.0
def resume_job_d(resume_text, job_desc):
  output_a = model_forward(resume_text)[1][0]
  output_b = model_forward(job_desc)[1][0]
  score = 1 - cosine(output_a, output_b)
  # print('MODEL: ', score)
  return score

def resume_job_tfidf(resume_text, job_desc):
  score = 1 - cosine(vect.transform([resume_text]).toarray()[0], vect.transform([job_desc]).toarray()[0])
  # print('TFIDF: ', score)
  return  score

# final_similarity_score
def similarity_score(resume_text, job_desc):
  return scale_score(0.6 * resume_job_tfidf(resume_text, job_desc) + 0.4 * resume_job_d(resume_text, job_desc))

def phrase_matching(resume_text, job_desc):
  output_a = model_forward(resume_text)[4][0]
  output_b = model_forward(job_desc)[4][0]
  xy = output_a @ output_b.T
  # print(scale_score(np.dot(output_a[2], output_b[2])/(np.linalg.norm(output_a[2])*np.linalg.norm(output_b[2]))))
  xy = np.divide(xy.T,np.linalg.norm(output_a, axis = 1)).T
  xy = np.divide(xy,np.linalg.norm(output_b, axis = 1))
  # print(scale_score(xy[2,2]))
  return scale_score(xy)

# Why our program feel that your resume matches with these jobs? if it wrong Feedback? else someerror might happen in the resume
def selfAttention(text):
  # TODO: Self Attention
  return expit(np.concatenate(model_forward(text)[3][..., 0]))

# Which class does it belongs to
def classifier(text, TOPK = 10, func = expit):
  # TODO: dicting
  y = func(model_forward(text)[2])
  return y.argsort()[0][-TOPK:][::-1] , np.sort(y)[0][-TOPK:][::-1]

##Initializing
tokenizer, session = model_setup(WEIGHTS, CONFIG)
gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector
from pathlib import Path
import numpy as np
import time
import pandas as pd
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import vstack, hstack, coo_matrix, csr_matrix

def gen_tok_pattern():
	single_toks = ['<=', '>=', '<', '>', '\\?', '\\/=', '\\+=', '\\-=', '\\+\\+', '--', '\\*=', '\\+', '-', '\\*',
				   '\\/', '!=', '==', '=', '!', '&=', '&', '\\%', '\\|\\|', '\\|=', '\\|', '\\$', '\\:']

	single_toks = '(?:' + '|'.join(single_toks) + ')'

	word_toks = '(?:[a-zA-Z0-9]+)'

	return single_toks + '|' + word_toks


# Extract features
def extract_features(config, start_n_gram, end_n_gram, token_pattern=None, vocabulary=None):
	if config == 1:

		return TfidfVectorizer(stop_words=None, ngram_range=(1, 1), use_idf=False, max_df=1.0, min_df=2,
							   norm=None, smooth_idf=False, lowercase=False, token_pattern=token_pattern,
							   vocabulary=vocabulary, max_features=None)

	elif config == 2:
		return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=True, min_df=0.001,
							   norm='l2', token_pattern=r'\S*[A-Za-z]\S+', vocabulary=vocabulary)
	elif config < 6:
		return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_n_gram, end_n_gram), use_idf=False,
							   min_df=0.001, norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+',
							   vocabulary=vocabulary)

	return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_n_gram, end_n_gram), use_idf=True,
						   min_df=0.001, norm='l2', token_pattern=r'\S*[A-Za-z]\S+', vocabulary=vocabulary)


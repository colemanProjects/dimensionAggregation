# this python module takes a set of ad feedback comments
# and reduces the number of rows into a single summary
# we want to explore using text summarization techniques to aggregate dimension rows

# 1. Phrasal Segmentation
# - do not remove stop words
# 2. PhraseRank
#  - build a list of phrases
#  - calculate the cosign similarity between all phrases
#  - phrases that are very similar to each other can count as one representative phrase
#  - perform page rank to get the top x phrases
# 3. Decide how to show them
#  - if word phrases: keep performing phraseRank til under preferred length + then drop anything over max length, + comma separate
# phrase rank for
import argparse
import json
import numpy as np
import re
from nltk.corpus import brown, stopwords
from operator import itemgetter
from nltk.cluster.util import cosine_distance
import networkx as nx
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def pagerank(A, eps=0.0001, d=0.85):
  P = np.ones(len(A)) / len(A)
  while True:
    new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
    delta = abs((new_P - P).sum())
    if delta <= eps:
      return new_P
    P = new_P


def sentence_similarity(sent1, sent2, stopwords=None):
  if stopwords is None:
    stopwords = []

  sent1 = [w.lower() for w in sent1]
  sent2 = [w.lower() for w in sent2]

  all_words = list(set(sent1 + sent2))

  vector1 = [0] * len(all_words)
  vector2 = [0] * len(all_words)

  # build the vector for the first sentence
  for w in sent1:
    if w in stopwords:
      continue
    vector1[all_words.index(w)] += 1

  # build the vector for the second sentence
  for w in sent2:
    if w in stopwords:
      continue
    vector2[all_words.index(w)] += 1

  # One out of 5 words differ => 0.8 similarity
  # One out of 2 non-stop words differ => 0.5 similarity
  # 0 out of 2 non-stop words differ => 1 similarity (identical sentences)
  # Completely different sentences=> 0.0
  return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words=None):
  # Create an empty similarity matrix
  S = np.zeros((len(sentences), len(sentences)))

  for idx1 in range(len(sentences)):
    for idx2 in range(len(sentences)):
      if idx1 == idx2:
        continue

      S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

  # normalize the matrix row-wise
  for idx in range(len(S)):
    S[idx] /= S[idx].sum()

  return S

def textrank(sentences, top_n=10000, stop_words=None):
  """
  sentences = a list of sentences [[w11, w12, ...], [w21, w22, ...], ...]
  top_n = how may sentences the summary should contain
  stopwords = a list of stopwords
  """
  S = build_similarity_matrix(sentences, stop_words)
  sentence_ranks = pagerank(S)
  print sentence_ranks
  print sentences

  # Sort the sentence ranks
  ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
  print ranked_sentence_indexes
  summary = [sentence for _,sentence in sorted(zip(ranked_sentence_indexes,sentences))]
  return summary

def remove_special(text):
  special_characters = '[^a-z -_]'
  return re.sub(special_characters, '', text.lower())

# load comments and remove special characters
def get_sentences(file):
  with open(file) as json_data:
    sentences = [remove_special(obj["id"]).split(' ') for obj in json.load(json_data)]
    return sentences

# sentences argument is an array of words
# convert a list of word vectors into a list of phrase vectors
def get_phrases(sentence_list, max_phrase_length=10):
  phrase_list = []
  # for each word, concat with the previous two words and the next two words in a sliding window
  i = 0
  for sentence in sentence_list:
    i = i +1
    phrases = []
    for i, word in enumerate(sentence):
      #place current word in list
      phrases.append(word)

      #lookback window
      for prev_word_index in range(i - max_phrase_length,i):
        if 0 <= prev_word_index < len(sentence):
          subset = sentence[prev_word_index: i+1]
          phrase_list.append(' '.join(subset))

  print phrase_list
  return phrase_list

#dedup based on the cosign similarity
def dedup(phrases):
  # compare each phrase
  # if it is similar to any other phrase, decide to throw this phrase out or that one
  #for index, phrase in enumerate(phrases):
  #  if phrase

  # decide based on whichever is ranked higher, could take length into account too
  return phrases


"""
def text_rank(document):
  sentence_tokenizer = PunktSentenceTokenizer()
  sentences = sentence_tokenizer.tokenize(document)

  bow_matrix = CountVectorizer().fit_transform(sentences)
  normalized = TfidfTransformer().fit_transform(bow_matrix)

  similarity_graph = normalized * normalized.T

  nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
  scores = nx.pagerank(nx_graph)
  return sorted(((scores[i],s) for i,s in enumerate(sentences)),
                reverse=True)
"""
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file', required=True, help='input file')
  return parser.parse_args()

def main(args):
  sentences = get_sentences(args.input_file)
  phrases = get_phrases(sentences)
  ranked_sentences = textrank(sentences)
  ranked_phrases = textrank(phrases)
  deduped = dedup(ranked_phrases)
  for idx, phrase in enumerate(ranked_phrases):
    print("%s. %s" % ((idx + 1), ' '.join(phrase)))
  for idx, phrase in enumerate(ranked_sentences):
    print("%s. %s" % ((idx + 1), ' '.join(phrase)))

if __name__ == "__main__":
  main(parse_args())
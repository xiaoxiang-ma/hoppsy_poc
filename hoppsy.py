import streamlit as st
import pandas as pd
import numpy as np
import pickle

import hdbscan
import umap
from numpy.linalg import norm

from pyabsa.functional import ATEPCCheckpointManager
from sentence_transformers import SentenceTransformer
import tokenizers



# @st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None})
# def fetch_aspect_extractor():
#     aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english')
#     return aspect_extractor

# @st.cache
# def fetch_sentence_transformer():
#     sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
#     return sentence_transformer

# ###############################
# st.write("# Loading resources")
# ###############################

# sentence_transformer = fetch_sentence_transformer()
# st.success("sentence_transformer loaded")
# aspect_extractor = fetch_aspect_extractor()
# st.success("aspect_extractor loaded")
try:
    st.write('numpy ', np.__version__)
except:
    st.write('cannot print numpy version')
try:
    st.write('hdbscan ', hdbscan.__version__)
except:
    st.write('cannot print hdbscan version')
try:
    st.write('umap ', umap.__version__)
except:
    st.write('cannot print umap version')


# with open('aspect_extractor.pkl', 'rb') as file:
#     aspect_extractor = pickle.load(file)
# st.write("aspect_extractor loaded")

# with open('POC_sentence_transformer.pkl', 'rb') as file:
#     sentence_transformer = pickle.load(file)
# st.write("sentence_transformer loaded")


with open('POC_umap_reducer.pkl', 'rb') as file:
    umap_reducer = pickle.load(file)
st.success("umap_reducer loaded")


with open('POC_hdbscan_clusters.pkl', 'rb') as file:
    hdbscan_clusters = pickle.load(file)
st.success("hdbscan_clusters loaded")


corpus_clusters = pd.read_csv("POC_clusters.csv")
st.success("corpus_clusters loaded")

if st.button('check load'):
    st.write(aspect_extractor)
    st.write(sentence_transformer)

###############################
st.write("# Update topic categories")
###############################


# topics = st.text_input('Enter topic categories', 'service, food, appetizer, desserts, atmosphere, menu, price, staff, manager')
# topic_categories = topics.split(", ")

# st.write(topic_categories)

# ###############################
# st.write("# Enter reviews")
# ###############################

# sample_review = '''The cucumber cocktail was very refreshing, the pork in the hangover ramen was a bit hard to chew...
# The pistachio icecream was a delight! Service was ok'''
# user_review = st.text_area('Text to analyze', sample_review).split("\n")
# st.write(user_review)

# if st.button('Compute Insights'):

#     ###############################
#     # Assign clusters to topic categories
#     ###############################

#     def embed(model, sentences):
#         embeddings = model.encode(sentences, show_progress_bar=False)
#         return embeddings

#     def cosine_sim(A,B):
#         cosine = np.dot(A,B)/(norm(A)*norm(B))
#         return cosine


#     topic_categories_embeddings = embed(sentence_transformer, topic_categories)
#     clusters_embedding = embed(sentence_transformer, corpus_clusters.group.to_list())


#     # Find cosine similarities
#     similarities = []
#     for i in range(len(clusters_embedding)):
#         similarities.append([cosine_sim(phrase, clusters_embedding[i]) for phrase in topic_categories_embeddings])
#     similarities = pd.DataFrame(similarities, columns = topic_categories)
#     similarities['topic'] = similarities.idxmax(axis=1)
#     similarities['topic_strength'] = similarities.max(axis=1)
#     similarities = similarities.round(2)

#     # Find label strength and correct "-1"
#     label_strength = pd.concat([corpus_clusters, similarities[['topic','topic_strength']]], axis=1)

#     label_strength_dict = dict(zip(label_strength['label_st1'] ,label_strength['topic_strength']))
#     label_dict = dict(zip(label_strength['label_st1'] ,label_strength['topic']))
#     label_dict[-1] = "_Generic"



#     ###############################
#     # ABSA on reviews 
#     ###############################

#     atepc_result = aspect_extractor.extract_aspect(inference_source=user_review,
#                                                 save_result=False,
#                                                 print_result=True,  # print the result
#                                                 pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
#                                                 )

#     for i in range(len(atepc_result)):
#         atepc_result[i]['raw_text'] = user_review[i]

#     reviews_absa = []
#     for i in atepc_result:
#         for j in range(len(i['aspect'])):
#             reviews_absa.append([
#                 i['raw_text'],
#                 i['aspect'][j],
#                 i['sentiment'][j],
#                 i['confidence'][j]
#             ])
#     reviews_absa = pd.DataFrame(reviews_absa)
#     reviews_absa.columns = ['text', 'aspect', 'sentiment', 'confidence']
#     st.write(reviews_absa)

#     # embed aspects, umap dim reduce, and find which clusters they belong
#     inference_aspects = reviews_absa['aspect'].tolist()
#     inference_aspects_embeddings = embed(sentence_transformer, inference_aspects)
#     inference_aspects_embeddings_umap = umap_reducer.transform(inference_aspects_embeddings)
#     test_labels, strengths = hdbscan.approximate_predict(hdbscan_clusters, inference_aspects_embeddings_umap)

#     reviews_absa['topic'] = [label_dict[i] for i in test_labels]
#     reviews_absa['topic_strength'] = [label_strength_dict[i] for i in test_labels]
#     reviews_absa.confidence = reviews_absa.confidence.round(2)

#     # Prettify
#     df_display = reviews_absa.groupby(['text','sentiment']).agg(
#         aspect=('aspect', lambda x: list(x)),
#         aspect_topic=('topic', lambda x: list(x)),
#         sentiment_confidence=('confidence', lambda x: list(x)),
#         topic_confidence=('topic_strength', lambda x: list(x)),
#     )

#     st.write(df_display)


# Build that can import correctly with Python 3.8
# hdbscan==0.8.27
# numpy==1.23.1
# umap-learn==0.5.1
# streamlit==1.11.1
# pandas==1.2.4
# sentence-transformers==2.2.2
# pyabsa==1.16.4
# tokenizers==0.12.1
# protobuf==3.20.*

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
# import copy



# @st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None}, allow_output_mutation=True)
def fetch_aspect_extractor():
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english')
    return aspect_extractor

# @st.cache(allow_output_mutation=True)
def fetch_sentence_transformer():
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence_transformer

###############################
st.write("## Hoppsy ML POC")
st.write("Scenario: Buisness owners can have a break down of their reviews by different categories...")
###############################

with st.spinner('Loading sentence transformer'):
    sentence_transformer = fetch_sentence_transformer()

with st.spinner('Loading aspect extractor'):
    aspect_extractor = fetch_aspect_extractor()

with st.spinner('Loading dimension reducer'):
    with open('POC_umap_reducer.pkl', 'rb') as file:
        umap_reducer = pickle.load(file)

with st.spinner('Loading hdbscan clusters'):
    with open('POC_hdbscan_clusters.pkl', 'rb') as file:
        hdbscan_clusters = pickle.load(file)

with st.spinner('Loading corpus clusters'):
    corpus_clusters = pd.read_csv("POC_clusters.csv")
    
st.success("Resources loaded successfully")


###############################
st.write("#### Define Topic Categories")
###############################


topics = st.text_input('Enter a few topic categories here (Separated by commas & Command + Enter to apply):', 'service, food, atmosphere, menu, price, staff, manager')
topic_categories = topics.split(", ")

st.write(topic_categories)

###############################
st.write("#### Enter Reviews")
###############################

sample_review = '''Our server was fantastic and when he found out the wife loves roasted garlic and bone marrow, he added extra to our meal and another marrow to go!
I LOVED their mussels cooked in this wine reduction, the duck was tender, and their potato dishes were delicious.
Although I very much liked the look and sound of this place, the actual experience was a bit disappointing.
I just don't know how this place managed to served the blandest food I have ever eaten when they are preparing Indian cuisine.
The guys all had steaks, and our steak loving son who has had steak at the best and worst places said it was the best steak he's ever eaten.
We thought you'd have to venture further away to get good sushi, but this place really hit the spot that night.
Bland... Not a liking this place for a number of reasons and I don't want to waste time on bad reviewing.. I'll leave it at that...
They have a good selection of food including a massive meatloaf sandwich, a crispy chicken wrap, a delish tuna melt and some tasty burgers.
Great Subway, in fact it's so good when you come here every other Subway will not meet your expectations.
He was extremely rude and really, there are so many other restaurants I would love to dine at during a weekend in Vegas.
The service was a little slow , considering that were served by 3 people servers so the food was coming in a slow pace.'''
user_review = st.text_area('Enter some reviews here (Separated by newline/Enter & Command + Enter to apply):', sample_review).split("\n")
st.write(user_review)

if st.button('Compute Insights'):
    with st.spinner('Computing...'):

        ###############################
        # Assign clusters to topic categories
        ###############################

        def embed(model, sentences):
            embeddings = model.encode(sentences, show_progress_bar=False)
            return embeddings

        def cosine_sim(A,B):
            cosine = np.dot(A,B)/(norm(A)*norm(B))
            return cosine


        topic_categories_embeddings = embed(sentence_transformer, topic_categories)
        clusters_embedding = embed(sentence_transformer, corpus_clusters.group.to_list())


        # Find cosine similarities
        similarities = []
        for i in range(len(clusters_embedding)):
            similarities.append([cosine_sim(phrase, clusters_embedding[i]) for phrase in topic_categories_embeddings])
        similarities = pd.DataFrame(similarities, columns = topic_categories)
        similarities['topic'] = similarities.idxmax(axis=1)
        similarities['topic_strength'] = similarities.max(axis=1)
        similarities = similarities.round(2)

        # Find label strength and correct "-1"
        label_strength = pd.concat([corpus_clusters, similarities[['topic','topic_strength']]], axis=1)

        label_strength_dict = dict(zip(label_strength['label_st1'] ,label_strength['topic_strength']))
        label_dict = dict(zip(label_strength['label_st1'] ,label_strength['topic']))
        label_dict[-1] = "_Generic"



        ###############################
        # ABSA on reviews 
        ###############################

        atepc_result = aspect_extractor.extract_aspect(inference_source=user_review,
                                                    save_result=False,
                                                    print_result=True,  # print the result
                                                    pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                                    )

        for i in range(len(atepc_result)):
            atepc_result[i]['raw_text'] = user_review[i]

        reviews_absa = []
        for i in atepc_result:
            for j in range(len(i['aspect'])):
                reviews_absa.append([
                    i['raw_text'],
                    i['aspect'][j],
                    i['sentiment'][j],
                    i['confidence'][j]
                ])
        reviews_absa = pd.DataFrame(reviews_absa)
        reviews_absa.columns = ['text', 'aspect', 'sentiment', 'confidence']
        # st.write(reviews_absa)

        # embed aspects, umap dim reduce, and find which clusters they belong
        inference_aspects = reviews_absa['aspect'].tolist()
        inference_aspects_embeddings = embed(sentence_transformer, inference_aspects)
        inference_aspects_embeddings_umap = umap_reducer.transform(inference_aspects_embeddings)
        test_labels, strengths = hdbscan.approximate_predict(hdbscan_clusters, inference_aspects_embeddings_umap)

        reviews_absa['topic'] = [label_dict[i] for i in test_labels]
        reviews_absa['topic_strength'] = [label_strength_dict[i] for i in test_labels]
        reviews_absa.confidence = reviews_absa.confidence.round(2)

        # Prettify
        df_display = reviews_absa.groupby(['text','sentiment']).agg(
            aspect=('aspect', lambda x: list(x)),
            aspect_topic=('topic', lambda x: list(x)),
            sentiment_confidence=('confidence', lambda x: list(x)),
            topic_confidence=('topic_strength', lambda x: list(x)),
        ).reset_index()
        st.write("#### Discover Insights")
        st.write(df_display)


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

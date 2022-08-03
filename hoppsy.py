import streamlit as st
import pandas as pd
import numpy as np
import pickle

import hdbscan
# import umap
from numpy.linalg import norm

from pyabsa.functional import ATEPCCheckpointManager
from sentence_transformers import SentenceTransformer
import tokenizers
import copy
# from streamlit import caching


@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None}, allow_output_mutation=True)
# @st.experimental_singleton
def fetch_aspect_extractor():
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english')
    return aspect_extractor

@st.cache(allow_output_mutation=True)
# @st.experimental_singleton
def fetch_sentence_transformer():
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence_transformer

###############################
st.write("## Hoppsy ML POC")
st.write("__Demo__: Digests customer reviews and gathers sentiment insights on any defined topics.")
###############################


with st.spinner('Loading sentence transformer'):
    sentence_transformer = fetch_sentence_transformer()

with st.spinner('Loading aspect extractor'):
    aspect_extractor = fetch_aspect_extractor()

# cloned_output = copy.deepcopy(my_cached_function(...))

# cloned_output = copy.deepcopy(my_cached_function(...))
# with st.spinner('Loading dimension reducer'):
#     with open('POC_umap_reducer.pkl', 'rb') as file:
#         umap_reducer = pickle.load(file)

# with st.spinner('Loading hdbscan clusters'):
#     with open('POC_hdbscan_clusters.pkl', 'rb') as file:
#         hdbscan_clusters = pickle.load(file)

# with st.spinner('Loading corpus clusters'):
#     corpus_clusters = pd.read_csv("POC_clusters.csv")
    
st.success("Resources loaded successfully")


###############################
st.write("#### Define Topic Categories")
###############################


topics = st.text_input('Enter a few topic categories here (Separated by commas, Command + Enter to apply):', 'service, food, environment, menu, price, staff')
topic_categories = topics.split(", ")

st.write(topic_categories)

###############################
st.write("#### Enter Reviews")
###############################

sample_review = '''The sushi were very flavorful, and the wasabi was a delight. I wished there were more vegeterian options in the menu.
The vibe was really nice. The scents were pleasant and the waitress was friendly.'''
user_review = st.text_area('Enter some reviews here (Separated by newline, Command + Enter to apply):', sample_review).split("\n")
st.write(user_review)
st.write("#### Discover Insights")

if st.button('Compute Insights'):
    with st.spinner('Computing...'):
        
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

        def embed(model, sentences):
            embeddings = model.encode(sentences, show_progress_bar=False)
            return embeddings

        def cosine_sim(A,B):
            cosine = np.dot(A,B)/(norm(A)*norm(B))
            return cosine
        
        aspects = reviews_absa['aspect'].tolist()
        aspects_embeddings = embed(sentence_transformer, aspects)
        topic_categories_embeddings = embed(sentence_transformer, topic_categories)
        
        assignments = []
        for i in range(len(reviews_absa.aspect)):
            # print('--- ', reviews_absa.aspect[i])
            temp = []
            for j in range(len(topic_categories_embeddings)):
        #             print(topic_categories[j], cosine_sim(aspects_embeddings[i], topic_categories_embeddings_st1[j]))
                    temp.append(cosine_sim(aspects_embeddings[i], topic_categories_embeddings[j]).round(2))
            loc = np.argmax(temp)
        #     print("!")
            # print(loc, topic_categories[loc], max(temp))
            assignments.append([ topic_categories[loc], max(temp)])
        assignments = np.array(assignments)
        reviews_absa['topic'] = assignments[:,0]
        reviews_absa['topic_confidence'] = assignments[:,1]

        # st.write(reviews_absa)
        ###############################
        # Assign clusters to topic categories
        ###############################

        


        # clusters_embedding = embed(sentence_transformer, corpus_clusters.group.to_list())


        # Find cosine similarities
        # similarities = []
        # for i in range(len(clusters_embedding)):
        #     similarities.append([cosine_sim(phrase, clusters_embedding[i]) for phrase in topic_categories_embeddings])
        # similarities = pd.DataFrame(similarities, columns = topic_categories)
        # similarities['topic'] = similarities.idxmax(axis=1)
        # similarities['topic_strength'] = similarities.max(axis=1)
        # similarities = similarities.round(2)

        # Find label strength and correct "-1"
        # label_strength = pd.concat([corpus_clusters, similarities[['topic','topic_strength']]], axis=1)

        # label_strength_dict = dict(zip(label_strength['label_st1'] ,label_strength['topic_strength']))
        # label_dict = dict(zip(label_strength['label_st1'] ,label_strength['topic']))
        # label_dict[-1] = "_Generic"



        
        # # st.write(reviews_absa)

        # # embed aspects, umap dim reduce, and find which clusters they belong
        # inference_aspects = reviews_absa['aspect'].tolist()
        # inference_aspects_embeddings = embed(sentence_transformer, inference_aspects)
        # inference_aspects_embeddings_umap = umap_reducer.transform(inference_aspects_embeddings)
        # test_labels, _ = hdbscan.approximate_predict(hdbscan_clusters, inference_aspects_embeddings_umap)

        # reviews_absa['topic'] = [label_dict[i] for i in test_labels]
        # reviews_absa['topic_strength'] = [label_strength_dict[i] for i in test_labels]
        # reviews_absa.confidence = reviews_absa.confidence.round(2)

    # for i in reviews_absa.text.unique():
    #     st.markdown("""---""")
    #     st.write(f"##### {i}")
    #     for index, row in reviews_absa[reviews_absa.text == i].iterrows():
    #         st.write(f"   _'{row['aspect']}'_ was identified as _'{row['sentiment']}'_ contributing to the topic _'{row['topic']}'_ with topic confidence _'{row['topic_strength']}'_")

    for i in reviews_absa.text.unique():
        st.markdown("""---""")
        st.write(f"##### {i}")
        for index, row in reviews_absa[reviews_absa.text == i].iterrows():
            st.write(f"   _'{row['aspect']}'_ was identified as _'{row['sentiment']}'_ to the topic _'{row['topic']}'_ with topic confidence _'{row['topic_confidence']}'_")



    
    # st.write(reviews_absa)
    # st.write(df_display)



    del topic_categories_embeddings
    # del clusters_embedding
    # del similarities
    # del label_strength
    # del label_strength_dict
    # del label_dict 
    del atepc_result
    del reviews_absa
    del sentence_transformer
    del aspect_extractor
    del assignments
    del aspects_embeddings
    # del inference_aspects_embeddings_umap

if st.button('Start Over'):
    # st.caching.clear_cache()
    st.experimental_rerun()

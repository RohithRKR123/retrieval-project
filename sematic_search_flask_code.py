
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from flask_cors import CORS, cross_origin

from flask import Flask, request, jsonify
#import json
#import gzip
#import os

#import numpy as np
#from rank_bm25 import BM25Okapi

"""here one more function for the pdf extraction will be there"""

def encode_passages(df, bi_encoder):
    bi_encoder.max_seq_length = 512
    top_k = 128
    passages_with_context = []
    for i, row in df.iterrows():
        pdf_name_f = row['pdf_name']#.split('/')[-1]
        passage_context = {
            'pdf_name': pdf_name_f,
            'page_num': row['page_num'],
            'para_num': row['para_num']
        }
        passages_with_context.append((row['documents'], passage_context))
    corpus_embeddings = bi_encoder.encode([p[0] for p in passages_with_context], convert_to_tensor=True, show_progress_bar=True) ###
    passages = [p[0] for p in passages_with_context]
    return corpus_embeddings, passages, passages_with_context


def search(query, passages, passages_with_context, bi_encoder, cross_encoder, corpus_embeddings):
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    # question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  

    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    results_lstt = []
    
    for hit in hits[0:10]:
      passage_index = hit['corpus_id']
      passage_text = passages_with_context[passage_index][0].replace("\n", " ")
      passage_context = passages_with_context[passage_index][1]
      pdf_name = passage_context['pdf_name']
      page_num = passage_context['page_num']
      #print("[PDF: {}, Page: {}]\t{}".format(pdf_name, page_num, passage_text))
      results_lstt.append({"Pdf_name":pdf_name, "Page_name": page_num, "Content":passage_text})
    return results_lstt

# BASE_DRIVE= 'C:\\Users\\nimai\\OneDrive\\Documents\\code testing\\AbleTech\\'
csv_path =  "testing_eng_docs.csv"
df = pd.read_csv(csv_path, encoding='latin-1')#albatross.csv')

df.documents = df['documents'].astype(str)
df.documents = df['documents'].apply(lambda x: x.strip())
df1 = df

# Load the biencoder model
model_path1 =  'biencoder.pkl'
#bi_encoder = SentenceTransformer('efederici/mmarco-sentence-BERTino')
#with open(model_path1, 'wb') as f:
#    pickle.dump(bi_encoder,f)


with open(model_path1, 'rb') as f:
    loaded_model_bi_encoder = pickle.load(f)

# Load the crossencoder model
model_path = 'crossencoder.pkl'
with open(model_path, 'rb') as file:
    loaded_model_cross_encoder = pickle.load(file)

#corpus_embeddings1, passages, passages_with_context = encode_passages(df1, loaded_model_bi_encoder)

embed_path = 'embedding.pkl'
with open(embed_path,'rb') as fp:
   corpus_embeddings1 = pickle.load(fp)

with open('passages.pkl','rb') as fp:
   passages = pickle.load(fp)

with open('passages_with_context.pkl','rb') as fp:
   passages_with_context = pickle.load(fp)

loaded_model_bi_encoder.max_seq_length = 512    
top_k = 128

#%%
app = Flask(__name__)

@app.route('/search', methods=['POST'])
@cross_origin()
def search_endpoint():
    # Get the query from the request
    query = request.json['query']

    # Perform the search using the query
    search_results = search(query, passages, passages_with_context, loaded_model_bi_encoder, loaded_model_cross_encoder, corpus_embeddings1)

    # Return the search results as a JSON response
    return jsonify({'results': search_results})

if __name__ == '__main__':
    app.run(host="0.0.0.0",threaded=True,debug=True)


import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
import time
from nltk.metrics.distance import edit_distance
from nltk.metrics.distance import jaccard_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from  scipy.sparse import find




def remove_punctuations(X):
    '''Remove punctuations from input string'''
    translator = X.maketrans('', '', string.punctuation)
    return X.translate(translator)

def make_lower(X):
    # convert to lower case
    return X.lower()

def matching(Sample_slice,tf_idf_G, G_index, threshold):
    '''
      Matching procedure by using cosine similarity
      Inputs:
            Sample_slice: one row from the sparse matrix of Test table
            tf_idf_G: tf-idf of Ground Truth (sparse)
            G_index: column company_id from G '''
    # Get the index of non-zero element in Sample_slice (the ngrams that are present in this document)    
    # # v contains the nonzero columns    
    i,v, j =find(Sample_slice)
    # find the row index of ground truth that are not zeros (those groud truth documents that contain some of ngrams of test document)
    x,y,z = find(tf_idf_G[:,v])
    # take of slice of G (reduce the search space)
    sub_G = tf_idf_G[:,v]
    slice_G = sub_G[x,:]
    # compute similarity
    similarity_score = cosine_similarity(slice_G, Sample_slice[:,v], dense_output=True)
    # index of highest similarity
    max_index= similarity_score.argmax()
    # check if the highest similarity is greater or equal to threshold
    if similarity_score[max_index] >= threshold:
        return G_index[x[max_index]]
    else:
        return -1


def name_matching(path_G,path_Test):
    '''
        Inputs:
                path_G: path for Ground Truth csv in string format
                path_Test: path for Test csv in string format
    '''
    
    #Get the file name
    G_file = path_G.split('/')[-1] # the last term is the file name
    Test_file = path_Test
    
    Test = pd.read_csv(path_Test, sep = '|')
    G = pd.read_csv(path_G, sep = '|')
    # Make sure the index starts with 0 
    G.reset_index(drop = True)
    print('Files loaded')
    
    # remove all the punctuations
    G['name_no_punc'] = G['name'].apply(remove_punctuations)
    # after removing the punctuations, creating another one with only capital letters
    #G['name_capital'] =G['name_no_punc'].apply(remove_lowercase)
    G['name_no_punc'] = G['name_no_punc'].apply(make_lower)
    #G['name_no_punc_no_space'] = G['name_no_punc'].apply(remove_whitespace)

    # Same transformation for STrain
    Test['name_no_punc'] = Test['name'].apply(remove_punctuations)
    #Test['name_capital'] =Test['name_no_punc'].apply(remove_lowercase)
    Test['name_no_punc'] = Test['name_no_punc'].apply(make_lower)
    #Test['name_no_punc_no_space'] = Test['name_no_punc'].apply(remove_whitespace)
    print('Text Preprocessing completed')
    
    # Perpare answer sheet
    Matching_table = pd.DataFrame([],index = Test.index, columns=['company_id'])
    # Company_id from Ground Truth
    Company_id = G['company_id']
    # tfidf ground truth
    tfidf = TfidfVectorizer(min_df=1, analyzer='char', ngram_range = (3,3))
    tf_idf_G = tfidf.fit_transform(G['name_no_punc'])
    # transform Test
    tf_idf_Test = tfidf.transform(Test['name_no_punc'])
    print('tf-idf transformation completed')
    
    # predict one company at a time
    t = time.time()
    print('Matching process begins..')
    predict_index = []
    for i in range(Test.shape[0]):
        pred = matching(tf_idf_Test[i,:],tf_idf_G,Company_id, 0.9)
        predict_index.append(pred)
        
    timespent = time.time() - t
    print('Matching process ended, running time is {} seconds'.format(timespent)) 
    
    # produce answer sheet
    Matching_table['company_id'] = predict_index
    Matching_table.to_csv('Matching_table.csv', sep='|', encoding='utf-8',index = True, index_label = 'test_index')
    print('Matching_table.csv is saved.')
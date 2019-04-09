import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
import time
from nltk.metrics.distance import edit_distance
from nltk.metrics.distance import jaccard_distance


STrain = pd.read_csv('/Users/selda/Home_docs/Selda/Job Search/ING WBAA/DS assignment/STrain.csv', sep = '|')
G = pd.read_csv('/Users/selda/Home_docs/Selda/Job Search/ING WBAA/DS assignment/G.csv', sep = '|')
merge = STrain.merge(G, how = 'left', on = 'company_id', suffixes = ['STrain','R'])

# Remove punctuations and lowercase

def remove_punctuations(X):
    translator = X.maketrans('', '', string.punctuation)
    return X.translate(translator)

# remove lowercase for comparing initials, remove space directly
def remove_lowercase(X):
    translator = X.maketrans('', '', string.ascii_lowercase)
    return X.translate(translator).replace(' ','')

def remove_whitespace(X):
    # remove the space and make lowercase here
    return X.replace(' ', '')

def make_lower(X):
    # convert to lower case
    return X.lower()


# 1. Use name_no_punc to find the separate word in bag of word of G
     # if found, get the index and compare similarity with name_no_punc_no_space, 
        #stopping rule is the threshold of matching score, the threshold needs to be tuned 
     # if not found and there is only one word, very likely it is initials, 
       #then compare editing distance with name_captial, threshold needs to be tuned


#### Cross Validation Stage

def matching_procedure(G,STrain,Matrix,jd_threshold,ld_threshold):
    
    ### Round I
    # Default prediction
    STrain['predict'] = -1
    STrain['loss'] = 0
    Round2 = []
    #### Round 1 #########
    t = time.time()
    for i in STrain.index:
        # split the name into separate words
        words = STrain['name_no_punc'].loc[i].split()
        length = len(words)
        # Find the word in the bag of words of G
        found = False
        j = 0
        while (found == False) & (j < length):
            found = words[j] in Matrix.columns 
            j = j + 1

        if found:
            # index of the bag of words, begin round 1 procedure
            Index = Matrix[Matrix[words[j-1]]>=1]
            to_compare= G.loc[Index.index]
            for k in Index.index:
                # the input object of jaccard distance is not string, but set
                jd = jaccard_distance(set(to_compare.at[k,'name_no_punc_no_space']), set(STrain.at[i,'name_no_punc_no_space']))
                if jd <= jd_threshold:
                    STrain.at[i,'predict'] = k
                    break
            # compute the loss right here
            if (STrain.loc[i].at['predict'] != STrain.loc[i].at['company_id']) & (STrain.loc[i].at['predict'] == -1):
                STrain.at[i,'loss'] = 1
            elif STrain.at[i,'predict'] != STrain.at[i,'company_id']:
                STrain.at[i,'loss'] = 5
            else:
                STrain.at[i,'loss'] = 0
                     
        else:
            # stack the index for round 2 procedure
            Round2.append(i)
            
            # compute the loss right here
            if (STrain.at[i,'predict'] != STrain.at[i,'company_id']) & (STrain.at[i,'predict'] == -1):
                STrain.at[i,'loss'] = 1
            elif STrain.at[i,'predict'] != STrain.at[i,'company_id']:
                STrain.at[i,'loss'] = 5
            else:
                STrain.at[i,'loss'] = 0
    elapsed = time.time() - t
    print('time spent for computing Round 1', elapsed)        
    ### Round II
    #for v in Round2:
    #    for l in G.index:
    #        ld = edit_distance(G['name_capital'].loc[l],STrain['name_capital'].loc[v])
    #        if ld <= ld_threshold:
    #            STrain['predict'].loc[v] = l
    #            break
    #    
    #    # compute the loss right here
    #    if (STrain['predict'].loc[v] != STrain['company_id'].loc[v]) & (STrain['predict'].loc[v] == -1):
    #        STrain['loss'] = 1
    #    elif STrain['predict'].loc[v] != STrain['company_id'].loc[v]:
    #        STrain['loss'].loc[v] = 5
    #    else:
    #        STrain['loss'].loc[v] = 0
    
    outcome = STrain[['predict','loss']]
    return outcome, Round2




# remove all the punctuations
G['name_no_punc'] = G['name'].apply(remove_punctuations)
# after removing the punctuations, creating another one with only capital letters
G['name_capital'] =G['name_no_punc'].apply(remove_lowercase)
G['name_no_punc'] = G['name_no_punc'].apply(make_lower)
G['name_no_punc_no_space'] = G['name_no_punc'].apply(remove_whitespace)

# Same transformation for STrain
STrain['name_no_punc'] = STrain['name'].apply(remove_punctuations)
STrain['name_capital'] =STrain['name_no_punc'].apply(remove_lowercase)
STrain['name_no_punc'] = STrain['name_no_punc'].apply(make_lower)
STrain['name_no_punc_no_space'] = STrain['name_no_punc'].apply(remove_whitespace)

print('There are total', STrain[STrain['company_id'] == -1].shape[0], 'unmatched companies.')



vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(G['name_no_punc']).todense() 

# convert the results of bag of words in to dataframe
features = vectorizer.get_feature_names()
Matrix = pd.DataFrame(matrix, columns = features, index = G.index)

# Test matching_procedure
print('start testing matching  procedure')
Sample = STrain.sample(n=100,random_state=10)
outcome, Round2 = matching_procedure(G,Sample,Matrix,0.1,4)
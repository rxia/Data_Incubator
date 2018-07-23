import numpy as np
import pandas as pd
import pickle
import scipy as sp
from time import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from sklearn.manifold import TSNE
from textblob import TextBlob
from nltk.corpus import stopwords
stopwords_list = set(stopwords.words('english'))


## Load data
with open('project/data/scraped_data_all_recipes_without_link.pickle', 'rb') as f:
    data_allrecipes = pickle.load(f)
data_allrecipes = pd.DataFrame(data_allrecipes)
print('Data length before filtering {}'.format(data_allrecipes.shape[0]))
data_allrecipes.drop_duplicates('instruction', inplace=True)
shortest_instruction = 50
data_allrecipes = data_allrecipes[data_allrecipes['instruction'].str.len()>=shortest_instruction]
data_allrecipes.reset_index(drop=True, inplace=True)
data_allrecipes.drop(29554, inplace=True)
data_allrecipes.drop(29551, inplace=True) # id = 101923,102126, drop it because of duplicate (similar items)
data_allrecipes.reset_index(drop=True, inplace=True)
print('Data length after filtering {}'.format(data_allrecipes.shape[0]))

## Get 1-gram features from instructions
instruction_words, instruction_id = [], []
data_allrecipes['instruction'] = data_allrecipes['instruction'].str.replace(r'\d', '') # remove numbers
# word count feature
t0 = time()
count_vect = CountVectorizer(max_df=0.95, min_df=2, stop_words=stopwords_list)
X_train_counts = count_vect.fit_transform(data_allrecipes['instruction']).todense()
print('Word count feature extraction done ({} sec)'.format(time()-t0))
# word frequency (tdidf) feature
t0 = time()
tfidf_vect = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=stopwords_list)
X_train_tfidf = tfidf_vect.fit_transform(data_allrecipes['instruction']).todense()
print('Tf-idf feature extraction done ({} sec)'.format(time()-t0))


## Topic model analysis
n_components = 10
n_top_words = 30

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(X_train_tfidf)
print("NMF done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
tfidf_feature_names = count_vect.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# print("Fitting LDA models with tf features, "
#       "n_samples=%d and n_features=%d..."
#       % (n_samples, n_features))
# lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
#                                 learning_method='online',
#                                 learning_offset=50.,
#                                 random_state=0)
# t0 = time()
# lda.fit(X_train_counts)
# print("done in %0.3fs." % (time() - t0))
#
# print("\nTopics in LDA model:")
# tf_feature_names = count_vect.get_feature_names()
# print_top_words(lda, tf_feature_names, n_top_words)

# Assign topic index to each observation
topic = []
for i in range(tsne_result.shape[0]):
    topic.append(np.argmax(nmf.transform(X_train_tfidf[i,:])))
topic = np.array(topic)


## PCA to reduce the number of features
t0 = time()
pca = PCA(n_components=100)
pca_result = pca.fit_transform(X_train_tfidf)
print("PCA done in %0.3fs." % (time() - t0))

## t-SNE embedding
t0 = time()
tsne = TSNE()
tsne_result = tsne.fit_transform(pca_result)
print("t_SNE done in %0.3fs." % (time() - t0))

## Visualize
plt.figure()
ax = plt.gca()
small_sample_index = np.random.choice(tsne_result.shape[0],30000)
n_titles_per_cluster = 5
tsne_result_small = tsne_result[small_sample_index,:]
topic_small = topic[small_sample_index]
titles_small = data_allrecipes['title'][small_sample_index]
for i in np.unique(topic_small)[-1::-1]:
    X = tsne_result_small[topic_small==i,0]
    Y = tsne_result_small[topic_small==i,1]
    plt.scatter(X,Y,c=matplotlib.cm.get_cmap('Set2')(i),label='Cluster{}'.format(i),s=10)
    titles = np.array(titles_small[topic_small==i])
    for j in np.random.choice(X.shape[0],n_titles_per_cluster):
        title = titles[j]
        ax.annotate(title,xy=(X[j],Y[j]),textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=7),
            horizontalalignment='middle', verticalalignment='top')
plt.legend()
plt.axis('off')

## Word co-occurance pearson correlation
# n_top_words = 50
# feature_choice = [i for topic in nmf.components_ for i in topic.argsort()[:-n_top_words - 1:-1]]
# feature_choice = np.random.choice(X_train_tfidf.shape[1],100)
X_train_tfidf = np.array(X_train_tfidf)
feature_choice = np.arange(0,X_train_tfidf.shape[1])
Cov = np.cov(np.transpose(X_train_tfidf[:,feature_choice]))
Std = np.std(X_train_tfidf[:,feature_choice], axis=0)
Corr = Cov / ((Std[:, None]+0.01) * (Std[None, :]+0.01))
Corr[Corr<0.1] = 0
names = np.array(count_vect.get_feature_names())[feature_choice]
df = pd.DataFrame(data = Corr, columns = names, index = names)
df.to_csv('to_gephi.csv', sep = ',')

# The visualization was then performed in Gephi





## Ingredients
import json
with open('project/data/kaggle_category_ingredients.json') as f:
    data = pd.DataFrame(json.load(f))
##
ingredients_all = []
for i in range(data.shape[0]):
    for ingredient in data['ingredients'][i]:
        if ingredient[-2:] == 'es':
            ingredient = ingredient[:-2]
        elif ingredient[-1:] == 's':
            ingredient = ingredient[:-1]
        ingredients_all.append(ingredient)
ingredient_vocabulary = set(ingredients_all)


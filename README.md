# Word Representations in Vector Space
This work has been implemented with the **Continuous Skip-gram Model** using the Jupyter Notebook.

In the Skip-gram model each current word is an input to the log-linear classifier and the model predicts words within a context window (neighbourhood) before and after the current word. The input xk represents the one-hot encoded vector corresponding to the current word in the training sample. The words y1, . . . , yC are the one-hot encodings of the output context words. The input word is projected to the hidden layer of N neurons through a weight matrix W of dimensions V × N. The ith row of W with N dimensions represents the word vector of the ith word in the vocabulary. Each output vector yi has an associated weight matrix N × V represented as W 0 with softmax as the activation. The projection matrix acts as a lookup table since we are using one-hot encodings.

Writing all the functions of the Skip-gram model by hand would be enormous work. However, there is a **Gensim** library that helped for our implementation.
**Gensim** is the fastest library for training of vector embeddings – Python or otherwise. The core algorithms in **Gensim** use battle-hardened, highly optimized & parallelized C routines.
Also **Gensim** has a module `Word2Vec` by which we could initialize our model in Skip-gram.

```python
import gensim
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/imdb_master.csv', encoding= 'unicode_escape')
```
There you can see what kind of libraries we have imported, then we read the .csv file using pandas. The dataset we are using here is a subset of **IMDB reviews** with the size of 135MB. The data is stored as a *CSV file* and can be read using pandas.
Link to the Dataset: https://www.kaggle.com/utathya/imdb-review-dataset
Further, you can see that the review column is the main target column of our project as it contains all reviews, sentences which could be helpful for training the Skip-gram model.

```python
review = df.review.apply(gensim.utils.simple_preprocess)
```
Our data consists of 100000 rows. The first thing we did for any data science task was to clean the data. For NLP, we applied various processes like converting all the words to lowercase, trimming spaces, removing punctuations. Additionally, we could also remove stop words like 'and', 'or', 'is', 'the', 'a', 'an' and convert words to their root forms like 'running' to 'run'. And all of them could be made by only one ready module of `gensim.utils.simplepreprocess`. 

```python
model = gensim.models.Word2Vec(window=10, sg=1, min_count=2, workers=3 )
model.build_vocab(review, progress_per=1000)
model.train(review, total_examples=model.corpus_count, epochs=model.epochs)
model.save('/content/drive/MyDrive/word2vec-imdb-reviews.model')
```

Trained the model for reviews. Used a `window` of size 10 i.e. 10 words before the present word and 10 words ahead. A sentence with at least 2 words should only be considered, configure this using `min_count` parameter. Workers define how many CPU threads to be used and ***`sg`*** is Skip-gram, if we give 1 to it then model is Skip-gram, else model would be CBOW.
The training took over 15 minutes. Now all our words are converted to a vector in our model.








import streamlit as st
import pandas as pd
import urllib.request
import io
import transformers
import tensorflow as tf
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# Specify the URL of the CSV file
url = 'https://drive.google.com/uc?id=1DNXnMO8murxUXA0Ll_Y-T2fPjMgHSG3U'

# Download the CSV file
response = urllib.request.urlopen(url)
data = response.read()

# Convert the downloaded data into DataFrame
df = pd.read_csv(io.BytesIO(data))

# df.drop(['SKU'],axis=1,inplace=True)
df.drop(df.columns[[1,6,7,12,13,14,15]],axis=1,inplace=True)
df.drop_duplicates()
df.dropna(subset = ['REVIEW_CONTENT'], inplace = True)

# Page 1: Introduction
def page_introduction():
    st.title('Welcome to My Streamlit App')
    st.write('This is the introduction page of the Streamlit app.')
    st.title('EDA:')
    if st.button('shape'):
        st.write(df.shape)
    if st.button('describe the dataset'):
        st.write(df.describe())

    if st.button('Product category vs review count'):
        # Create bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(df['PRODUCT_CATEGORY'], df['REVIEW_COUNT'], color='skyblue')
        
        # Adding title and labels
        plt.title('Product Category vs Review Count')
        plt.xlabel('Product Category')
        plt.ylabel('Review Count')
        
        # Rotate x-axis labels for better readability if needed
        plt.xticks(rotation=45, ha='right')
        
        # Show plot
        plt.tight_layout()
        st.pyplot(plt)

    if st.button('Price vs Price Rating'):
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df['PRICE'], df['PRICE_RATING'], color='blue', alpha=0.5)
        
        # Adding title and labels
        plt.title('Price vs. Price Rating')
        plt.xlabel('Price')
        plt.ylabel('Price Rating')
        
        # Show plot
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)

# Page 2: Key Phrase Extraction
def key_phrase():
    st.title('Key Phrase Extration')
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=500)

    # Fit and transform the 'Review_content' column to TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['REVIEW_CONTENT'])

    # Get feature names (phrases)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Calculate TF-IDF scores for each feature (phrase)
    tfidf_scores = tfidf_matrix.sum(axis=0).A1

    # Create a dictionary to store phrases and their TF-IDF scores
    phrases_tfidf = dict(zip(feature_names, tfidf_scores))

    # Sort the phrases based on TF-IDF scores in descending order
    sorted_phrases_tfidf = sorted(phrases_tfidf.items(), key=lambda x: x[1], reverse=True)

    # Display the top 15 most frequent words/phrases
    top_phrases = sorted_phrases_tfidf[:25]
    if st.button('Display top 15 most frequent words:'):
        for phrase, score in top_phrases:
            st.write(f"{phrase}: {score}")
    phrases, scores = zip(*top_phrases)
    if st.button('Plot the top phrases'):
        plt.figure(figsize=(10, 6))
        plt.barh(phrases, scores, color='skyblue')
        plt.xlabel('TF-IDF Score')
        plt.ylabel('Phrases')
        plt.title('Top 15 Most Frequent Phrases with TF-IDF Scores')
        plt.gca().invert_yaxis()
        st.pyplot(plt)


    nltk.download('stopwords')
    nltk.download('punkt')
    from tqdm import tqdm
    from rake_nltk import Rake

    # Initialize RAKE
    r = Rake()

    # Initialize an empty list to store extracted key phrases
    key_phrases_list = []

    # Iterate over each review in the 'REVIEW_CONTENT' column
    for review in tqdm(df['REVIEW_CONTENT']):
        # Extract key phrases from the review
        r.extract_keywords_from_text(review)
        key_phrases = r.get_ranked_phrases()

        # Append key phrases to the list
        key_phrases_list.append(key_phrases)

    # Add the list of key phrases to the DataFrame
    df['KEY_PHRASES'] = key_phrases_list

    if st.button('Print key phrases:'):
        st.write(df['KEY_PHRASES'])


# Page 3: Topic Modelling
def topic_modelling():
    st.title('Topic Modelling')
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    from gensim.models import CoherenceModel

    
    # from prev topic
    nltk.download('stopwords')
    nltk.download('punkt')
    from tqdm import tqdm
    from rake_nltk import Rake

    # Initialize RAKE
    r = Rake()

    # Initialize an empty list to store extracted key phrases
    key_phrases_list = []

    # Iterate over each review in the 'REVIEW_CONTENT' column
    for review in tqdm(df['REVIEW_CONTENT']):
        # Extract key phrases from the review
        r.extract_keywords_from_text(review)
        key_phrases = r.get_ranked_phrases()

        # Append key phrases to the list
        key_phrases_list.append(key_phrases)

    # Add the list of key phrases to the DataFrame
    df['KEY_PHRASES'] = key_phrases_list

    
    # now new code starts
    # Create a dictionary representation of the key phrases
    key_phrases = df['KEY_PHRASES'].tolist()
    dictionary = Dictionary(key_phrases)

    # Convert the dictionary to a bag of words corpus
    corpus = [dictionary.doc2bow(phrase) for phrase in key_phrases]

    # Set the number of topics
    num_topics = 5

    # Build the LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

    if st.button('Print topic and top words for each topic'):
        # Print the topics and the top words for each topic
        for topic_num in range(num_topics):
            st.write(f"Topic #{topic_num + 1}:")
            st.write(lda_model.print_topic(topic_num, topn=10))
            st.write()

    # Calculate coherence score to evaluate the model
    coherence_model_lda = CoherenceModel(model=lda_model, texts=key_phrases, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    if st.button('Coherence Score'):
        st.write(f'\nCoherence Score: {coherence_lda}')


# Page 4: Sentiment analysis
def sentiment_analysis():
    st.title('Sentiment Analysis')
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.special import softmax
    from sklearn.metrics import accuracy_score
    from multiprocessing import Pool
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.special import softmax

    # Initialize the SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment_vader(review):
        sentiment_score = analyzer.polarity_scores(review)['compound']
        if sentiment_score > 0.05:
            return 1  # Positive
        elif sentiment_score < -0.05:
            return -1  # Negative
        else:
            return 0  # Neutral

    df['VaderSentiment'] = df['REVIEW_CONTENT'].apply(analyze_sentiment_vader)

    # Print the counts of each sentiment
    sentiment_counts = df['VaderSentiment'].value_counts()
    if st.button('Check counts of each sentiment:'):
        st.write("Positive:", sentiment_counts[1])
        st.write("Neutral:", sentiment_counts[0])
        st.write("Negative:", sentiment_counts[-1])

    if st.button('Visualize distribution of vader sentiment labels:'):
        plt.figure(figsize=(8, 6))
        plt.bar(['Positive', 'Neutral', 'Negative'], df['VaderSentiment'].value_counts(), color=['green', 'gray', 'red'])
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title('Vader Sentiment Analysis Results')
        st.pyplot(plt)  
    


# Main function to run the app
def main():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", ["Introduction", "Key Phrase Extraction", "Topic Modelling", "Sentiment Analysis"])

    if selection == "Introduction":
        page_introduction()
    elif selection == "Key Phrase Extraction":
        key_phrase()
    elif selection == "Topic Modelling":
        topic_modelling()
    elif selection == "Sentiment Analysis":
        sentiment_analysis()

if __name__ == "__main__":
    main()

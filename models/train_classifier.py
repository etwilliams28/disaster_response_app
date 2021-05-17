import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib
import sys
import nltk
import re

# import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def load_data(database_filepath):
    """ Load data from SQlite.
    
    Args:
    database_filename - file name of path in the database
    
    Returns:
    X = messages
    y = encoded categories 
    category_name = y.columns (list)
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messagedf', engine)
    X = df['message']
    y = df.drop(['id','message','original','genre'], axis=1)
    y = y.astype(int)
    category_names = y.columns.tolist()
    
    return X, y, category_names


def tokenize(text):
    """ Clean/tokenize messages
    
    Input:
        text - message for tokenization
    
    Returns: Cleand and tokenized list
    """
    
    text = re.sub(r'[^\w\s]','',text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
   
    pipeline = Pipeline([
        ('vec', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 100)))
    ])
    
    # grid search
    parameters = {'clf__estimator__max_features':['sqrt', 0.5],
              'clf__estimator__n_estimators':[50, 100]}

    cv = GridSearchCV(estimator=pipeline, param_grid = parameters, cv = 5, n_jobs = 10)
   
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    # predict on the X_test
    y_pred = model.predict(X_test)
    
    # build classification report on every column
    performances = []
    for i in range(len(category_names)):
        performances.append([f1_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             precision_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             recall_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro')])
    # build dataframe
    performances = pd.DataFrame(performances, columns=['f1 score', 'precision', 'recall'],
                                index = category_names)   
    return performances


def save_model(model, model_filepath):
    """
        Save model to pickle
    """
    joblib.dump(model, open(model_filepath, 'wb'))
   

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
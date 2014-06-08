pipeline = Pipline([





pipeline.fit()
pipeline.predict()

parameters = {
        'vect__max_df': (),
        'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1,1), (1,2)),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.001)
        }

grid_search = 
GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring="roc_auc")
# n_jobs = -1 tells grid search to use all cores on your computer, scoring will be kaggle's recommended
# score for the particular competition

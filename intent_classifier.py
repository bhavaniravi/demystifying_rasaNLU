from rasa_nlu.train import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.utils.spacy_utils import SpacyNLP
from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
import numpy as np, spacy

training_data = load_data("data/examples/rasa/demo-rasa.json")
config = RasaNLUModelConfig()
SpacyNLP(nlp=spacy.load("en")).train(training_data, config)
SpacyTokenizer().train(training_data, config)
SpacyFeaturizer().train(training_data, config)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

labels = [e.get("intent") for e in training_data.intent_examples]
le = LabelEncoder()

y = le.fit_transform(labels)
X = np.stack([example.get("text_features") for example in training_data.intent_examples])


defaults = {
        # C parameter of the svm - cross validation will select the best value
        "C": [1, 2, 5, 10, 20, 100],

        # the kernels to use for the svm training - cross validation will
        # decide which one of them performs best
        "kernels": ["linear"],

        # We try to find a good number of cross folds to use during
        # intent training, this specifies the max number of folds
        "max_cross_validation_folds": 5
}

# Create Classifier

C = defaults["C"]
kernels = defaults["kernels"]
tuned_parameters = [{"C": C, "kernel": [str(k) for k in kernels]}]

# aim for 5 examples in each fold
folds = defaults["max_cross_validation_folds"]
cv_splits = max(2, min(folds, np.min(np.bincount(y)) // 5))

clf = GridSearchCV( SVC(C=1, probability=True, class_weight='balanced'),
                    param_grid=tuned_parameters, n_jobs=1, cv=cv_splits,
                    scoring='f1_weighted', verbose=1)

clf.fit(X, y)

# Predict

X_test = training_data.intent_examples[24].get("text_features").reshape(1, -1)
pred_result = clf.predict_prob(X)
# sort the probabilities retrieving the indices of
# the elements in sorted order
sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
sorted_indices, pred_result[:, sorted_indices]
print (clf.predict_proba(X))
print (le.inverse_transform(y))

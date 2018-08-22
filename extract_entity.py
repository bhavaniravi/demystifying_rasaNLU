from rasa_nlu.train import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.utils.spacy_utils import SpacyNLP
from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
import spacy

config = RasaNLUModelConfig()
training_data = load_data("data/examples/rasa/demo-rasa.json")
SpacyNLP(nlp=spacy.load("en")).train(training_data, config)
SpacyTokenizer().train(training_data, config)

print (training_data.training_examples[25].as_dict())

crf = CRFEntityExtractor()
filtered_data = crf.filter_trainable_entities(training_data.training_examples)

# Create Dataset

# dataset = crf._create_dataset(filtered_data)

## Convert Examples


dataset = []

## Convert JSON TO CRF
for training_example in filtered_data:
    entity_offsets = crf._convert_example(training_example)
    print ("Entity Offset"  , entity_offsets)
    # b = crf._from_json_to_crf(training_example, entity_offsets)
    # print("JSON to CRF", b)
    ### _bilou tags from offset

    ents = crf._bilou_tags_from_offsets(training_example.get("tokens"), entity_offsets)
    print ("BILOU tags....", ents)
    text = crf._from_text_to_crf(training_example, ents)
    print ("TEXT to CRF", text)
    dataset.append(text)
#print (dataset)
# Token, POSTag, Entity, pattern(In case of regex features)

# Train Model

import sklearn_crfsuite
X_train = [crf._sentence_to_features(sent) for sent in dataset]
print("X_Train...", X_train[-1])
y_train = [crf._sentence_to_labels(sent) for sent in dataset]
print ("Y_Train.......", y_train[-1])

crf.ent_tagger = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                # coefficient for L1 penalty
                c1=crf.component_config["L1_c"],
                # coefficient for L2 penalty
                c2=crf.component_config["L2_c"],
                # stop earlier
                max_iterations=crf.component_config["max_iterations"],
                # include transitions that are possible, but not observed
                all_possible_transitions=True
        )
crf.ent_tagger.fit(X_train, y_train)

test_example = filtered_data[24]
test_example.data.pop("intent")
test_example.data.pop("entities")
print (test_example.as_dict())
text_data = crf._from_text_to_crf(test_example)
features = crf._sentence_to_features(text_data)

print (text_data)
print (features)
ents = crf.ent_tagger.predict_marginals_single(features)
print (ents)
print (crf._from_crf_to_json(test_example, ents))


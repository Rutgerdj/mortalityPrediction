from config import config
if config["biobert_path"] == "<path to biobert files>":
    raise Exception("Please change 'biobert_path' in config.py to the folder where you extracted the Biobert files")

from transformers import BertTokenizer, BertModel
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Dense, GlobalAveragePooling1D, add, concatenate
from tensorflow.keras.models import Model



def get_tokenizer():
    return BertTokenizer.from_pretrained(config["biobert_path"])

def get_bert_embed_matrix():
    bert = BertModel.from_pretrained(config["biobert_path"])
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat

def build_model(lstm_units = config["lstm_units"], max_length = config["max_length"], dropout = config["dropout"]):
    dense_hidden_units = 4 * lstm_units
    embeddings = get_bert_embed_matrix()
    words = Input(shape=(max_length,))
    x = Embedding(*embeddings.shape, weights=[embeddings], trainable=False)(words)
    x = SpatialDropout1D(dropout)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)

    hidden = concatenate([GlobalMaxPooling1D()(x),GlobalAveragePooling1D()(x),])
    hidden = add([hidden, Dense(dense_hidden_units, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=words, outputs=result)
    model.compile(loss="binary_crossentropy", metrics = ["accuracy"])
    return model

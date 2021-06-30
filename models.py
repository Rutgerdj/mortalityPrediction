from config import config
from transformers import BertTokenizer, BertModel
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Dense, GlobalAveragePooling1D, add, concatenate
from tensorflow.keras.models import Model



def get_tokenizer(pretrained_bert_tokenizer_path):
    return BertTokenizer.from_pretrained(pretrained_bert_tokenizer_path)

def get_bert_embed_matrix(pretrained_bert_model_path):
    bert = BertModel.from_pretrained(pretrained_bert_model_path)
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat

def build_model(lstm_units = config["lstm_units"], max_length = config["max_length"], dropout = config["dropout"],
                model_path = config["biobert_path"]):
    dense_hidden_units = 4 * lstm_units
    embeddings = get_bert_embed_matrix(model_path)
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

from config import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import models
import re


def split_and_tokenize_data(tokenizer, max_length=config["max_length"]):
    raw_data = pd.read_csv("traindata.csv")
    raw_data["found"] = raw_data.sentence.apply(lambda x: mort_regex.search(x) is not None)

    min_label = raw_data.label.value_counts().values[-1]
    num_test = int(min_label * 0.2)

    selected = raw_data.loc[raw_data.label == 1]
    not_selected = raw_data.loc[raw_data.label == 0]

    with_mort = not_selected.loc[not_selected.found]
    without_mort = not_selected.loc[not_selected.found == False]
    ''' Better:
    without_mort = not_selected.loc[not not_selected.found]
    But you should check it gives the same result'''

    test_df = selected.sample(n=num_test)
    train_df = selected.drop(index=test_df.index.values)

    test_with_mort = with_mort.sample(n=int(num_test / 2))
    test_without_mort = without_mort.sample(n=int(num_test / 2))

    train_with_mort = with_mort.drop(index=test_with_mort.index).sample(n=3 * num_test)
    train_without_mort = without_mort.drop(index=test_without_mort.index).sample(n=3 * num_test)

    test_df = test_df.append(test_with_mort)
    test_df = test_df.append(test_without_mort).sample(frac=1)

    train_df = train_df.append(train_with_mort)
    train_df = train_df.append(train_without_mort).sample(frac=1)

    assert all([x not in test_df.index for x in train_df.index.values]), "THERE IS OVERLAP IN TRAIN AND TEST DATA!"

    if tokenizer is None:
        tokenizer = models.get_tokenizer()

    X_train = np.array([tokenizer.encode(x, max_length=max_length, pad_to_max_length=True, truncation=True) for x in
                        train_df.sentence])
    X_test = np.array(
        [tokenizer.encode(x, max_length=max_length, pad_to_max_length=True, truncation=True) for x in test_df.sentence])

    y_train = np.array(train_df.label)
    y_test = np.array(test_df.label)

    return X_train, y_train, X_test, y_test


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

if __name__ == '__main__':

    if config["biobert_path"] == "<path to biobert files>":
        raise Exception("Please change 'biobert_path' in config.py to the folder where you extracted the Biobert files")


    mort_regex = re.compile(r"fatal|died|death|mortality", flags = re.IGNORECASE)



    model_tokenizer = models.get_tokenizer()
    model = models.build_model()

    X_train_model, y_train_model, X_test_model, y_test_model = split_and_tokenize_data(model_tokenizer)

    fit_history = model.fit(X_train_model,
                            y_train_model,
                            epochs = config["epochs"],
                            batch_size= config["batch_size"],
                            validation_data = (X_test_model, y_test_model))


    plot_graphs(fit_history, "accuracy")
    plot_graphs(fit_history, "loss")

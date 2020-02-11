import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# from sklearn.model_selection import GridSearchCV
import pickle

def read_dataset(path):
    sentences = []
    intents = []

    with open(path, "r") as f:
        lines = f.read().splitlines()

    # Get the string and intent out of the line. Super tricky yet performance friendly.
    # Noticed that some item contains multiple labels, And AFAIK that RF does not support this feature.
    # So I guess we have to throw away some labels here.
    for line in lines:
        # Make sure this is not an empty line.
        if len(line) == 0:
            continue

        sentences.append(line[4:line.index("EOS")-1])
        intents.append(line[line.index("atis_"):].split("#")[0])

    print("Done reading \"%s\" with %d line(s) of item" % (path, len(sentences)))
    
    return sentences, intents

def main():
    train_data_path = "./datasets/atis.train.w-intent.iob"
    test_data_path = "./datasets/atis.test.w-intent.iob"
    
    # Read datasets.
    # Train set has no label 'atis_day_name'.
    # Test set has no label 'atis_cheapest' 'atis_restriction'.
    train_sentences, train_intents = read_dataset(train_data_path)
    test_sentences, test_intents = read_dataset(test_data_path)
    print("Read %d classes of intentions" % len(set(train_intents + test_intents)))

    # Pack some words into a bag.
    print("Making BoW...")
    vectorizer = TfidfVectorizer(stop_words="english")
    train_features = vectorizer.fit_transform(train_sentences)
    test_features = vectorizer.transform(test_sentences)

    # Train RF model.
    # The parameters below was found in a grid search on a Ryzen 3700X in 60.1 minutes,
    # with an accuracy of 0.9376596831885539
    print("Start training RF model...")
    forest = RandomForestClassifier(
        n_estimators=240,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
    forest = forest.fit(train_features, train_intents)

    # Evaluate the model.
    print("Train Accuracy: %f" % forest.score(train_features, train_intents))
    print("Test Accuracy: %f" % forest.score(test_features, test_intents))

    # Save model for evaluation.
    if not os.path.exists("./models"):
        os.mkdir("./models")
    pickle.dump(vectorizer, open("./models/vectorizer.pkl","wb"))
    pickle.dump(forest, open("./models/model.pkl","wb"))

    # Un-comment this part and line 7 if you want to find a better hyper-parameters.
    # # Create the parameter grid.
    # param_grid = {
    #     'n_estimators': range(50,301,10),
    #     'max_features':["auto", "sqrt", "log2"],
    #     'max_depth': [None, 3, 5, 8, 15, 25, 30, 50],
    #     'min_samples_split': [2, 5, 10, 15, 30, 50],
    #     'min_samples_leaf': [1, 2, 5, 10, 30, 50]
    # }
    # # Instantiate the grid search model.
    # grid_search = GridSearchCV(estimator=forest, param_grid=param_grid,
    #                         cv=3, n_jobs=6, verbose=1)

    # # Make a bigger set for cross-validate.
    # train_sentences += test_sentences
    # train_features = vectorizer.fit_transform(train_sentences)
    # train_intents += test_intents

    # print("Note: Here may return UserWarning because we have only two or less sample in some classes")
    # grid_search.fit(train_features, train_intents)
    # print(grid_search.best_params_)
    # print(grid_search.best_score_)


if __name__ == "__main__":
    main()
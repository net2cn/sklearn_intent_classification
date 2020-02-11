import os, argparse
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    parser = argparse.ArgumentParser(description="Inference script for determining intentions of ATIS dataset's sentences.")
    parser.add_argument("--sentence", "-s", dest="sentence", type=str,
                        help="Single sentence with no punctuation mark.")

    args = parser.parse_args()

    # Check if models exists
    if not os.path.exists("./models"):
        print("WARNING: \"models\" folder does not exist! This means that a valid model for classification not exists!")
        return

    # Load pretrain model.
    vectorizer = pickle.load(open("./models/vectorizer.pkl", "rb"))
    forest = pickle.load(open("./models/model.pkl", "rb"))

    # See if text has "BOS" tag.
    text = args.sentence
    if text[0:3] == "BOS":
        text = text[4:text.index("EOS") if "EOS" in text else None]
    print("Classifying sentence \"%s\"" % text)

    # Do transformation and inference.
    feature = vectorizer.transform([text])
    result = forest.predict_proba(feature)

    # Print result.
    print("Model returned \"%s\" with a confidence of %f" % (forest.classes_[np.argmax(result)], np.max(result)))

if __name__ == "__main__":
    main()
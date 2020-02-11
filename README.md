# sklearn_intent_classification
An implementation of intent classification using random forest model on ATIS datasets with scikit-learn.

Datasets downloaded from [JointSLU](https://github.com/yvchen/JointSLU)

# Usage
If you have "models" folder in the root directory, just run:
```
python run_classifier.py -s "[single_cleaned_sentence_to_classify]"
```
Else you will have to train a model first with:
```
python train_classifier.py
```
This will train a usable model and save it in "models" folder. (Make sure you have placed datasets correctly.)

# Known issue
Given test dataset contains intention "atis_day_name" that train dataset does not contain. This could lead to wrong classification.

I suggest getting a bigger and more even dataset if possible to fix this issue.

2020, net2cn.
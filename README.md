Restaurant Review Model

What we want to achieve?
Let's say we have 1000s of restaurant reviews and we want to have a ML model
to tell us that this is a positive review or a negative review.

Steps performed:
1. We imported a dataset of restaurant review to train our model.
2. Preprocessing of data: We removed the unneccessary commom words which doesn't make sense in prediction. Also, changed the words to their root form using PorterStemmer.
3. Feature Extraction: Using sklearn library, we converted text data into numberical format.
4. Splitting data in 2 sets: train dataset and test dataset.
5. Then, We trained our classification model using GaussianNB which is a naive bayes classifier.
6. Then we predicted the sentiments of reviews present in our test dataset.
7. And in last we checked our prediction using confusion matrix to check the number of true positive, true negative, false positive, false negative.

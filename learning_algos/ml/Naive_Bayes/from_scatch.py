import pandas as pd
data = {
    'Temperature': ['Hot', 'Hot', 'Cold', 'Hot', 'Cold', 'Cold', 'Cold'],
    'Humidity': ['High', 'High', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Weather': ['Sunny', 'Sunny', 'Snowy', 'Rainy', 'Snowy', 'Snowy', 'Sunny']
}


# TODo - implement  naive_bayes_classifier, calculate_likelihoods_with_smoothing, calculate_prior_probabilities

def calculate_prior_probabilities(y):
    prior = {}
    for x in y:
        prior[x] = prior.get(x, 0)+1
    return {x : prior[x]/len(y) for x in prior.keys()}

def calculate_likelihoods_with_smoothing(data_dict):

    y = data_dict["Weather"]
    del data_dict["Weather"]
    classes = list(set(y))
    log_likelihoods = {}
    for column in data_dict.keys():
        log_likelihoods[column] = {}

        unique_count = len(set(data_dict[column]))
        for class_ in classes:
            log_likelihoods[column][class_] = {}
            index = [i for i in range(len(y)) if y[i]==class_]
            features = [data_dict[column][i] for i in index]
            count = len(features)
            for feature in list(set(features)):
                feat_count = features.count(feature)
                log_likelihoods[column][class_][feature] = (feat_count+ 1)/(count+ unique_count)

    return log_likelihoods

def naive_bayes_classifier(test_set, likelihoods_, prior):
    predictions = []
    for test in test_set:
        class_prob = {}
        for class_ in prior:
            class_prob[class_] = prior[class_]
            for column in test.keys():
                feat_prob = likelihoods_[column][class_]
                class_prob[class_]  *= feat_prob.get(test[column], 1/(len(feat_prob)+ 1))

        # Predict class with maximum posterior probability
        predictions.append(max(class_prob, key=class_prob.get))
    return predictions





# Calculate prior probabilities
priors = calculate_prior_probabilities(data["Weather"])

# Calculate likelihoods with smoothing
likelihoods = calculate_likelihoods_with_smoothing(data)

# print(likelihoods)
# New observation
X_test = [{'Temperature': 'Cold', 'Humidity': 'Normal'}]

# Make prediction
prediction = naive_bayes_classifier(X_test, likelihoods, priors)
print("Predicted Weather: ", prediction[0])  # Output: Predicted Weather:  Snowy


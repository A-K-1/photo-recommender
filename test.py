import numpy as np
import pandas as pd
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score,f1_score
from sklearn.model_selection import cross_val_score
from sklearn import tree


def train_tree():
    # train a tree for each column of location data(3 trees) with the same test and training sets

    column_headers = list(labels.columns.values)
    for idx, column in enumerate(labels):
        column_marker = column_headers[0]
        if idx == 1:
            column_marker = column_headers[1]
        elif idx == 2:
            column_marker = column_headers[2]

        y_train_single = y_train.loc[:, [column_marker]]
        y_test_single = y_test.loc[:, [column_marker]]

        # train the classifier
        classifier = RandomForestClassifier(n_estimators=20, criterion='entropy')
        # print(np.mean(cross_val_score(classifier, X_train, y_train_single.values.ravel(), cv=2)))
        classifier.fit(X_train, y_train_single.values.ravel())

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # combine predicted labels with the user name (converted from numbers to strings)
        predicted_location_names = []
        predicted_location_numbers = []
        for idx2, prediction in enumerate(y_pred):
            location_array = []
            location_array_numbers = []
            user_number = X_test.iloc[idx2, 0]

            # find name corresponding to user number
            user_name = user_data.loc[user_data['User Number'] == user_number, 'Full Name'].iloc[0]
            location_array.append(user_number)  # change user_number to user_name if you want to see the name
            location_array_numbers.append(user_number)

            # find name corresponding to spot number
            location_name = labels_names.loc[labels_names['Spot Number'] == prediction, 'Spot Name'].iloc[0]
            location_array.append(location_name)
            location_array_numbers.append(prediction)

            # calculate relevance factor of prediction
            relevance_score = round(relevance_factor(user_number, prediction),3)
            location_array.append(relevance_score)

            # calculate accuracy of prediction
            accuracy_single = calculate_accuracy_single(user_number, prediction)
            location_array.append(accuracy_single)

            # all_types = user_row.iloc[:, 2:15].values

            predicted_location_names.append(location_array)
            predicted_location_numbers.append(location_array_numbers)

        # output results to a dataFrame
        data = predicted_location_names
        data_numbers = predicted_location_numbers

        final_predictions.append(data)
        final_predictions_numbers.append(data_numbers)


def calculate_accuracy_single(user_number, prediction):
    # calculate the accuracy of a single prediction
    row_data = y_test.loc[[user_number], :]  # get row from the target data
    is_correct = prediction in row_data.values  # check if prediction appears in the row

    if is_correct:
        accuracy_single = 1
    else:
        accuracy_single = 0

    return accuracy_single


def relevance_factor(user_number, location_number):

            # get location details, relevant to the current prediction
            location_list = retrieve_location_info(location_number)
            location_types = location_list[0]  # split into types only
            location_tags = location_list[1]  # split into tags only

            # get user details, relevant to the current prediction
            user_list = retrieve_user_info(user_number)
            user_types = user_list[0]  # split into types only
            user_tags = user_list[1]  # split into tags only

            # get number of types that are the same between location and user
            same_types = list(set(user_types).intersection(location_types))
            number_same_types = len(same_types)

            # get number of tags that are the same between location and user
            same_tags = list(set(user_tags).intersection(location_tags))
            number_same_tags = len(same_tags)

            # calculate relevance score
            relevance_types = number_same_types / len(user_types)
            relevance_tags = number_same_tags / len(user_tags)
            relevance_score = relevance_types #  * relevance_tags - relevance score should not be determined by tags

            return relevance_score


def retrieve_user_info(predictions):

    user_types = []
    user_tags = []
    final_output = []

    # get whole row for relevant user
    user_row = user_data.loc[user_data['User Number'] == predictions]

    all_types = user_row.iloc[:, 2:15].values  # get photography type values
    all_tags = user_row.iloc[:, 15:211].values  # get photography tag values

    for sublist in all_types:
        for idx, types in enumerate(sublist):
            # add types that are == 1 to list
            if types == 1:
                user_types.append(headers_types[idx])

    for sublist in all_tags:
        for idx, tags in enumerate(sublist):
            # add tags that are == 1 to list
            if tags == 1:
                user_tags.append(headers_tags[idx])

    final_output.append(user_types)
    final_output.append(user_tags)
    return final_output


def retrieve_location_info(predictions):
    location_types = []
    location_tags = []
    final_output = []

    # get whole row for relevant location
    location_row = labels_names.loc[labels_names['Spot Number'] == predictions]

    all_types = location_row.iloc[:, 2:15].values  # get photography type values
    all_tags = location_row.iloc[:, 15:211].values  # get photography tag values

    all_types[0][2] = int(all_types[0][2])  # ensure all values are integers

    for sublist in all_types:
        for idx, types in enumerate(sublist):
            # add types that are == 1 to list
            if types == 1:
                location_types.append(headers_types[idx])

    for sublist in all_tags:
        for idx, tags in enumerate(sublist):
            # add tags == 1 to list
            if tags == 1:
                location_tags.append(headers_tags[idx])

    final_output.append(location_types)
    final_output.append(location_tags)
    return final_output


def average_accuracy():
    # calculate the average accuracy of predictions for each user

    iterations = len(final_predictions[0])

    for i in range(0, iterations):
        total_accuracy = 0
        for j in range(0, 3):

            # get accuracy
            total_accuracy += final_predictions[j][i][3]

        accuracy.append(round(total_accuracy / 3, 3))


def model_accuracy():
    # calculate the average accuracy of predictions for each user

    iterations = len(accuracy)
    total_accuracy = 0

    for i in range(0, iterations):

        total_accuracy += accuracy[i]

    print('\nModel Accuracy',
          round(total_accuracy / len(final_predictions[0]), 3))


def average_relevance():
    # calculate the average relevance of predictions for each user

    iterations = len(final_predictions[0])

    for i in range(0, iterations):
        total_relevance = 0
        for j in range(0, 3):

            # get accuracy
            total_relevance += final_predictions[j][i][2]

        avg_relevance.append(round(total_relevance / 3, 3))


def model_relevance():
    # calculate the average relevance of predictions for each user

    iterations = len(avg_relevance)
    total_relevance = 0

    for i in range(0, iterations):
        total_relevance += avg_relevance[i]

    print('\nModel Relevance',
          round(total_relevance / len(final_predictions[0]), 3))


# load user data which has been normalised
user_data = pd.read_csv("User Data Normalised.csv")
headers_types = list(user_data.iloc[:, 2:15].columns)
headers_tags = list(user_data.iloc[:, 15:211].columns)
model_data = user_data.iloc[:, 1:]

# load normalised labels
labels = pd.read_csv("Labels.csv")
labels.columns = ['spot1', 'spot2', 'spot3']

# load label names with their corresponding number
labels_names = pd.read_csv("Location Data Simple.csv")

# split data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(model_data, labels, test_size=0.08)

# array to store all predictions
final_predictions = []
final_predictions_numbers = []
accuracy = []
avg_relevance = []
result = []

# train tree and calculate accuracy
train_tree()
average_accuracy()
average_relevance()

# output predictions in a readable format
df = pd.DataFrame(final_predictions)
df_transpose = df.T
df_final = df_transpose.assign(Accuracy=accuracy, Avg_Relevance=avg_relevance)
print(df_final.to_string())

# print model accuracy & relevance
model_accuracy()
model_relevance()

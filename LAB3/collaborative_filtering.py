"""
Authors: Krzysztof Skwira & Tomasz Lemke
To run program install
pip install numpy
"""

import argparse
import json
import csv
import numpy as np

from compute_scores import euclidean_score


# Capturing the user name as an argument for the later use
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find users who are similar to the input user')
    parser.add_argument('--user', dest='user', required=True,
                        help='Input user')
    return parser


# Finds users in the dataset that are similar to the input user
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    # Compute Pearson score between input user 
    # and all the users in the dataset
    scores = np.array([[x, euclidean_score(dataset, user,
                                           x)] for x in dataset if x != user])

    # Sort the scores in decreasing order
    scores_sorted = np.argsort(scores[:, 1])[::-1]

    # Extract the top 'num_users' scores
    top_users = scores_sorted[:num_users]

    return scores[top_users]


# Specifying the input files
csvFilePath = 'ratings_csv.csv'
jsonFilePath = 'ratings_json.json'


# Reading CSV file
with open(csvFilePath, mode='r', encoding='utf-8-sig') as file:
    # reading the CSV file
    csvFile = csv.DictReader(file, delimiter=';')
    movies = {}

    # preparing the CSV file to json format
    for lines in csvFile:
        item = movies.get(lines['User'], dict())
        item[lines['Movie']] = int(lines['Rating'])
        movies[lines['User']] = item

# writing the movies into json file
with open(jsonFilePath, 'w', encoding='utf-8') as jsonFile:
    jsonFile.write(json.dumps(movies, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    # reading Json
    with open(jsonFilePath, 'r', encoding='utf-8-sig') as f:
        data = json.loads(f.read())

    # finding similarities between selected user and possible users from Json list
    print('\nUsers similar to ' + user + ':\n')
    similar_users = find_similar_users(data, user, 1)
    print('User\t\t\tSimilarity score')
    print('-' * 41)

    # Printing out the results
    for item in similar_users:
        print(item[0], '\t\t', round(float(item[1]), 2))
        print('\n')
        if item[0] in data:
            # sorting movies for the current similar user from best rated to worst rated
            data[item[0]] = dict(sorted(data[item[0]].items(), reverse=True, key=lambda it: it[1]))
            print("All movies for the similar user:")
            print(data[item[0]])
            print('\n')

            # removing the movies seen by the input user
            unique_movies = {k: v for k, v in data[item[0]].items() if k not in data[user]}
            print("Movies not seen by the input user:")
            print(unique_movies)
            print('\n')

            # listing the top 5 recommended unique movies
            print("Top 5 recommended movies:")
            top5 = list(unique_movies.keys())[:5]
            for i in top5:
                print(i)

            print('\n')

            # listing the 5 not recommended unique movies
            print("5 movies NOT recommended:")
            worst5 = list(unique_movies.keys())[-5:]
            for i in worst5:
                print(i)

            print('\n')

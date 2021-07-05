import os

import numpy as np
import pandas as pd
from scipy import stats
from pprint import pprint
import operator
import random
import statistics

fields = ["number_of_holes", "horizental_symetry", "vertical_symetry", "mean_vtd",
          "stddev_vtd", "mean_htd", "stddev_htd", "bottom_line_density", "hu7", "Class"]

singletons = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '']

classes = {
    'number_of_holes': ['123457', '069', '8'],
    'horizental_symetry': ['5', '7', '4', '6', '02389', '1236'],
    'vertical_symetry': ['46', '179', '13', '279', '024578'],
    'mean_vtd': ['0147', '047', '023476', '235689'],
    'stddev_vtd': ['4', '0147', '017', '236', '5689', '235689'],
    'mean_htd': ['0', '12', '12357', '3456', '345', '69', '45', '7', '17', '08', '689', '2345'],
    'stddev_htd': ['0689', '35', '2345', '123457', '147', '1247', '7'],
    'bottom_line_density': ['12', '0345689', '7']
}


def precision_recall_fmeasure(conf_mat):
    true_pos = np.diag(conf_mat1)
    false_pos = np.sum(conf_mat1, axis=0) - true_pos
    false_neg = np.sum(conf_mat1, axis=1) - true_pos

    precision = (np.sum(true_pos / (true_pos + false_pos))) / 10
    recall = (np.sum(true_pos / (true_pos + false_neg))) / 10
    fmeasure = 2 * precision * recall / (precision + recall)

    return precision, recall, fmeasure


def train_test_on_one_fold(train, test, classes):
    def find_mass_and_combine(image, classes):

        mass_values = {'number_of_holes': {
            '123457': 0,
            '069': 0,
            '8': 0
        }}
        # calculate 'number of holes' mass function:
        # fill with appropriate values according to the image

        if int(image['number_of_holes']) == 0:
            mass_values['number_of_holes']['123457'] = 1
        if int(image['number_of_holes']) == 1:
            mass_values['number_of_holes']['069'] = 1
        if int(image['number_of_holes']) == 2:
            mass_values['number_of_holes']['8'] = 1

        for attribute in classes:
            if attribute == 'number_of_holes':
                continue
            subclasses = classes[attribute]
            mass_values_per_class = {}
            for subclass in subclasses:
                if subclass in singletons:  # simple class, not a compound class
                    kde = stats.gaussian_kde(train[attribute][train['Class'] == int(subclass)])
                    mass_values_per_class[subclass] = float(kde(image[attribute]))
                else:
                    # here we sum up the instances of the classes in 'subclasses' and then we estimate the density
                    number_of_subclasses = len(subclass)
                    format = []
                    for single_class in subclass:
                        format.append(train[attribute][train['Class'] == int(single_class)])
                    total_df = pd.concat(format, ignore_index=True)
                    kde = stats.gaussian_kde(total_df)
                    mass_values_per_class[subclass] = float(kde(image[attribute]))
            mass_values[attribute] = mass_values_per_class

        # normalisation:
        for attribute in mass_values.keys():
            for classes in mass_values[attribute].keys():
                factor = 1.0 / sum(mass_values[attribute].values())
                for k in mass_values[attribute]:
                    mass_values[attribute][k] = mass_values[attribute][k] * factor

        # pprint(mass_values)

        # combination by dempster shafer rule:
        def intersect(d1, d2):
            result = ""
            for element in d1:
                if element in d2:
                    result = result + element

            return result

        combined = mass_values
        number_of_keys = len(combined.keys())
        while number_of_keys > 1:
            # select two attributes to combine
            # print(number_of_keys)
            keys = list(combined.keys())
            m12_phi = 0
            new_masses = {}
            '''
            pprint(combined[keys[0]])
            print('******')
            pprint(combined[keys[1]])
            '''
            for class0 in combined[keys[0]]:
                for class1 in combined[keys[1]]:
                    new_class = intersect(class0, class1)
                    if new_class == "":  # empty intersection, mass conflectuelle:
                        m12_phi += combined[keys[0]][class0] * combined[keys[1]][class1]
                    else:
                        if new_class in new_masses:
                            new_masses[new_class] += combined[keys[0]][class0] * combined[keys[1]][class1]
                        else:
                            new_masses[new_class] = combined[keys[0]][class0] * combined[keys[1]][class1]
            if m12_phi > 0.00001:
                new_masses[""] = m12_phi
            combined.pop(keys[0])
            combined.pop(keys[1])
            combined[keys[0] + '/' + keys[1]] = new_masses
            number_of_keys -= 1

        # pprint(combined)
        key = list(combined.keys())
        new_keys = list(combined[key[0]].keys())
        final_mass_values = combined[key[0]]
        all_singletons = True
        for key in new_keys:
            if key not in singletons:
                # apply max of plausibility
                all_singletons = False

        if all_singletons:
            masse_conflictuelle = final_mass_values.pop('')
            predicted_class = max(final_mass_values.items(), key=operator.itemgetter(1))[0]
            #print('masse de la classe predite: ', final_mass_values[predicted_class], ' masse conf: ',
                  #masse_conflictuelle)

        else:

            masses_of_singletons = {
                '0': 0,
                '1': 0,
                '2': 0,
                '3': 0,
                '4': 0,
                '5': 0,
                '6': 0,
                '7': 0,
                '8': 0,
                '9': 0
            }
            masse_conflictuelle = final_mass_values.pop('')
            for key, mass in zip(final_mass_values.keys(), final_mass_values.values()):
                for element in key:
                    masses_of_singletons[element] += mass / len(key)
            print('need for max of plausibility')

            predicted_class = max(masses_of_singletons.items(), key=operator.itemgetter(1))[0]

        return predicted_class, masse_conflictuelle


















    success = 0
    count = 0
    conf_mat = np.zeros([10, 10], int)
    ms_phi = []
    for index, image in test.iterrows():

        predicted_class, mass_conflictuelle = find_mass_and_combine(image, classes)
        ms_phi.append(mass_conflictuelle)
        if int(predicted_class) == int(image['Class']):
            success += 1
        else:
            print('true class is: ', image['Class'], 'predicted class is: ', predicted_class)

        conf_mat[int(image['Class'])][int(predicted_class)] += 1
        count += 1
        # print('true class is: ', image['Class'], 'predicted class is: ', predicted_class)
    mean = statistics.mean(ms_phi)
    success_rate = success / count
    print('mean is :', mean, 'and success rate is: ', success_rate)

    return success_rate, conf_mat


data_file = r'attributes.csv'

df = pd.read_csv(data_file).dropna()

avg_accuracy = 0
accuracies = []
number_of_folds = 1
precisions = 0
recalls = 0
fmeasures = 0
for i in range(number_of_folds):
    rand = random.sample(range(1, 1000), 5)
    train1 = df.sample(frac=0.9, random_state=rand[0])  # random state is a seed value
    test1 = df.drop(train1.index)
    train2 = df.sample(frac=0.9, random_state=rand[1])  # random state is a seed value
    test2 = df.drop(train2.index)
    train3 = df.sample(frac=0.9, random_state=rand[2])  # random state is a seed value
    test3 = df.drop(train3.index)
    train4 = df.sample(frac=0.9, random_state=rand[3])  # random state is a seed value
    test4 = df.drop(train4.index)
    train5 = df.sample(frac=0.9, random_state=rand[4])  # random state is a seed value
    test5 = df.drop(train5.index)

    accuracy1, conf_mat1 = train_test_on_one_fold(train1, test1, classes)
    precision1, recall1, fmeasure1 = precision_recall_fmeasure(conf_mat1)

    accuracy2, conf_mat2 = train_test_on_one_fold(train2, test2, classes)
    precision2, recall2, fmeasure2 = precision_recall_fmeasure(conf_mat2)

    accuracy3, conf_mat3 = train_test_on_one_fold(train3, test3, classes)
    precision3, recall3, fmeasure3 = precision_recall_fmeasure(conf_mat1)

    accuracy4, conf_mat4 = train_test_on_one_fold(train4, test4, classes)
    precision4, recall4, fmeasure4 = precision_recall_fmeasure(conf_mat1)

    accuracy5, conf_mat5 = train_test_on_one_fold(train5, test5, classes)
    precision5, recall5, fmeasure5 = precision_recall_fmeasure(conf_mat1)

    precision = (precision1 + precision2 + precision3 + precision4 + precision5) / 5
    recall = (recall1 + recall2 + recall3 + recall4 + recall5) / 5
    fmeasure = (fmeasure1 + fmeasure2 + fmeasure3 + fmeasure4 + fmeasure5) / 5

    precisions += precision
    recalls += recall
    fmeasures += fmeasure
    print(conf_mat5)
    print(conf_mat4)
    print(conf_mat3)
    print(conf_mat2)
    print(conf_mat1)
    print(precision)
    print(recall)
    print(fmeasure)

    print(accuracy1, accuracy2, accuracy3, accuracy4, accuracy5)
    total_accuracy = (accuracy1 + accuracy2 + accuracy3 + accuracy4 + accuracy5) / 5
    print('total accuracy is: ', total_accuracy)

    avg_accuracy += total_accuracy
    accuracies.append(total_accuracy)

avg_accuracy = avg_accuracy / number_of_folds

avg_precision = precisions / number_of_folds
avg_recall = recalls / number_of_folds
avg_fmeasure = fmeasures / number_of_folds

print('the average accuracy for ', number_of_folds, ' fold is: ', avg_accuracy)
print('the average precision for ', number_of_folds, ' fold is: ', avg_precision)
print('the average recall for ', number_of_folds, ' fold is: ', avg_recall)
print('the average f-measure for ', number_of_folds, ' fold is: ', avg_fmeasure)

print('the maximum accuracy is: ', max(accuracies))

'''
    rand = random.randint(1, 1000)
    p1 = df.sample(frac=0.2, random_state=rand)
    p2 = p1.sample(frac=0.25, random_state=rand)
    p3 = p2.sample(frac=0.333, random_state=rand)
    p4 = p3.sample(frac=0.5, random_state=rand)
    p5 = p3.drop(p4.index)

    train1 = pd.concat([p2, p3, p4, p5])
    test1 = p1
    train2 = pd.concat([p1, p3, p4, p5])
    test2 = p2
    train3 = pd.concat([p1, p2, p4, p5])
    test3 = p3
    train4 = pd.concat([p1, p2, p3, p5])
    test4 = p4
    train5 = pd.concat([p1, p2, p3, p4])
    test5 = p5
    '''

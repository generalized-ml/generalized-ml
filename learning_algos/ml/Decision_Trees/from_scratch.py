
dataset = [
    [18, 1, 0],
    [20, 0, 1],
    [23, 2, 1],
    [25, 1, 1],
    [30, 1, 0],
]
def gini_index(classes, groups ):
    n_instances = float(sum([len(group) for group in groups]))
    gini = [1]*len(groups)
    for i, group in enumerate(groups):
        score = 0
        if len(group)==0:
            continue
        for class_ in classes:
            p = ([x[-1] for x in group].count(class_))/len(group)
            score += p*p
        gini[i]=(gini[i] - score)*(len(group)/n_instances) ##weighted gini index 

    return sum(gini)

def split_data(row , index , dataset):
    groups_ = [[], []]
    for data in dataset:
        if data[index]>row[index]:
            groups_[1].append(data)
        else:
            groups_[0].append(data)
    return groups_

def get_split(dataset):
    classes = list(set([x[-1] for x in dataset]))
    best_score = 9999
    for feat in range(len(dataset[0])  -1):

        for row in dataset:
            groups  = split_data(row, feat, dataset)
            gini = gini_index(classes, groups)

            if gini<best_score:
                b_index, b_value, b_score, b_groups = feat, row[feat], gini, groups

    return {'index': b_index, 'value': b_value, 'groups': b_groups}




split = get_split(dataset)
print('\nBest Split:')
print('Column Index: %s, Value: %s' % ((split['index']), (split['value'])))



# Sample groups after a hypothetical split
group1 = [[20, 1, 0], [22, 1, 0]]  # Users who might not watch the movie
group2 = [[25, 2, 1], [30, 0, 1]]  # Users who might watch the movie

# Calculate and display the Gini index for the split
split_gini = gini_index([0,1], [group1, group2])
print(f'Gini Index of the split: {split_gini:.3f}')

# dataset = [
#     [18, 1, 0],
#     [20, 0, 1],
#     [23, 2, 1],
#     [25, 1, 1],
#     [30, 1, 0],
# ]
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
        if data[index]>=row[index]:
            groups_[1].append(data)
        else:
            groups_[0].append(data)
    return groups_

def get_split(dataset, min_size, depth_reach):
    classes = list(set([x[-1] for x in dataset]))

    class_count = {cls : [x[-1] for x in dataset].count(cls) for cls in classes}
    if len(dataset)==0:
        return None
    if (len(dataset)<=min_size) or depth_reach:
        return max(class_count, key = class_count.get)
    if len(classes)==1:
        return classes[0]
    best_score = 9999
    for feat in range(len(dataset[0])  -1):

        for row in dataset:
            groups  = split_data(row, feat, dataset)
            gini = gini_index(classes, groups)
            # if gini==0:
            #     print(groups)
            # print(feat, row[feat], gini, best_score)
            if gini<best_score:
                b_index, b_value, best_score, b_groups = feat, row[feat], gini, groups

    # print({'index': b_index, 'value': b_value, 'left': b_groups[0], "right" :b_groups[1]})
    return {'index': b_index, 'value': b_value, 'left': b_groups[0], "right" :b_groups[1]}

def split_recursion(split, max_depth, min_size, depth = 1):
    depth_reach = False
    if depth>=max_depth:
        depth_reach = True
    output = get_split(split["left"], min_size, depth_reach)
    if output is None:
        split["left"] = output
    else:
        try:
            int(output)
            split["left"] = output
        except:
            split["left"] = split_recursion(output, max_depth, min_size, depth+1)

    output = get_split(split["right"], min_size, depth_reach)
    if output is None:
        split["right"] = output
    else:
        try:
            int(output)
            split["right"] = output
        except:
            split["right"] = split_recursion(output, max_depth, min_size, depth+1)

    return split


def build_tree(dataset, max_depth, min_size = 1):
    # i will save a tree in a dictionary 
    #{"index" : , "value" : "left": {"index" :}, "right"}

    tree = get_split(dataset, min_size, False)

    split_recursion(tree, max_depth, min_size)

    return tree


#  Test building a tree with recursive functions
dataset = [[2.771244718,1.784783929,0],
           [9.728571309,1.169761413,0],
           [3.678319846,2.81281357,0],
           [3.961043357,2.61995032,0],
           [2.999208922,2.209014212,0],
           [7.497545867,3.162953546,1],
           [9.00220326,3.339047188,1],
           [7.444542326,0.476683375,1],
           [10.12493903,3.234550982,1],
           [6.642287351,3.319983761,1]]
max_depth = 3
min_size = 1
tree = build_tree(dataset, max_depth, min_size)

print(tree)
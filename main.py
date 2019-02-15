"""
1. Build 10 neural networks
2. Run each of them once
3. Get their readings
4. Apply genetics
5. Run from point 2
"""
import numpy as np
import pickle
import os
import random
# import math


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


def interpret_output(layer5):
    # print(layer5)
    global direction
    layer5 = np.argmax(layer5)
    direction = -1
    if layer5 == 0:
        direction = 0
    elif layer5 == 1:
        direction = 0.33
    elif layer5 == 2:
        direction = 0.67
    else:
        direction = 1
    return direction


def build_rand_network():
    # randomly initialize our weights with mean 0
    syn0 = 2 * np.random.random((16, 12)) - 1
    syn1 = 2 * np.random.random((12, 8)) - 1
    syn2 = 2 * np.random.random((8, 4)) - 1
    syn3 = 2 * np.random.random((4, 4)) - 1
    return [syn0, syn1, syn2, syn3]


def feed_forward_network(features, model_index):

    model = []
    db_name = "model" + str(model_index) + ".db"
    if os.path.isfile(db_name):
        with open(db_name, "rb") as f:
            model = pickle.load(f)

    model[0] = np.array(model[0])
    model[1] = np.array(model[1])
    model[2] = np.array(model[2])
    model[3] = np.array(model[3])
    features = np.array([features])

    for runs in range(20):
        l0 = features
        l1 = nonlin(np.dot(l0, model[0]))
        l2 = nonlin(np.dot(l1, model[1]))
        l3 = nonlin(np.dot(l2, model[2]))
        l4 = nonlin(np.dot(l3, model[3]))

        for idx, item in enumerate(features[0]):
            if item == 0:
                features[0][idx] = 0.0001

        # print('//////////////////////////////////////////////////////////')
        # print(features)
        # print('//////////////////////////////////////////////////////////')

        y = l4

        # Result = (For-all directions) Ghost * Biscuit * Wall * Move
        dir_max = np.argmax(y[0])
        if dir_max == 0 or True:
            y[0][0] = (6*features[0][0] + features[0][1] + 5.5*features[0][2] + 0.5*features[0][3]) / 13
        if dir_max == 1 or True:
            y[0][1] = (6*features[0][4] + features[0][5] + 5.5*features[0][6] + 0.5*features[0][7]) / 13
        if dir_max == 2 or True:
            y[0][2] = (6*features[0][8] + features[0][9] + 5.5*features[0][10] + 0.5*features[0][11]) / 13
        if dir_max == 3 or True:
            y[0][3] = (6*features[0][12] + features[0][13] + 5.5*features[0][14] + 0.5*features[0][15]) / 13

        l4_error = y - l4

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l4_delta = l4_error * nonlin(l4, deriv=True)

        ###################################################################################
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l3_error = l4_delta.dot(model[3].T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l3_delta = l3_error * nonlin(l3, deriv=True)

        ###################################################################################
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l2_error = l3_delta.dot(model[2].T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error * nonlin(l2, deriv=True)

        ###################################################################################
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(model[1].T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1, deriv=True)

        model[3] += l3.T.dot(l4_delta)
        model[2] += l2.T.dot(l3_delta)
        model[1] += l1.T.dot(l2_delta)
        model[0] += l0.T.dot(l1_delta)

    with open(db_name, "wb") as f:
        pickle.dump(model, f)

    return l4[0]


def breed_ten_models():
    np.random.seed(1)

    for i in range(1, 21):
        model = build_rand_network()
        db_name = "model"+str(i)+".db"
        with open(db_name, "wb") as f:
            pickle.dump(model, f)


def use_model(features, model_index):
    """
    Selects the right model and outputs the result of the model
    :param features: The feature vector
    :param model_index: The game which is running (out of 20)
    :return: result
    """
    return feed_forward_network(features, model_index)


def save_score(model_index, score):
    scores = []

    if os.path.isfile("scores.db"):
        with open("scores.db", "rb") as f:
            scores = pickle.load(f)

    scores[model_index] = score

    with open("scores.db", "wb") as f:
        pickle.dump(scores, f)

    # print(scores)


def init_scores():
    scores = [0 for x in range(20)]

    with open("scores.db", "wb") as f:
        pickle.dump(scores, f)


def get_fittest_parents():
    scores = []
    if os.path.isfile("scores.db"):
        with open("scores.db", "rb") as f:
            scores = pickle.load(f)

    p1 = np.argmax(scores)      # Highest fitness
    scores[p1] = -1
    p2 = np.argmax(scores)      # 2nd Highest fitness
    return [p1, p2]


def cross_over(p1_index, p2_index):
    p1_model = []
    p2_model = []

    db_name = "model" + str(p1_index+1) + ".db"
    if os.path.isfile(db_name):
        with open(db_name, "rb") as f:
            p1_model = pickle.load(f)

    db_name = "model" + str(p2_index+1) + ".db"
    if os.path.isfile(db_name):
        with open(db_name, "rb") as f:
            p2_model = pickle.load(f)

    child_model = p1_model

    for xi in range(len(p1_model)):
        for yi in range(len(p1_model[xi])):
            for zi in range(len(p1_model[xi][yi])):
                if random.uniform(0, 1) > 0.75:
                    child_model[xi][yi][zi] = p2_model[xi][yi][zi]

    return child_model


def mutate_model(model):
    for xi in range(len(model)):
        for yi in range(len(model[xi])):
            for zi in range(len(model[xi][yi])):
                if random.uniform(0, 1) > 0.75:
                    change = random.uniform(-0.5, 0.5)
                    model[xi][yi][zi] += change
    return model


def rebreed_ten_models(model):

    for i in range(1, 21):
        model = mutate_model(model)
        db_name = "model"+str(i)+".db"
        with open(db_name, "wb") as f:
            pickle.dump(model, f)


breed_ten_models()
init_scores()

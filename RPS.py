# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
"""
def player(prev_play, opponent_history=[]):

  opponent_history.append(prev_play)

    guess = "R"
    if len(opponent_history) > 2:
        guess = opponent_history[-2]

    return guess
"""

wtf = {}
"""
def player(prev_play, opponent_history=[]):
  global wtf

  n = 6

  if prev_play in ["R","P","S"]:
    opponent_history.append(prev_play)

  guess = "R" # default, until statistic kicks in

  if len(opponent_history)>n:
    inp = "".join(opponent_history[-n:])

    if "".join(opponent_history[-(n+1):]) in wtf.keys():
      wtf["".join(opponent_history[-(n+1):])]+=1
    else:
      wtf["".join(opponent_history[-(n+1):])]=1

    possible =[inp+"R", inp+"P", inp+"S"]

    for i in possible:
      if not i in wtf.keys():
        wtf[i] = 0

    predict = max(possible, key=lambda key: wtf[key])

    if predict[-1] == "P":
      guess = "S"
    if predict[-1] == "R":
      guess = "P"
    if predict[-1] == "S":
      guess = "R"


  return guess



def player(prev_opponent_play,
          opponent_history=[],
          play_order=[{
              "RR": 0,
              "RP": 0,
              "RS": 0,
              "PR": 0,
              "PP": 0,
              "PS": 0,
              "SR": 0,
              "SP": 0,
              "SS": 0,
          }]):

    if not prev_opponent_play:
        prev_opponent_play = 'R'
    opponent_history.append(prev_opponent_play)

    last_two = "".join(opponent_history[-5:])
    if len(last_two) == 6:
        play_order[0][last_two] += 1

    potential_plays = [
        prev_opponent_play + "R",
        prev_opponent_play + "P",
        prev_opponent_play + "S",
    ]

    sub_order = {
        k: play_order[0][k]
        for k in potential_plays if k in play_order[0]
    }

    prediction = max(sub_order, key=sub_order.get)[-1:]

    guess = {'P': 'S', 'R': 'P', 'S': 'R'}
    return guess[prediction]
"""
"""
# FAILS: VS. ABBEY;  VS. KRIS
# Chukwuemeka Aham
# RPS.py
import random

order = {}

def player(prev_play, opponent_history=[]):
  global order

  n = 3

  if prev_play in ["R","P","S"]:
    opponent_history.append(prev_play)

  guess = "R"

  if len(opponent_history) >= n:
    input = "".join(opponent_history[-n:])

    if "".join(opponent_history[-(n+1):]) in order.keys():
      order["".join(opponent_history[-(n+1):])]+=1
    else:
      order["".join(opponent_history[-(n+1):])]=1

    possible =[input + "R", input + "P", input + "S"]

    for i in possible:
      if not i in order.keys():
        order[i] = 0

    predict = max(possible, key=lambda key: order[key])

    if predict[-1] == "P":
      guess = "S"
    if predict[-1] == "R":
      guess = "P"
    if predict[-1] == "S":
      guess = "R"


  return guess
"""
"""
# FAILS: VS. ABBEY
# Jackzensen
# RPS.py
# Used to generate a longer dictionary

# from itertools import product
# count_values = {}
# play_order = []
# values = list(product("RPS", repeat=4))
# for i in values:
#   count_values[''.join(i)] = 0
# play_order.append(count_val


# This program remembers and counts the opponent's previous 4 plays, since 3 did not seem sufficient to meet the test case criteria

#It is the same strategy that Abbey uses, only with a longer memory.

def player(prev_opponent_play,
          opponent_history=[],
          
          play_order = [{'RRRR': 0, 'RRRP': 0, 'RRRS': 0, 'RRPR': 0, 'RRPP': 0, 'RRPS': 0, 'RRSR': 0, 'RRSP': 0, 'RRSS': 0, 'RPRR': 0, 'RPRP': 0, 'RPRS': 0, 'RPPR': 0, 'RPPP': 0, 'RPPS': 0, 'RPSR': 0, 'RPSP': 0, 'RPSS': 0, 'RSRR': 0, 'RSRP': 0, 'RSRS': 0, 'RSPR': 0, 'RSPP': 0, 'RSPS': 0, 'RSSR': 0, 'RSSP': 0, 'RSSS': 0, 'PRRR': 0, 'PRRP': 0, 'PRRS': 0, 'PRPR': 0, 'PRPP': 0, 'PRPS': 0, 'PRSR': 0, 'PRSP': 0, 'PRSS': 0, 'PPRR': 0, 'PPRP': 0, 'PPRS': 0, 'PPPR': 0, 'PPPP': 0, 'PPPS': 0, 'PPSR': 0, 'PPSP': 0, 'PPSS': 0, 'PSRR': 0, 'PSRP': 0, 'PSRS': 0, 'PSPR': 0, 'PSPP': 0, 'PSPS': 0, 'PSSR': 0, 'PSSP': 0, 'PSSS': 0, 'SRRR': 0, 'SRRP': 0, 'SRRS': 0, 'SRPR': 0, 'SRPP': 0, 'SRPS': 0, 'SRSR': 0, 'SRSP': 0, 'SRSS': 0, 'SPRR': 0, 'SPRP': 0, 'SPRS': 0, 'SPPR': 0, 'SPPP': 0, 'SPPS': 0, 'SPSR': 0, 'SPSP': 0, 'SPSS': 0, 'SSRR': 0, 'SSRP': 0, 'SSRS': 0, 'SSPR': 0, 'SSPP': 0, 'SSPS': 0, 'SSSR': 0, 'SSSP': 0, 'SSSS': 0}]):


    if not prev_opponent_play:
        prev_opponent_play = 'SSS'
    opponent_history.append(prev_opponent_play)

    last_three = "".join(opponent_history[-4:])    
    if len(last_three) == 4:                     
        try:
          play_order[0][last_three] += 1
        except:
          play_order[0][last_three] = 1


    potential_plays = [
        last_three[-3:] + "R",
        last_three[-3:] + "S",
        last_three[-3:] + "P",
    ]

    sub_order = {
        k: play_order[0][k]
        for k in potential_plays if k in play_order[0]
    }
    prediction = max(sub_order, key=sub_order.get)[-1:]

    ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}
    return ideal_response[prediction]

"""
"""
# pkarczma
# RPS.py
import random
# import numpy as np
#import pandas as pd
# from tensorflow import keras

# Global variables
moves = ['R', 'P', 'S']
ideal_response = {'R': 'P', 'P': 'S', 'S': 'R'}

# Variables for Keras method
df_train_x = None
df_train_y = None
model = None
hlen = 5
hentries = 20

# Variables for Markov Chain method
use_markov_chain = True
pair_keys = ['RR', 'RP', 'RS', 'PR', 'PP', 'PS', 'SR', 'SP', 'SS']
matrix = {}
memory = 0.9
my_history = []

def player(prev_play, opponent_history=[]):

    # Use a random choice by default
    guess = random.choice(moves)

    # Use Markov Chain method
    # - wins with all players with > 60% efficiency
    # - possible to adjust results with 'memory' variable
    if use_markov_chain == True:
        global matrix, my_history
        # initialize variables in the first game
        if prev_play == '':
            for pair_key in pair_keys:
                matrix[pair_key] = {'R': 1 / 3,
                                    'P': 1 / 3,
                                    'S': 1 / 3}
            opponent_history = []
            my_history = []
        # otherwise, add previous opponent play to the history
        else:
            opponent_history.append(prev_play)

        # make a prediction when enough entries in the history
        if len(my_history) >= 2:
            # create a pair from 2 plays ago
            prev_pair = my_history[-2] + opponent_history[-2]
            # introduce a memory loss of earlier observations for that pair,
            # memory decay speed can be adjusted using 'memory' variable
            for rps_key in matrix[prev_pair]:
                matrix[prev_pair][rps_key] = memory * matrix[prev_pair][rps_key]
            # then, update matrix for that pair
            matrix[prev_pair][prev_play] += 1

            # create a pair from the last play
            last_pair = my_history[-1] + opponent_history[-1]
            # if the matrix values are not equal for that pair,
            # make a prediction using the move with the higest value
            if max(matrix[last_pair].values()) != min(matrix[last_pair].values()):
                prediction = max([(v, k) for k, v in matrix[last_pair].items()])[1]
                guess = ideal_response[prediction]

        # append my guess to the history
        my_history.append(guess)
       # Return player guess
    return guess
"""
"""
    # [Deprecated] Use Keras library instead
    # Warning:
    # - 1st attempt for this problem that does not win with all players
    # - requires much more computation time
    if use_markov_chain == False:
        global df_train_x, df_train_y, model
        # initialize variables in the first game
        if prev_play == '':
            df_train_x = pd.DataFrame()
            df_train_y = pd.DataFrame()
            model = keras.Sequential([
                keras.layers.Dense(hlen, input_shape=(hlen,)),
                keras.layers.Dense(3, activation='softmax')
            ])
            model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
            opponent_history = []
        # otherwise, add previous opponent play to the history
        else:
            opponent_history.append(moves.index(prev_play))

        # use opponents play history to build a dataframe of opponent history
        # series of length of 'hlen' each as x axis and their following moves
        # as y axis for the training after at least 'hlen+1' plays
        if len(opponent_history) > hlen:
            df_train_x = df_train_x.append(pd.Series(opponent_history[-(hlen+1):-1]), ignore_index=True).astype('int8')
            df_train_y = df_train_y.append(pd.Series(opponent_history[-1]), ignore_index=True).astype('int8')

        # after 'hlen+hentries' plays, fit the model and make a prediction
        if len(opponent_history) >= (hlen+hentries):
            model.fit(df_train_x, df_train_y, epochs=5, verbose=0)
            df_test_x = pd.DataFrame([opponent_history[-hlen:]])
            predictions = model.predict([df_test_x])
            guess = ideal_response[moves[np.argmax(predictions[0])]]

  # Return player guess
  return guess

"""


# pkarczma
# RPS.py
import random
import numpy as np
#import pandas as pd
# from tensorflow import keras

# Global variables
moves = ['R', 'P', 'S']
ideal_response = {'R': 'P', 'P': 'S', 'S': 'R'}

# Variables for Keras method
df_train_x = None
df_train_y = None
model = None
hlen = 5
hentries = 20

# Variables for Markov Chain method
use_markov_chain = True
pair_keys = ['RR', 'RP', 'RS', 'PR', 'PP', 'PS', 'SR', 'SP', 'SS']
matrix = {}
memory = 0.9
my_history = []

def player(prev_play, opponent_history=[]):

    # Use a random choice by default
    guess = random.choice(moves)

    # Use Markov Chain method
    # - wins with all players with > 60% efficiency
    # - possible to adjust results with 'memory' variable
    if use_markov_chain == True:
        global matrix, my_history
        # initialize variables in the first game
        if prev_play == '':
            for pair_key in pair_keys:
                matrix[pair_key] = {'R': 1 / 3,
                                    'P': 1 / 3,
                                    'S': 1 / 3}
            opponent_history = []
            my_history = []
        # otherwise, add previous opponent play to the history
        else:
            opponent_history.append(prev_play)

        # make a prediction when enough entries in the history
        if len(my_history) >= 2:
            # create a pair from 2 plays ago
            prev_pair = my_history[-2] + opponent_history[-2]
            # introduce a memory loss of earlier observations for that pair,
            # memory decay speed can be adjusted using 'memory' variable
            for rps_key in matrix[prev_pair]:
                matrix[prev_pair][rps_key] = memory * matrix[prev_pair][rps_key]
            # then, update matrix for that pair
            matrix[prev_pair][prev_play] += 1

            # create a pair from the last play
            last_pair = my_history[-1] + opponent_history[-1]
            # if the matrix values are not equal for that pair,
            # make a prediction using the move with the higest value
            if max(matrix[last_pair].values()) != min(matrix[last_pair].values()):
                prediction = max([(v, k) for k, v in matrix[last_pair].items()])[1]
                guess = ideal_response[prediction]

        # append my guess to the history
        my_history.append(guess)
       # Return player guess
    return guess
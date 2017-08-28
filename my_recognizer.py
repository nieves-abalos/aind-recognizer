import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """

   #recognize(models, test_set)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for index_uword, unknown_word in test_set.get_all_Xlengths().items():
        X = unknown_word[0] 
        lengths = unknown_word[1]
        dict_words = {}
        best_word = None
        selected_score = float('-inf')
        # Let's see what word based on its model is more likely to be
        for word, model in models.items():
            # calculate the scores for each model(word) and update the 'probabilities' list.
            try:
                logL = model.score(X, lengths)
                #print("logL: {} for word {}".format(logL, word))
                dict_words[word] = logL
                 # determine the maximum score for each model.
                if logL > selected_score:
                    best_word = word
                    selected_score = logL
            except:
                #print("Can't get the score from model.score() for {}".format(word))
                dict_words[word] = float("-inf")
                pass    
           
        # Append the corresponding word (the tested word is deemed to be the word for which with the model was trained) to the list 'guesses'.
        guesses.append(best_word) 
        probabilities.append(dict_words)
    return probabilities, guesses
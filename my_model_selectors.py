import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None
    
    def base_model_CV(self, num_states, X, lengths):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # SelectorBIC(sequences, Xlengths, word, min_n_components=2, max_n_components=15, random_state = 14).select()
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        #print("\n WORD: {}".format(self.this_word))
        
        selected_model = None
        selected_score = float('+inf')
        selected_num_states = float('-inf')
        bic_score = float('+inf')
        for num_states in range(self.min_n_components, self.max_n_components):
            # lets find best model
            #print("\n Num states: {}".format(num_states))
            model = self.base_model(num_states)
            if model is not None:    
                try:
                    logL = model.score(self.X,self.lengths)
                    #print("score: logL {} and {} states".format(logL, num_states))
                    # p = n_components ** 2 + 2 * n_components * n_features - 1
                    # n_features >> variable defined in the model
                    # N = number of data points or sequences > sequences are self.sequences
                    parameters = num_states * num_states + 2 * num_states * len(self.X[0]) - 1
                    bic_score = - 2 * logL + parameters * math.log(len(self.X))
                    #print("BIC score: {} ".format(bic_score))
                except:
                    #print("Exception at model.score() with word {} and states {}".format(self.this_word, num_states))
                    pass
            else:
                #print("Not able to train HMM model (self.base_model) for word {} with the train samples and {} states.".format(self.this_word, num_states)) 
                pass   
        
            # update best model if BIC is better than the other BICs (LOWER)
            if bic_score < selected_score:
                selected_score = bic_score
                selected_num_states = num_states
                selected_model = model
                #print("UPDATED selected_score {} and selected_num_states {}".format(selected_score, selected_num_states))
            
        return selected_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        # SelectorDIC(sequences, Xlengths, word, min_n_components=2, max_n_components=15, random_state = 14).select()
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        #print("\n WORD: {}".format(self.this_word))
        
        selected_model = None
        selected_score = float('-inf')
        dic_score = float('-inf')
        selected_num_states = float('-inf')
        for num_states in range(self.min_n_components, self.max_n_components):
            # lets find best model
            #print("\n Num states: {}".format(num_states))
            model = self.base_model(num_states)
            if model is not None:    
                try:
                    # DIC is that we are trying to find the model that gives a high likelihood(small negative number) to the original word and low likelihood(very big negative number) to the other words. So DIC score is
                    # DIC = log(P(original world)) - average(log(P(otherwords)))
                    # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

                    # log(P(X(i))) in terms of hmmlearn is simply the model's score for that particular word
                    logL = model.score(self.X,self.lengths)
                    #print("score: logL {} and {} states".format(logL, num_states))
                    logL_others = 0
                    num_words = 0
                    for other_word in self.hwords:
                        if other_word != self.this_word:
                            X_other, lengths_other = self.hwords[other_word]
                            # SUM(log(P(X(all but i))
                            try:
                                logL_others = logL_others + model.score(X_other, lengths_other)
                                num_words = num_words + 1 
                            except:
                                #print("Exception at model.score(other_word) with word {} and states {}".format(self.this_word, num_states))
                                pass
                            
                    # all logLs calculated, now average:
                    logL_others = logL_others / num_words
                    # DIC = log(P(original world)) - average(log(P(otherwords)))
                    dic_score = logL - logL_others
                    #print("DIC score: {} ".format(dic_score))
                except:
                    #print("Exception at model.score() with word {} and states {}".format(self.this_word, num_states))
                    pass
            else:
                #print("Not able to train HMM model (self.base_model) for word {} with the train samples and {} states.".format(self.this_word, num_states)) 
                pass   
        
            # update best model if DIC is HIGHER
            if dic_score > selected_score:
                selected_score = dic_score
                selected_num_states = num_states
                selected_model = model
                #print("UPDATED selected_score {} and selected_num_states {}".format(selected_score, selected_num_states))
            
        return selected_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
        
    def select(self):
        # (sequences, Xlengths, word, min_n_components=2, max_n_components=15, random_state = 14)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
       # print("\nSelectorCV > WORD: {}".format(self.this_word))
        selected_model = None
        selected_score = float('-inf')
        selected_num_states = float('-inf')
        if len(self.lengths) > 1:
            # k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more
            n_splits = min(3, len(self.sequences))
            split_method = KFold(n_splits) 

            for num_states in range(self.min_n_components, self.max_n_components):
                # break the training set into "folds" and rotate which fold is left out of training. 
                #print("\n Num states: {}".format(num_states))
                avgLogL = 0
                num_splits = 0
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences): 
                    X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    model = self.base_model_CV(num_states, X_train, lengths_train)
                    if model is not None:    
                        try:
                            # The "left out" fold scored, this gives us a proxy method of finding the best model to use on "unseen data"
                            X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                            logL = model.score(X_test, lengths_test)
                            #print("score: logL {} and {} states".format(logL, num_states))
                            avgLogL = avgLogL + logL
                            num_splits = num_splits + 1
                        except:
                            #print("Exception at model.score() with word {} and states {}".format(self.this_word, num_states))
                            pass
                    else:
                        #print("Not able to train HMM model (self.base_model_CV) for word {} with the train samples and {} states.".format(self.this_word, num_states)) 
                        pass   
                
                # Calculate avg log likelihood of all splits
                #print("> sum LogL {} and {} num_splits".format(avgLogL, num_splits))
                if num_splits != 0:
                    avgLogL = avgLogL / num_splits
                else:
                    avgLogL = float('-inf')
                #print("> avgLogL {} for {} states".format(avgLogL, num_states))
                # update best model if avgLogL is better than the other  states
                if avgLogL > selected_score:
                    selected_score = avgLogL
                    selected_num_states = num_states
                    #print("UPDATED selected_score {} and selected_num_states {}".format(selected_score, selected_num_states))
                
            # rebuild model with all samples (that is fit with self.X and self.lengths)
            #print("Average Log Likelihood: {}".format(selected_score))
            #print("Best number of states: {}".format(selected_num_states))
            try:
                selected_model = self.base_model(selected_num_states)
            except:
                print("Exception at creating the selected HMM model self.base_model >> model.initialize and fit. All samples. {} states.".format(self.this_word, num_states)) 
                pass
        
        return selected_model

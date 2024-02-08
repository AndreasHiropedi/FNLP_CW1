"""
Foundations of Natural Language Processing

Assignment 1

Please complete functions, based on their doc_string description
and instructions of the assignment. 

To test your code run:

  [hostname]s1234567 python3 s1234567.py
or
  [hostname]s1234567 python3 -i s1234567.py

The latter is useful for debugging, as it will allow you to access many
 useful global variables from the python command line

*Important*: Before submission be sure your code works _on a DICE machine_
with the --answers flag:

  [hostname]s1234567 python3 s1234567.py --answers

Also use this to generate the answers.py file you need for the interim
checker.

Best of Luck!
"""
from collections import defaultdict, Counter
from typing import Tuple, List, Any, Set, Dict, Callable

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk import MaxentClassifier
from nltk.corpus import brown, ppattach  # import corpora

# Import LgramModel
from nltk_model import *

# Import the Twitter corpus
from twitter.twitter import *

twitter_file_ids = "20100128.txt"
assert twitter_file_ids in xtwc.fileids()


# Some helper functions

import matplotlib.pyplot as plt

def hist(hh: List[float], title: str, align: str = 'mid',
         log: bool = False, block: bool = False):
  """
  Show a histgram with bars showing mean and standard deviations
  :param hh: the data to plot
  :param title: the plot title
  :param align: passed to pyplot.hist, q.v.
  :param log: passed to pyplot.hist, q.v.  If present will be added to title
  """
  hax=plt.subplots()[1] # Thanks to https://stackoverflow.com/a/7769497
  sdax=hax.twiny()
  hax.hist(hh,bins=30,color='lightblue',align=align,log=log)
  hax.set_title(title+(' (log plot)' if log else ''))
  ylim=hax.get_ylim()
  xlim=hax.get_xlim()
  m=np.mean(hh)
  sd=np.std(hh)
  sdd=[(i,m+(i*sd)) for i in range(int(xlim[0]-(m+1)),int(xlim[1]-(m-3)))]
  for s,v in sdd:
       sdax.plot([v,v],[0,ylim[0]+ylim[1]],'r' if v==m else 'pink')
  sdax.set_xlim(xlim)
  sdax.set_ylim(ylim)
  sdax.set_xticks([v for s,v in sdd])
  sdax.set_xticklabels([str(s) for s,v in sdd])
  plt.show(block=block)


def compute_accuracy(classifier, data: List[Tuple[List, str]]) -> float:
    """
    Computes accuracy (range 0 - 1) of a classifier.
    :type classifier: e.g. NltkClassifierWrapper or NaiveBayes
    :param classifier: the classifier whose accuracy we compute.
    :param data: A list with tuples of the form (list with features, label)
    :return accuracy (range 0 - 1).
    """
    correct = 0
    for d, gold in data:
        predicted = classifier.classify(d)
        correct += predicted == gold
    return correct/len(data)


def apply_extractor(extractor_f: Callable[[str, str, str, str, str], List[Any]], data: List[Tuple[Tuple[str], str]])\
        -> List[Tuple[List[Any], str]]:
    """
    Helper function:
    Apply a feature extraction method to a labeled dataset.
    :param extractor_f: the feature extractor, that takes as input V, N1, P, N2 (all strings) and returns a list of features
    :param data: a list with tuples of the form (id, V, N1, P, N2, label)

    :return a list with tuples of the form (list with features, label)
    """
    r = []
    for d in data:
        r.append((extractor_f(*d[1:-1]), d[-1]))
    return r


def get_annotated_tweets():
    """
    :rtype list(tuple(list(str), bool))
    :return: a list of tuples (tweet, a) where tweet is a tweet preprocessed by us,
    and a is True, if the tweet is in English, and False otherwise.
    """
    import ast
    with open("twitter/annotated_dev_tweets.txt") as f:
        return [ast.literal_eval(line) for line in f.readlines()]


class NltkClassifierWrapper:
    """
    This is a little wrapper around the nltk classifiers so that we can interact with them
    in the same way as the Naive Bayes classifier.
    """
    def __init__(self, classifier_class: nltk.classify.api.ClassifierI, train_features: List[Tuple[List[Any], str]], **kwargs):
        """

        :param classifier_class: the kind of classifier we want to create an instance of.
        :param train_features: A list with tuples of the form (list with features, label)
        :param kwargs: additional keyword arguments for the classifier, e.g. number of training iterations.
        :return None
        """
        self.classifier_obj = classifier_class.train(
            [(NltkClassifierWrapper.list_to_freq_dict(d), c) for d, c in train_features], **kwargs)

    @staticmethod
    def list_to_freq_dict(d: List[Any]) -> Dict[Any, int]:
        """
        :param d: list of features

        :return: dictionary with feature counts.
        """
        return Counter(d)

    def classify(self, d: List[Any]) -> str:
        """
        :param d: list of features

        :return: most likely class
        """
        return self.classifier_obj.classify(NltkClassifierWrapper.list_to_freq_dict(d))

    def show_most_informative_features(self, n = 10):
        self.classifier_obj.show_most_informative_features(n)

# End helper functions

# ==============================================
# Section I: Language Identification [60 marks]
# ==============================================

# Question 1.1 [7.5 marks]
def train_LM(corpus: nltk.corpus.CorpusReader) -> LgramModel:
    """
    Build a bigram letter language model using LgramModel
    based on the lower-cased all-alpha subset of the entire corpus

    :param corpus: An NLTK corpus

    :return: A padded letter bigram model based on nltk.model.NgramModel
    """

    # subset the corpus to only include all-alpha tokens converted to lowercase
    corpus_tokens = []

    for word in corpus.words():

        if word.isalpha():

            lower_word = word.lower()
            corpus_tokens.append(lower_word)

    bigram_model = LgramModel(2, corpus_tokens, pad_left=True, pad_right=True)

    # Return a smoothed padded bigram letter language model
    return bigram_model



# Question 1.2 [7.5 marks]
def tweet_ent(file_name: str, bigram_model: LgramModel) -> List[Tuple[float, List[str]]]:
    """
    Using a character bigram model, compute sentence entropies
    for a subset of the tweet corpus, removing all non-alpha tokens and
    tweets with less than 5 all-alpha tokens, then converted to lowercase

    :param file_name: twitter file to process

    :return: ordered list of average entropies and tweets"""

    list_of_tweets = xtwc.sents(file_name)

    # Clean up the tweet corpus to remove all non-alpha
    # tokens and tweets with less than 5 remaining tokens
    # and convert remaining tweets to lowercase
    cleaned_list_of_tweets = []

    for tweet in list_of_tweets:

        cleaned_tweet = []

        for word in tweet:

            if word.isalpha():

                cleaned_tweet.append(word.lower())

        if len(cleaned_tweet) >= 5:

            cleaned_list_of_tweets.append(cleaned_tweet)

    # Construct a list of tuples of the form: (entropy, tweet)
    # for each tweet in the cleaned corpus, where entropy is the
    # average word for the tweet, and return the list of
    # (entropy, tweet) tuples sorted by entropy  
    entropy_tweet_pairs = []

    for tweet in cleaned_list_of_tweets:
        
        total_tweet_entropy = 0

        for word in tweet:

            word_ent = bigram_model.entropy(word, pad_left=True, pad_right=True, perItem=True)
            total_tweet_entropy += word_ent

        average_entropy = total_tweet_entropy / len(tweet)
        entropy_tweet_pairs.append((average_entropy, tweet))

    entropy_tweet_pairs.sort()

    return entropy_tweet_pairs


# Question 1.3 [3 marks]
def short_answer_1_3() -> str:
    """
    Briefly explain what left and right padding accomplish and why
    they are a good idea. Assuming you have a bigram model trained on
    a large enough sample of English that all the relevant bigrams
    have reliable probability estimates, give an example of a string
    whose average letter entropy you would expect to be (correctly)
    greater with padding than without and explain why.
   
    :return: your answer
    """

    return inspect.cleandoc(
        """
        Padding can be used to denote the beginning and end of a string or sentence.  
        Padding is a good idea because it helps the model accurately estimate the 
        probabilities of characters or words at the beginning and end of 
        strings or sentences. Consider the string 'za'; since the letter z 
        is rarely found at the beginning of English words, padding would result 
        in a higher average letter entropy.
        """
    )


# Question 1.4 [3 marks]
def short_answer_1_4() -> str:
    """
    Explain the output of lm.entropy('bbq',verbose=True,perItem=True)
    See the Coursework 1 instructions for details.

    :return: your answer
    """

    return inspect.cleandoc(
        """
        p(b|('<s>',)) = [2-gram] 0.046511       bigram probability of the first letter being b
        p(b|('b',)) = [2-gram] 0.007750         bigram probability of the next letter being b given the first letter was b
        backing off for ('b', 'q')              since there was no bigram probability for the next letter being q given the previous letter being b, back off to unigram model
        p(q|()) = [1-gram] 0.000892             unigram probability of the letter q
        p(q|('b',)) = [2-gram] 0.000092         scaled probability of the unigram model 
        p(</s>|('q',)) = [2-gram] 0.010636      bigram probability of the last letter being q
        7.85102054894183                        approximate cross-entropy of the n-gram model for the word bbq
        """
    )


# Question 1.5 [3 marks]
def short_answer_1_5() -> str:
    """
    Inspect the distribution of tweet entropies and discuss.
    See the Coursework 1 instructions for details.

    :return: your answer
    """

    global ents
    # Uncomment the following lines when you are ready to work on this.
    # Please comment them out again or delete them before submitting.
    # Note that you will have to close the two plot windows to allow this
    # function to return.
    # just_e = [e for (e,tw) in ents]
    # hist(just_e,"Bi-char entropies from cleaned twitter data")
    # hist(just_e,"Bi-char entropies from cleaned twitter data",log=True,block=True)
    
    return inspect.cleandoc(
        """
        The log histogram makes visualizing the tweet entropies distribution clearer. 
        We see that most tweets have quite a low entropy (within one standard deviation 
        from the mean), and there are very few points with high entropy. We can use entropy to 
        distinguish between different types of tweets, such as English versus non-English, 
        where we can assume that non-English tweets will have a higher entropy since our model 
        was trained on the Brown corpus.
        """
    )


# Question 1.6 [10 marks]
def is_English(bigram_model: LgramModel, tweet: List[str]) -> bool:
    """
    Classify if the given tweet is written in English or not.

    :param bigram_model: the bigram letter model trained on the Brown corpus
    :param tweet: the tweet
    :return: True if the tweet is classified as English, False otherwise
    """

    tweet_entropy = 0

    # for each word in the tweet, get its entropy and compute total entropy of the tweet
    for word in tweet:

        word_entropy = bigram_model.entropy(word, pad_left=True, pad_right=True, perItem=True)
        tweet_entropy += word_entropy
    
    # compute the average letter entropy of the tweet
    average_tweet_entropy = tweet_entropy / len(tweet)

    # classify it as English or not based on the observed threshold in q1.5
    return average_tweet_entropy < 4


# Question 1.7 [16 marks]
def essay_question():
    """

    THIS IS AN ESSAY QUESTION WHICH IS INDEPENDENT OF THE PREVIOUS
    QUESTIONS ABOUT TWITTER DATA AND THE BROWN CORPUS!

    See the Coursework 1 instructions for a question about the average
    per word entropy of English.
    1) Name 3 problems that the question glosses over
    2) What kind of experiment would you perform to get a better estimate
       of the per word entropy of English?
    3) Elaborate on the temporal dynamics of language evolution and its implications for 
    per-word entropy. How might the meaning and usage of words change over time, and how 
    can your experimental design capture these changes?

    There is a limit of 600 words for this question.
    :return: your answer
    """
    return inspect.cleandoc(
        """
        1. Determining the per-word cross-entropy of English poses a great challenge that requires
        more information in order to obtain a meaningful answer. Additionally, there are several
        assumptions that this question makes that could be problematic:

`           - One big issue that is not accounted for is that of new words being added to the English
            language. This would result in sparse data, as the new words would generate zero probability 
            sequences that are gramatical but do not appear in the corpus.

            - Another issue is the omission of the corpus to be used. We know that per-word entropy depends
            on the frequency of words, which is highly dependent on the context in which different words appear. 
            Therefore, using different corpora from different genres, demographics, time periods, etc., 
            would most likely result in different entropy values. 

            - Yet another issue is the ambiguity around the meaning of the term 'English'. This term would need to 
            be clearely defined, as there is no corpus currently that can capture the entirety of the English language;
            therefore, to produce a more accurate estimate for per-word entropy, a specific subset of the English 
            language should be clearly specified.

        2. One possible experiment to obtain a better estimate for the per-word entropy would involve using a large, diverse 
        corpus, such as the Brown corpus. Before performing any analysis, the data should be cleaned and pre-processed
        to remove any unwanted characters. The words in the corpus should also be tokenized and converted to lowercase, 
        and we could use word counts to estimate the probabilities of each word. Additionally, we would use our model's cross
        entropy value to approximate the value for entropy. To compute the cross entropy value, we would use the following formula:

        H_m (w_1 ... w_n) = (−1/n) * log_2 (P_m (w1 ... w_n))

        This formula would produce the average negative log probability that the model assigns to each word in a sequence, which
        is also normalized for sequence length. This value can be used as an estimate for how well the model can predict
        the next word in the sequence: the lower the entropy, the better the model. 

        This setup would allow us to come up with a reasonable estimate of the per-word entropy of English. Firstly, the use 
        of a diverse corpus ensures our data is not biased towards a specifc genre or demographic. Additionally, our measure of
        cross-entropy acts as an upper bound for the true entropy, therefore making it a reasonable estimate that is more likely to 
        overestimate the true value than underestimate it. Lastly, the choice of log base 2 allows us to provide an answer that 
        measures the necessary number of bits to encode the information in a word, thus aligning with the information-theoretical 
        concept of entropy as a measure of the amount of information in a system.

        3. The evolution of language over time is influenced by several factors, such as cultural shifts and social
        changes. As language evolves, new words are introduced to describe new concepts, older words may change meaning or
        their usage may decrease over time, and the language may undergo lexical diversification. All these changes impact per-word 
        entropy, since per-word entropy measures the uncertainty of a word's occurrence in a given context. Therefore, the use of up-to-date,
        diversified corpora, as mentioned in the experiment above, would help combat this issue, by providing a greater scope of which words
        are used more frequently at that time period. Additionally, the use of cross-entropy provides a good approximation for the 
        per-word entropy, as it acts as an upper bound for the true entropy value. 
        """
    )


#############################################
# SECTION II - RESOLVING PP ATTACHMENT AMBIGUITY
#############################################

# Question 2.1 [15 marks]
class NaiveBayes:
    """
    Naive Bayes model with Lidstone smoothing (parameter alpha).
    """

    def __init__(self, data: List[Tuple[List[Any], str]], alpha: float):
        """
        :param data: A list with tuples of the form (list with features, label)
        :param alpha: \alpha value for Lidstone smoothing
        """

        self.vocab = self.get_vocab(data)
        self.alpha = alpha
        self.prior, self.likelihood = self.train(data, alpha, self.vocab)

    @staticmethod
    def get_vocab(data: List[Tuple[List[Any], str]]) -> Set[Any]:
        """
        Compute the set of all possible features from the (training) data.
        :param data: A list with tuples of the form (list with features, label)

        :return: The set of all features used in the training data for all classes.
        """

        vocab = []
        # search all the data
        for document in data:
            # get the list of features
            features = document[0]
            # for all features
            for feature in features:
                    # if feature is not already in the vocab
                    if feature not in vocab:
                        # add to the vocab
                        vocab.append(feature)

        return vocab

    @staticmethod
    def train(data: List[Tuple[List[Any], str]], alpha: float, vocab: Set[Any]) -> Tuple[Dict[str, float],
          Dict[str, Dict[
          Any, float]]]:
        """
        Estimates the prior and likelihood from the data with Lidstone smoothing.

        :param data: A list of tuples ([f1, f2, ... ], c) with
                    the first element being a list of features and
                    the second element being its class.
        :param alpha: alpha value for Lidstone smoothing
        :param vocab: The set of all features used in the training data
                      for all classes.

        :return: Two dictionaries: the prior and the likelihood
                 (in that order).
        The returned values should relate as follows to the probabilities:
            prior[c] = P(c)
            likelihood[c][f] = P(f|c)
        """

        assert alpha >= 0.0

        # Create a dictionary of classes and the pairing of 
        # (number of documents class appears in, features of the class)
        classes = {}

        for document in data:

            current_class = document[1]

            if current_class not in classes:

                classes[current_class] = [1, np.zeros(len(vocab))]

            else:

                classes[current_class][0] += 1

            for i in range(len(vocab)):

                current_features = document[0]

                if vocab[i] in current_features:

                    classes[current_class][1][i] += 1

        # Calculate the priors and likelihoods 
        priors = {}
        likelihoods = {}

        for c, feature_counts in classes.items():

            priors[c] = feature_counts[0] / len(data)
            c_likelihoods = {}

            for i in range(len(vocab)):

                c_likelihoods[vocab[i]] = (feature_counts[1][i] + alpha) / (sum(feature_counts[1]) + alpha * len(vocab))
            
            likelihoods[c] = c_likelihoods

        return priors, likelihoods

    def prob_classify(self, d: List[Any]) -> Dict[str, float]:
        """
        Compute the probability P(c|d) for all classes.
        :param d: A list of features.

        :return: The probability p(c|d) for all classes as a dictionary.
        """

        d_probabilities = {}
        probability_of_d = 0

        # Calculate P(d|c) * P(c) seperately from P(d) 
        # for each class individual class
        for c, c_likelihoods in self.likelihood.items():

            features_given_c = 1

            for feature in d:

                if feature in c_likelihoods:

                    features_given_c = c_likelihoods[feature] * features_given_c

            d_probabilities[c] = self.prior[c] * features_given_c
            probability_of_d += self.prior[c] * features_given_c

        # Combine results to calculate P(c|d) as P(d|c) * P(c) / P(d)
        # for each individual class
        for c in d_probabilities.keys():
            
            d_probabilities[c] = d_probabilities[c] / probability_of_d

        return d_probabilities
    
    def classify(self, d: List[Any]) -> str:
        """
        Compute the most likely class of the given "document" with ties broken arbitrarily.
        :param d: A list of features.

        :return: The most likely class.
        """

        # Compute the class probabilities for d
        d_probabilities = self.prob_classify(d)

        # and return the class matching the highest one
        return max(d_probabilities, key=d_probabilities.get)


# Question 2.2 [15 marks]
def open_question_2_2() -> str:
    """
    See the Coursework 1 instructions for detail of the following:
    1) The differences in accuracy between the different ways
        to extract features?
    2) The difference between Naive Bayes vs Logistic Regression
    3) An explanation of a binary feature that returns 1
        if V=`imposed' AND N_1 = `ban' AND P=`on' AND N_2 = `uses'.

    Limit: 150 words for all three sub-questions together.
    """

    return inspect.cleandoc(
        """
        1. The difference in accuracy implies that feature selection impacts 
        the performance of the logistic regression model. We see that, despite
        all four words providing some degree of information, ultimately a 
        combination of more features results in the highest accuracy (81.08).
        However, of all individual features, we see that P provides the most 
        information (74.13), suggesting that prepositional attachment heavily
        relies on the choice of P.

        2. The observed accuracy in Q2.1 was 79.50%, slightly below that of the 
        logistic regression model. This could be because Naive Bayes assumes 
        that all features are conditionally independent; however, since feature 
        P provides more information, its importance should be elevated (which is 
        accounted for in the logistic regression model).

        3. Using such a feature would reduce the model's ability to generalise to
        unseen data. Additionally, as it provides no information about individual
        features, it would be more beneficial to use individual features.
        """
    )


# Feature extractors used in the table:

def feature_extractor_1(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("v", v)]

def feature_extractor_2(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("n1", n1)]

def feature_extractor_3(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("p", p)]

def feature_extractor_4(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("n2", n2)]

def feature_extractor_5(v: str, n1: str, p: str, n2: str) -> List[Any]:
    return [("v", v), ("n1", n1), ("p", p), ("n2", n2)]


# Question 2.3, part 1 [10 marks]
def your_feature_extractor(v: str, n1: str, p:str, n2:str) -> List[Any]:
    """
    Takes the head words and produces a list of features. The features may
    be of any type as long as they are hashable.

    :param v: The verb.
    :param n1: Head of the object NP.
    :param p: The preposition.
    :param n2: Head of the NP embedded in the PP.

    :return: A list of features produced by you.
    """

    features = [("v", v), ("n1", n1), ("p", p), ("n2", n2), ("v p", (v, p)), ("n1 p", (n1, p)), ("n2, p", (n2, p)),
                ("v n2", (v, n2)), ("len n1 n2", len(n1 + n2)), ("verb to", (v, p == "to")),
                ("n1 proper", n1[0].isupper()), ("n2 proper", n2[0].isupper()), ("n2 number", n2.isdigit()),
                ("n1 v", (n1.isdigit(), v)), ("n1 p n2", (n1.isdigit(), p, n2.isdigit()))]
    
    return features


# Question 2.3, part 2 [10 marks]
def open_question_2_3() -> str:
    """
    Briefly describe your feature templates and your reasoning for them.
    Pick three examples of informative features and discuss why they make sense or why they do not make sense
    and why you think the model relies on them.

    There is a limit of 300 words for this question.
    """

    return inspect.cleandoc(
        """
        We can clearly see from q2.2 that using individual words as features 
        can be informative, and that a combination of words can also help improve
        the accuracy of the model. Therefore, I tried to incorporate these as part 
        of my chosen features, and extend the list of features by also considering 
        proper nouns and numbers. Additionally, based on the data provided, and also 
        intuitively, numbers result in PP attachement to the verb in some instances, 
        and to the noun in others. Moreover, I tried to also lemmatize the words, and
        also include POS taggs as features, but that seemed to make the accuracy worse.

        Example 1: ('verb to')
        A quite common pattern in English that can indicate noun attachment is the
        combination of verb and "to". An example of this would be with the verb "rose", 
        e.g., "rose to fame"; this pattern works well as a feature and is part of the top 
        30 features. However, it is worth noting that "to" alone does not indicate attachment,
        but in conjunction with certain verbs it does.

        Example 2: (n1 p n2)
        This feature is also part of the top 30 features, and produces a negative correlation
        between 'not numeric by numeric' and the PP being attached to the noun. This is to be 
        expected, as there are plenty of examples in English where this occurs, such as 
        'divide by 10'. 

        Example 3: (len n1 n2)
        Although this feature was not part of the top 30, it appears that low values of len(n1 + n2)
        seem to indicate verb attachment, and having this feature helps increase accuracy. I assume that
        a reasonable explanation for this would be when n1 and n2 are both numeric, and hence len(n1 + n2)
        is lower, this increases the likelihood of verb attachment.
        """
    )


"""
Format the output of your submission for both development and automarking. 
!!!!! DO NOT MODIFY THIS PART !!!!!
"""

def answers():
    # Global variables for answers that will be used by automarker
    global ents, lm, top10_ents, bottom10_ents
    global answer_open_question_2_2, answer_open_question_2_3
    global answer_short_1_4, answer_short_1_5, answer_short_1_3, answer_essay_question

    global naive_bayes
    global acc_extractor_1, naive_bayes_acc, lr_acc, logistic_regression_model, dev_features
    global dev_tweets_preds


    print("*** Part I***\n")

    print("*** Question 1.1 ***")
    print('Building Brown news bigram letter model ... ')
    lm = train_LM(brown)
    print('Letter model built')

    print("*** Question 1.2 ***")
    ents = tweet_ent(twitter_file_ids, lm)

    top10_ents = ents[:10]
    bottom10_ents = ents[-10:]

    answer_short_1_3 = short_answer_1_3()
    print("*** Question 1.3 ***")
    print(answer_short_1_3)

    answer_short_1_4 = short_answer_1_4()
    print("*** Question 1.4 ***")
    print(answer_short_1_4)

    answer_short_1_5 = short_answer_1_5()
    print("*** Question 1.5 ***")
    print(answer_short_1_5)

    print("*** Question 1.6 ***")
    all_dev_ok = True
    dev_tweets_preds = []
    for tweet, gold_answer in get_annotated_tweets():
        prediction = is_English(lm, tweet)
        dev_tweets_preds.append(prediction)
        if prediction != gold_answer:
            all_dev_ok = False
            print("Missclassified", tweet)
    if all_dev_ok:
        print("All development examples correctly classified! "
              "We encourage you to test and tweak your classifier on more tweets.")

    answer_essay_question = essay_question()
    print("*** Question 1.7 (essay question) ***")
    print(answer_essay_question)

    print("*** Part II***\n")

    print("*** Question 2.1 ***")
    naive_bayes = NaiveBayes(apply_extractor(feature_extractor_5, ppattach.tuples("training")), 0.1)
    naive_bayes_acc = compute_accuracy(naive_bayes, apply_extractor(feature_extractor_5, ppattach.tuples("devset")))
    print(f"Accuracy on the devset: {naive_bayes_acc * 100}%")

    print("*** Question 2.2 ***")
    answer_open_question_2_2 = open_question_2_2()
    print(answer_open_question_2_2)

    print("*** Question 2.3 ***")
    training_features = apply_extractor(your_feature_extractor, ppattach.tuples("training"))
    dev_features = apply_extractor(your_feature_extractor, ppattach.tuples("devset"))
    logistic_regression_model = NltkClassifierWrapper(MaxentClassifier, training_features, max_iter=10)
    lr_acc = compute_accuracy(logistic_regression_model, dev_features)

    print("30 features with highest absolute weights")
    logistic_regression_model.show_most_informative_features(30)

    print(f"Accuracy on the devset: {lr_acc*100}")

    answer_open_question_2_3 = open_question_2_3()
    print("Answer to open question:")
    print(answer_open_question_2_3)



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        from autodrive_embed import run, carefulBind
        import adrive1

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive1.extract_answers, errlog)
    else:
        answers()

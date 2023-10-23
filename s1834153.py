import itertools
import math

import numpy as np

import nltk, inspect, sys, hashlib

from nltk.corpus import brown

# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist, LidstoneProbDist

from nltk.tag import map_tag

from adrive2 import trim_and_warn

assert map_tag('brown', 'universal', 'NR-TL') == 'NOUN', '''
Brown-to-Universal POS tag map is out of date.'''


class HMM:
    def __init__(self, train_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []
        self.viterbi = []
        self.backpointer = []

    # Q1

    # Compute emission model using ConditionalProbDist with a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with +0.001 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.
    def emission_model(self, train_data):
        """Compute an emission model based on labelled training data.
        Don't forget to lowercase the observation otherwise it mismatches the test data.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """

        # Don't forget to lowercase the observation otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences
        data = []
        for sentence in train_data:
            for word, tag in sentence:
                data.append((tag, word.lower()))

        emission_FD = ConditionalFreqDist(data)
        lidstone_estimator = lambda fd: LidstoneProbDist(fd, 0.001, fd.B() + 1)
        self.emission_PD = ConditionalProbDist(emission_FD, lidstone_estimator)
        self.states = emission_FD.conditions()

        return self.emission_PD, self.states

    # Q1

    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4
    def elprob(self, state, word):
        """
        The log of the estimated probability of emitting a word from a state

        :param state: the state name
        :type state: str
        :param word: the word
        :type word: str
        :return: log base 2 of the estimated emission probability
        :rtype: float
        """
        return math.log(self.emission_PD[state].prob(word), 2)

    # Q2
    # Compute transition model using ConditionalProbDist with the same
    #  estimator as above (but without the extra bin)
    # See comments for emission_model above for details on the estimator.
    def transition_model(self, train_data):
        """
        Compute a transition model using a ConditionalProbDist based on
          labelled data.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """
        data = []
        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL </s> and the END SYMBOL </s>
        for sentence in train_data:
            cleaned_sentence = sentence[:]
            cleaned_sentence.insert(0, ("<s>", "<s>"))
            cleaned_sentence.append(("</s>", "</s>"))
            data.append(cleaned_sentence)
        tagGenerators = (((s[i][1], s[i + 1][1]) for i in range(len(s) - 1)) for s in data)
        data = itertools.chain.from_iterable(tagGenerators)

        transition_FD = ConditionalFreqDist(data)
        lidstone_estimator = lambda fd: LidstoneProbDist(fd, 0.001, fd.B())
        self.transition_PD = ConditionalProbDist(transition_FD, lidstone_estimator)

        return self.transition_PD

    # Q2
    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self, state1, state2):
        """
        The log of the estimated probability of a transition from one state to another

        :param state1: the first state name
        :type state1: str
        :param state2: the second state name
        :type state2: str
        :return: log base 2 of the estimated transition probability
        :rtype: float
        """
        return math.log(self.transition_PD[state1].prob(state2), 2)

    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Q3
    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag and the total number of observations.
    def initialise(self, observation, number_of_observations):
        """
        Initialise data structures self.viterbi and self.backpointer for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        :param number_of_observations: the number of observations
        :type number_of_observations: int
        """
        number_of_states = len(self.states)
        # self.viterbi is a created as an N by M array of floats (as it contains logs), where N is
        # the number of states and M is the number observations (i.e. words in the sentence). Which after initialisation
        # is turned into a list to ensure the use of python floats instead on numpy.float64.
        #
        # We can then access a specific viterbi cost for a state at a step, by using self.viterbi[state][step]
        self.viterbi = np.empty((number_of_states, number_of_observations), float).tolist()
        # Like before self.backpointer is a created as an N by M array, but of ints (as it contains indexes), where N is
        # the number of states and M is the number observations (i.e. words in the sentence). Which after initialisation
        # is turned into a list to ensure the use of python ints instead on numpy.int64, though this is not strictly
        # necessary.
        #
        # We can then access a specific backpointer value for a state at a step, by using self.backpointer[state][step]
        self.backpointer = np.empty((number_of_states, number_of_observations), int).tolist()
        for i in range(number_of_states):
            # Initialise step 0 of viterbi, including
            #  transition from <s> to observation
            # use costs (- log-base-2 probabilities)
            self.viterbi[i][0] = -(self.tlprob("<s>", self.states[i]) + self.elprob(self.states[i], observation))
            # Initialise step 0 of backpointer
            self.backpointer[i][0] = 0

    # Q3
    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42
    def get_viterbi_value(self, state, step):
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The value (a cost) for state as of step
        :rtype: float
        """
        return self.viterbi[self.states.index(state)][step]

    # Q3
    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state, step):
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The state name to go back to at step-1
        :rtype: str
        """
        return self.states[self.backpointer[self.states.index(state)][step]]

    # Q4a
    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer data structures.
    # Describe your implementation with comments.
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.

        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """
        number_of_observations = len(observations)
        number_of_states = len(self.states)
        for t in range(1, number_of_observations):
            for s in range(number_of_states):
                self.viterbi[s][t] = min([(self.viterbi[s_][t - 1] - (self.tlprob(self.states[s_], self.states[s])
                                                                      + self.elprob(self.states[s], observations[t])))
                                          for s_ in range(number_of_states)])
                self.backpointer[s][t] = np.argmin(
                    [(self.viterbi[s_][t - 1] - (self.tlprob(self.states[s_], self.states[s])
                                                 + self.elprob(self.states[s], observations[t]))) for s_ in
                     range(number_of_states)])

        # Add a termination step with cost based solely on cost of transition to </s> , end of sentence.
        best_viterbi = min([(self.viterbi[s_][number_of_observations - 1]
                             - self.tlprob(self.states[s_], "</s>")) for s_ in range(number_of_states)])
        best_backpointer = np.argmin([(self.viterbi[s_][number_of_observations - 1]
                                       - self.tlprob(self.states[s_], "</s>")) for s_ in range(number_of_states)])

        # Reconstruct the tag sequence using the backpointers.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.
        tags = ["" for i in range(number_of_observations)]
        backpointer = best_backpointer
        for t in range(number_of_observations - 1, -1, -1):
            tags[t] = self.states[backpointer]
            backpointer = self.backpointer[backpointer][t]

        return tags

    def tag_sentence(self, sentence):
        """
        Initialise the HMM, lower case and tag a sentence. Returns a list of tags.
        :param sentence: the sentence
        :type sentence: list(str)
        :rtype: list(str)
        """
        observations = [word.lower() for word in sentence]
        self.initialise(observations[0], len(observations))
        return self.tag(observations)


def answer_question4b():
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :rtype: list(tuple(str,str)), list(tuple(str,str)), str
    :return: incorrectly tagged sequence, correctly tagged sequence and your answer [max 280 chars]
    """

    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand
    tagged_sequence = [("I'm", 'PRT'), ('useless', 'ADJ'), ('for', 'ADP'), ('anything', 'NOUN'), ('but', 'CONJ'),
                       ('racing', 'ADJ'), ('cars', 'NOUN'), ('.', '.')]
    correct_sequence = [("I'm", 'PRT'), ('useless', 'ADJ'), ('for', 'ADP'), ('anything', 'NOUN'), ('but', 'ADP'),
                        ('racing', 'VERB'), ('cars', 'NOUN'), ('.', '.')]
    # Why do you think the tagger tagged this example incorrectly?
    answer = inspect.cleandoc("""\
    Here we see the algorithm understanding sentence structure,but not context.Though to us it's clear it is one
    phrase,it works grammatically to have two phrases:"Im useless for anything" and "racing cars",conjoined by a
    but.And it may have seen racing cars ADJ NOUN in training""")

    return tagged_sequence, correct_sequence, trim_and_warn("Q4a", 280, answer)


# Q5a
def hard_em(labeled_data, unlabeled_data, k):
    """
    Run k iterations of hard EM on the labeled and unlabeled data.
    Follow the pseudo-code in the coursework instructions.

    :param labeled_data:
    :param unlabeled_data:
    :param k: number of iterations
    :type k: int
    :return: HMM model trained with hard EM.
    :rtype: HMM
    """
    hard_em_model = HMM(labeled_data)
    hard_em_model.train()
    for i in range(k):
        P = [list(zip(sentence, hard_em_model.tag_sentence(sentence))) for sentence in unlabeled_data]
        hard_em_model = HMM(labeled_data + P)
        hard_em_model.train()
    return hard_em_model


def answer_question5b():
    """
    Sentence:  In    fact  he    seemed   delighted  to  get   rid  of  them   .
    Gold POS:  ADP   NOUN  PRON  VERB     VERB      PRT  VERB  ADJ  ADP  PRON  .
    T_0     :  PRON  VERB  NUM    ADP     ADJ       PRT  VERB  NUM  ADP  PRON  .
    T_k     :  PRON  VERB  PRON  VERB     ADJ       PRT  VERB  NUM  ADP  NOUN  .

    1) T_0 erroneously tagged "he" as "NUM" and T_k correctly identifies it as "PRON".
        Speculate why additional unlabeled data might have helped in that case.
        Refer to the training data (inspect the 20 sentences!).
    2) Where does T_k mislabel a word but T_0 is correct? Why do you think did hard EM hurt in that case?

    :rtype: str
    :return: your answer [max 500 chars]
    """

    return trim_and_warn("Q5b", 500, inspect.cleandoc("""\
    1:Labeled data has no cases of he,but unlabelled does,so tk can learn the tag which makes the most sense
    structurally,ie PRON
    
    2:t0 tags them as PROUN,but tk tags it NOUN.tk must have found a case where its more likely a noun.As nouns
    are more common,the prob its noun increases,and each time the data is tagged,more ones are tagged as a
    noun,increasing the prob more.This shows EM working badly as a wrong tag is used to tag unlabelled data,creating
    more wrong tags and reinforcing that tag"""))


def answer_question6():
    """
    Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]
    """

    return trim_and_warn("Q6", 500, inspect.cleandoc("""\
    This could be done tagging each word,then using PCFG to create a constituent tree from the tags.This will create a
    parse(as long as POS is mostly correct),but not always the right one as words have many usages.Like 4b even if
    grammatical,still incorrect,without context(aided by lexical coverage)
    
    But the less lexical coverage PCFG has,the more it benefits from POS tagging.But if POS tagger is bad,and coverage
    is still high,it will have a negative affect,creating conflicting sentence structures"""))


def answer_question7():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """

    return trim_and_warn("Q7", 500, inspect.cleandoc("""\
    The Brown tagset contains a large amount of very specific grammatical tags.This means the algorithm will need lots
    of examples of each tag's usage to create an accurate tagger.And more tags mean less examples in the same amount of
    data,and more uncommon tags,so the Brown tagset would create a less accurate model,and worse when using EM,as you
    start with a very small sample size.To get a similar accuracy to Universal tags,you'd need a larger dataset,with
    an equivalent amount of examples of tags"""))


def compute_acc(hmm, test_data, print_mistakes):
    """
    Computes accuracy (0.0 - 1.0) of model on some data.
    :param hmm: the HMM
    :type hmm: HMM
    :param test_data: the data to compute accuracy on.
    :type test_data: list(list(tuple(str, str)))
    :param print_mistakes: whether to print the first 10 model mistakes
    :type print_mistakes: bool
    :return: float
    """
    if print_mistakes:
        to_print = 10
        print("\nFirst 10 misclassified sentences in the model:\n")
    else:
        to_print = 0
    correct = 0
    incorrect = 0
    for sentence in test_data:
        mistake = False
        s = [word for (word, tag) in sentence]
        tags = hmm.tag_sentence(s)

        for ((word, gold), tag) in zip(sentence, tags):
            if tag == gold:
                correct += 1
            else:
                incorrect += 1
                mistake = True

        if mistake and to_print > 0:
            print([(word, tag) for ((word, gold), tag) in zip(sentence, tags)])
            print(sentence, "\n")
            to_print -= 1

    return float(correct / (correct + incorrect))


# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def answers():
    global tagged_sentences_universal, test_data_universal, \
        train_data_universal, model, test_size, train_size, ttags, \
        correct, incorrect, accuracy, \
        good_tags, bad_tags, answer4b, answer5, answer6, answer7, answer5b, \
        t0_acc, tk_acc

    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    test_size = 500
    train_size = len(tagged_sentences_universal) - test_size

    # tail test set
    test_data_universal = tagged_sentences_universal[-test_size:]  # [:test_size]
    train_data_universal = tagged_sentences_universal[:train_size]  # [test_size:]
    if hashlib.md5(''.join(map(lambda x: x[0],
                               train_data_universal[0] + train_data_universal[-1] + test_data_universal[0] +
                               test_data_universal[-1])).encode(
        'utf-8')).hexdigest() != '164179b8e679e96b2d7ff7d360b75735':
        print('!!!test/train split (%s/%s) incorrect -- this should not happen, please contact a TA !!!' % (
            len(train_data_universal), len(test_data_universal)), file=sys.stderr)

    # nltk.help.brown_tagset()
    # Create instance of HMM class and initialise the training set.
    model = HMM(train_data_universal)

    # Train the HMM.
    model.train()

    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample = model.elprob('VERB', 'is')
    if not (type(e_sample) == float and e_sample <= 0.0):
        print('elprob value (%s) must be a log probability' % e_sample, file=sys.stderr)

    t_sample = model.tlprob('VERB', 'VERB')
    if not (type(t_sample) == float and t_sample <= 0.0):
        print('tlprob value (%s) must be a log probability' % t_sample, file=sys.stderr)

    if not (type(model.states) == list and \
            len(model.states) > 0 and \
            type(model.states[0]) == str):
        print('model.states value (%s) must be a non-empty list of strings' % model.states, file=sys.stderr)

    print('states: %s\n' % model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s = 'the cat in the hat came back'.split()
    ttags = model.tag_sentence(s)
    print("Tagged a trial sentence:\n  %s" % list(zip(s, ttags)))

    v_sample = model.get_viterbi_value('VERB', 5)
    if not (type(v_sample) == float and 0.0 <= v_sample):
        print(type(v_sample))
        print('viterbi value (%s) must be a cost' % v_sample, file=sys.stderr)

    b_sample = model.get_backpointer_value('VERB', 5)
    if not (type(b_sample) == str and b_sample in model.states):
        print('backpointer value (%s) must be a state name' % b_sample, file=sys.stderr)

    # check the model's accuracy (% correct) using the test set
    accuracy = compute_acc(model, test_data_universal, print_mistakes=True)
    print('\nTagging accuracy for test set of %s sentences: %.4f' % (test_size, accuracy))

    # Tag the sentence again to put the results in memory for automarker.
    model.tag_sentence(s)

    # Question 5a
    # Set aside the first 20 sentences of the training set
    num_sentences = 20
    semi_supervised_labeled = train_data_universal[:num_sentences]
    # type list(list(tuple(str, str)))
    semi_supervised_unlabeled = [[word for (word, tag) in sent] for sent in train_data_universal[num_sentences:]]
    # type list(list(str))
    print("Running hard EM for Q5a. This may take a while...")
    # for sentence in semi_supervised_labeled:
    #     print(sentence)
    t0 = hard_em(semi_supervised_labeled, semi_supervised_unlabeled, 0)  # 0 iterations
    tk = hard_em(semi_supervised_labeled, semi_supervised_unlabeled, 3)
    print("done.")

    t0_acc = compute_acc(t0, test_data_universal, print_mistakes=True)
    tk_acc = compute_acc(tk, test_data_universal, print_mistakes=True)
    print('\nTagging accuracy of T_0: %.4f' % (t0_acc))
    print('\nTagging accuracy of T_k: %.4f' % (tk_acc))
    ########

    # Print answers for 4b, 5b, 6 and 7.
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nA tagged-by-your-model version of a sentence:')
    print(bad_tags)
    print('The tagged version of this sentence from the corpus:')
    print(good_tags)
    print('\nDiscussion of the difference:')
    print(answer4b)
    print(len(answer4b))
    answer5b = answer_question5b()
    print("\nFor Q5b:")
    print(answer5b)
    print(len(answer5b))
    answer6 = answer_question6()
    print('\nFor Q6:')
    print(answer6)
    print(len(answer6))
    answer7 = answer_question7()
    print('\nFor Q7:')
    print(answer7)
    print(len(answer7))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        import adrive2
        from autodrive_embed import run, carefulBind

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive2.a2answers, errlog)
    else:
        answers()

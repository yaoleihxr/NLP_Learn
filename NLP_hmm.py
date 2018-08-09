# /usr/bin/python
# -*- encoding:utf-8 -*-

import sys
import nltk
from nltk.corpus import brown


# 准备语料，在头尾添加start、end
def prepare_corpus():
    brown_tags_words = []
    for sent in brown.tagged_sents():
        brown_tags_words.append(('START', 'START'))
        brown_tags_words.extend([(tag[:2], word) for (word, tag) in sent])
        brown_tags_words.append(('END', 'END'))
    return brown_tags_words


# 计算conditional probability distribution: P(wi|ti)=count(wi,ti)/count(ti)
def cal_cpd_tagwords(tags_words):
    cfd_tagwords = nltk.ConditionalFreqDist(tags_words)
    cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)
    print("The probability of an adjective (JJ) being 'new' is", cpd_tagwords["JJ"].prob("new"))
    print("The probability of a verb (VB) being 'duck' is", cpd_tagwords["VB"].prob("duck"))
    return cpd_tagwords


# 计算P(ti|t{i-1})=count(t{i-1},ti)/count(t{i-1})
def cal_cpd_tags(tags_words):
    brown_tags = [tag for (tag, word) in tags_words]
    cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
    cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)
    print("If we have just seen 'DT', the probability of 'NN' is", cpd_tags["DT"].prob("NN"))
    print("If we have just seen 'VB', the probability of 'JJ' is", cpd_tags["VB"].prob("DT"))
    print("If we have just seen 'VB', the probability of 'NN' is", cpd_tags["VB"].prob("NN"))
    return cpd_tags, list(set(brown_tags))


# viterbi算法
def hmm_viterbi(cpd_tags, cpd_tagwords, tags, sent):
    viterbi = []
    backpointer = []

    # 计算第一项，从START开始
    first_viterbi = {}
    first_backpointer = {}
    for tag in tags:
        if tag != 'START':
            first_viterbi[tag] = cpd_tags['START'].prob(tag) * cpd_tagwords[tag].prob(sent[0])
            first_backpointer[tag] = 'START'

    viterbi.append(first_viterbi)
    backpointer.append(first_backpointer)
    currbest = max(first_viterbi.keys(), key=lambda tag: first_viterbi[tag])
    print("Word", "'" + sent[0] + "'", "current best two-tag sequence:", first_backpointer[currbest], currbest)

    for wordindex in range(1, len(sent)):
        this_viterbi = {}
        this_backpointer = {}
        prev_viterbi = viterbi[-1]
        for tag in tags:
            if tag == 'START':
                continue
            best_previous = max(prev_viterbi.keys(), key=lambda prevtag:
                                prev_viterbi[prevtag] * cpd_tags[prevtag].prob(tag) *
                                cpd_tagwords[tag].prob(sent[wordindex]))
            this_viterbi[tag] = prev_viterbi[best_previous] * cpd_tags[best_previous].prob(tag) * \
                                cpd_tagwords[tag].prob(sent[wordindex])
            this_backpointer[tag] = best_previous
        currbest = max(this_viterbi.keys(), key=lambda tag: this_viterbi[tag])
        print("Word", "'" + sent[wordindex] + "'", "current best two-tag sequence:", this_backpointer[currbest], currbest)
        viterbi.append(this_viterbi)
        backpointer.append(this_backpointer)

    # 计算END,当前的backpionter尚未计算最后一个单词的tag
    prev_viterbi = viterbi[-1]
    best_previous = max(prev_viterbi.keys(), key=lambda prevtag : prev_viterbi[prevtag] * cpd_tags[prevtag].prob('END'))
    prob_tagsequence = prev_viterbi[best_previous] * cpd_tags[best_previous].prob('END')
    best_tagsequence = ['END', best_previous]
    backpointer.reverse()
    current_best_tag = best_previous
    for bp in backpointer:
        best_tagsequence.append(bp[current_best_tag])
        current_best_tag = bp[current_best_tag]
    best_tagsequence.reverse()
    return best_tagsequence, prob_tagsequence


if __name__ == '__main__':
    brown_tags_words = prepare_corpus()
    cpd_tagwords = cal_cpd_tagwords(brown_tags_words)
    cpd_tags, tags = cal_cpd_tags(brown_tags_words)
    # print(brown_tags_words)
    # print(tags)
    sentence = ['I', 'want','to', 'race']
    best_seq, prob_seq = hmm_viterbi(cpd_tags, cpd_tagwords, tags, sentence)
    print(best_seq)
    print(prob_seq)
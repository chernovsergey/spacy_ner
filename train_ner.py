from __future__ import unicode_literals, print_function
import json
import pathlib
import random

import spacy
from spacy.pipeline import EntityRecognizer
from spacy.gold import GoldParse
from spacy.tagger import Tagger
import codecs
import subprocess
import stat
import os

try:
    unicode
except:
    unicode = str


def conlleval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += u'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + u' ' + wl + u' ' + wp + '\n'
        out += u'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out[:-1])  # remove the ending \n on last line
    f.close()

    return get_perf(filename)


def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.dirname(os.path.realpath(__file__)) + '/conlleval.pl'
    os.chmod(_conlleval, stat.S_IRWXU)  # give the execute permissions

    proc = subprocess.Popen(["perl",
                             _conlleval],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    stdout, _ = proc.communicate(''.join(open(filename).readlines()))
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall = float(out[8][:-2])
    f1score = float(out[10])

    return {'p': precision, 'r': recall, 'f1': f1score}


def train(nlp, data, ents, num_iterations=20):
    """

    :param nlp: nlp instance
    :param data: training data(look at required format below)
    :param ents: list of entities
    :param num_iterations: number iterations to train
    :return: trained NER tagger
    """

    # Example :
    # train_data = [
    #     (
    #         'Who is Shaka Khan?',
    #         [(len('Who is '), len('Who is Shaka Khan'), 'PERSON')]
    #     ), ...
    # ]

    for sent, _ in data:
        doc = nlp.make_doc(sent)
        for word in doc:
            _ = nlp.vocab[word.orth]

    result_NER = EntityRecognizer(nlp.vocab, entity_types=ents)
    for _ in range(num_iterations):
        random.shuffle(data)
        for sent, entity_offsets in data:
            doc = nlp.make_doc(sent)
            gold = GoldParse(doc, entities=entity_offsets)
            result_NER.update(doc, gold)
    return result_NER


def save_model(ner, model_dir):
    """

    :param ner: NER tagger instance
    :param model_dir: path
    """
    model_dir = pathlib.Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()
    assert model_dir.is_dir()

    with (model_dir / 'config.json').open('wb') as file_:
        data = json.dumps(ner.cfg)
        if isinstance(data, unicode):
            data = data.encode('utf8')
        file_.write(data)
    ner.model.dump(str(model_dir / 'model'))
    if not (model_dir / 'vocab').exists():
        (model_dir / 'vocab').mkdir()
    ner.vocab.dump(str(model_dir / 'vocab' / 'lexemes.bin'))
    with (model_dir / 'vocab' / 'strings.json').open('w', encoding='utf8') as file_:
        ner.vocab.strings.dump(file_)


def load_data(prefix):
    """

    :param prefix: train/test/valid
    :return: list of sentences, list of annotations for each sentence
    """
    with codecs.open("./data/ATIS_sample/{0}/{0}.seq.in".format(prefix), "r", encoding="utf-8") as seq_in:
        seq_l = map(lambda x: x.strip(), seq_in.readlines())

    with codecs.open("./data/ATIS_sample/{0}/{0}.seq.out".format(prefix), "r", encoding="utf-8") as seq_out:
        seq_r = map(lambda x: x.strip().upper(), seq_out.readlines())

    return seq_l, seq_r


def make_train_data():
    """

    :return: shaped training data(without words annotated as O, _UNK and _PAD)
    """

    seq_l, seq_r = load_data("train")
    assert len(seq_l) == len(seq_r)
    train_data = []

    for entry_l, entry_r in zip(seq_l, seq_r):
        annotation_list = []
        for word, annot in zip(entry_l.split(), entry_r.split()):
            if annot in ["O", "_UNK", "_PAD"]:
                continue
            word_pos = entry_l.find(word)
            assert word == entry_l[word_pos:word_pos + len(word)]
            annotation_list.append((word_pos, word_pos + len(word), annot))

        train_data.append((entry_l, annotation_list))

    return train_data


def load_entyty_types():
    """

    :return: list of entity types
    """
    f = codecs.open("./data/ATIS_sample/out_vocab_10000.txt", "r", encoding="utf-8")
    entities = map(lambda x: x.strip().upper(), f.readlines())
    f.close()
    return entities


def test_model(nlp, ner):
    """

    :param nlp: NLP instance
    :param ner: NER instance
    :return:
    """
    seq_l, seq_r = load_data("test")

    hyp_tag_list = []
    ref_tag_list = []
    word_list = []
    for sent, annot in zip(seq_l, seq_r):
        annot.replace("_UNK", "O")
        annot.replace("_PAD", "O")

        doc = nlp.make_doc(sent)
        nlp.tagger(doc)
        ner(doc)

        word_list.append(sent.split())
        hyp_tag_list.append([w.ent_type_ if w.ent_type_ != u'' else u"O" for w in doc])
        ref_tag_list.append(annot.split())

    tagging_eval_result = conlleval(hyp_tag_list, ref_tag_list, word_list, "./tagging.test.hyp.txt")
    print("f1-score: %.2f" % (tagging_eval_result['f1']))
    print("precision: %.2f" % tagging_eval_result["p"])
    print("recall: %.2f" % tagging_eval_result["r"])


def main(model_dir=None):
    train_data = make_train_data()
    entity_types = load_entyty_types()

    nlp = spacy.load('en', parser=False, entity=False, add_vectors=False)
    if nlp.tagger is None:
        nlp.tagger = Tagger(nlp.vocab, features=Tagger.feature_templates)

    ner = train(nlp, train_data, entity_types, 20)

    # small test
    doc = nlp.make_doc(u'is there a delta flight from denver to san francisco')
    nlp.tagger(doc)
    ner(doc)
    for word in doc:
        print(word.text, word.ent_type_)
    #

    test_model(nlp, ner)

    if model_dir is not None:
        save_model(ner, model_dir)


if __name__ == '__main__':
    main("./ATIS_trained_NER")

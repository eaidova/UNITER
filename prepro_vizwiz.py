"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess VizWiz annotations into LMDB
"""
import argparse
import json
import pickle
import copy
import os
import re
from collections import Counter
from os.path import exists

from cytoolz import curry
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

from data.data import open_lmdb


@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


def process_vizwiz(json_data, db, tokenizer, vocab):
    def prepare_answer(answers):
        prepared_sample_answers = []
        for answer in answers:
            answer = answer.lower()

            # define desired replacements here
            punctuation_dict = {'.': ' ', "'": '', '?': ' ', '_': ' ', '-': ' ', '/': ' ', ',': ' '}

            rep = punctuation_dict
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            answer = pattern.sub(lambda m: rep[re.escape(m.group(0))], answer)
            # if answer in ['unsuitable', 'unsuitable image', 'unanswerable', 'too blurry']:
            #     prepared_sample_answers.append('unanswerable')
            prepared_sample_answers.append(answer)

        return prepared_sample_answers

    id2len = {}
    txt2img = {}  # not sure if useful
    for example in tqdm(json_data, desc='processing VizWiz'):
        img_id = example['image'].rsplit('.')[0]
        id_ = img_id
        img_fname = f'vizwiz_{img_id}.npz'
        input_ids_ru = tokenizer(example['question_ru'])
        input_ids = tokenizer(example['question'])
        if 'answers' in example:
            answers = example['answers']
            labels, scores = [], []
            count_answ = Counter(prepare_answer([ans['answer'] for ans in answers]))
            for answer, score in count_answ.items():
                if answer not in vocab:
                    continue
                labels.append(vocab[answer])
                scores.append(min(1, score / 3))
            target = {'labels': labels, 'scores': scores}
        else:
            target = None
        txt2img[id_] = img_fname
        id2len[id_] = len(input_ids)
        example['input_ids'] = input_ids
        example['img_fname'] = img_fname
        example['target'] = target
        example['question_type'] = example['answer_type']
        db[id_] = example
        example_ru = copy.deepcopy(example)
        example_ru['input_ids'] = input_ids_ru
        db[id_ + '_ru'] = example_ru
        id2len[id_+'_ru'] = len(input_ids_ru)
        txt2img[id_ + '_ru'] = img_fname

    return id2len, txt2img

def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    toker = BertTokenizer.from_pretrained(
        opts.toker, do_lower_case='uncased' in opts.toker)
    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        with open(opts.vocab) as v:
            vocab = json.load(v)
        with open(opts.annotation) as f:
            ann = json.load(f)
            id2lens, txt2img = process_vizwiz(ann, db, tokenizer, vocab)

    with open(f'{opts.output}/id2len.json', 'w') as f:
        json.dump(id2lens, f)
    with open(f'{opts.output}/txt2img.json', 'w') as f:
        json.dump(txt2img, f)
    with open(f'{opts.output}/ans2label.pkl', 'wb') as pkl:
        pickle.dump(vocab, pkl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--missing_imgs',
                        help='some training image features are corrupted')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()
    main(args)

""" Usage:
    --ds=DATASET_FILE --bi=IN_FILE --align=ALIGN_FILE --out=OUT_FILE --lang=LANG  [--debug]
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
import sys
from docopt import docopt
from collections import defaultdict, Counter
from operator import itemgetter
from tqdm import tqdm
from typing import List, Optional, Union
import csv
import re

# Local imports
from languages.spacy_support import SpacyPredictor
# from languages.german import GermanPredictor
from languages.gendered_article import GenderedArticlePredictor, \
    get_german_determiners, GERMAN_EXCEPTION, get_french_determiners
from languages.pymorph_support import PymorphPredictor
from languages.semitic_languages import HebrewPredictor, ArabicPredictor
from languages.morfeusz_support import MorfeuszPredictor
from evaluate import evaluate_bias
from languages.czech import CzechPredictor
#=-----
ENGLISH_ONLY = False

LANGAUGE_PREDICTOR = {
    "es": lambda: SpacyPredictor("es_core_news_sm"),
    "fr": lambda: SpacyPredictor("fr"),
    "it": lambda: SpacyPredictor("it"),
    "ru": lambda: PymorphPredictor("ru"),
    "uk": lambda: PymorphPredictor("uk"),
    "he": lambda: HebrewPredictor(),
    "ar": lambda: ArabicPredictor(),
    "de": lambda: GenderedArticlePredictor("de", get_german_determiners, GERMAN_EXCEPTION),
    "cs": lambda: CzechPredictor(),
    "pl": lambda: MorfeuszPredictor(),
}

def no_punct(sentence: str):
    clean = re.sub(r"""
                [,.;@#?!&$]+  # Accept one or more copies of punctuation
                \ *           # plus zero or more copies of a space,
                """,
                " ",          # and replace it with a single space
                sentence, flags=re.VERBOSE)
    return clean

def get_src_indices(instance: List[str]) -> List[int]:
    """
    (English)
    Determine a list of source side indices pertaining to a
    given instance in the dataset. # TODO nederlands moet ook het lidwoord erbij
    """
    _, src_word_ind, sent = instance[: 3]
    src_word_ind = int(src_word_ind)
    sent_tok = sent.split(" ")
    articles = ["the", "an", "a", "de", "het", "een"]
    if ENGLISH_ONLY:
        articles = articles[:3]
    if (src_word_ind > 0) and (sent_tok[src_word_ind - 1].lower() in articles):
        src_indices = [src_word_ind -1]
    else:
        src_indices = []
    src_indices.append(src_word_ind)

    return src_indices

def get_translated_professions(alignment_fn: str, ds: List[List[str]], bitext: List[List[str]]) -> List[str]:
    """
    (Language independent)
    Load alignments from file and return the translated profession according to
    source indices.
    """
    # Load files and data structures
    ds_src_sents = list(map(itemgetter(2), ds))
    bitext_src_sents = [src_sent for ind, (src_sent, tgt_sent) in bitext]

    # Sanity checks
    assert len(ds) == len(bitext)
    mismatched = [ind for (ind, (ds_src_sent, bitext_src_sent)) in enumerate(zip(ds_src_sents, bitext_src_sents))
                  if ds_src_sent != bitext_src_sent]
    if len(mismatched) != 0:
        raise AssertionError

    bitext = [(ind, (src_sent.split(), tgt_sent.split()))
              for ind, (src_sent, tgt_sent) in bitext]

    src_indices = list(map(get_src_indices, ds)) # add articles in the mix

    full_alignments = []
    for line in open(align_fn):
        cur_align = defaultdict(list)
        for word in line.split():
            src, tgt = word.split("-")
            cur_align[int(src)].append(int(tgt))
        full_alignments.append(cur_align)


    bitext_inds = [ind for ind, _ in bitext]

    alignments = []
    for ind in bitext_inds:
        alignments.append(full_alignments[ind])


    assert len(bitext) == len(alignments)
    assert len(src_indices) == len(alignments)

    translated_professions = []
    target_indices = []

    for (_, (src_sent, tgt_sent)), alignment, cur_indices in tqdm(zip(bitext, alignments, src_indices), disable=False):
        # cur_translated_profession = " ".join([tgt_sent[cur_tgt_ind]
        #                                       for src_ind in cur_indices
        #                                       for cur_tgt_ind in alignment[src_ind]])
        cur_tgt_inds = ([cur_tgt_ind
                         for src_ind in cur_indices
                         for cur_tgt_ind in alignment[src_ind]])

        cur_translated_profession = " ".join([tgt_sent[cur_tgt_ind]
                                              for cur_tgt_ind in cur_tgt_inds])
        print_prof_alignment(src_sent, tgt_sent, cur_indices, cur_tgt_inds, file_name="output/highlights_met_lidwoorden.txt")
        target_indices.append(cur_tgt_inds)
        translated_professions.append(cur_translated_profession)

    return translated_professions, target_indices


def output_predictions(target_sentences, gender_predictions, out_fn):
    """
    Write gender predictions to output file, for comparison
    with human judgments.
    """
    assert(len(list(target_sentences)) == len(list(gender_predictions)))
    with open(out_fn, "w", encoding = "utf8") as fout:
        writer = csv.writer(fout, delimiter=",")
        writer.writerow(["Sentence", "Predicted gender"])
        for sent, gender in zip(target_sentences, gender_predictions):
            writer.writerow([sent, str(gender).split(".")[1]])

def align_bitext_to_ds(bitext, ds):
    """
    Return a subset of bitext that's aligned to ds.
    """
    bitext_dict = dict([(src, (ind, tgt)) for ind, (src, tgt) in enumerate(bitext)])
    new_bitext = []
    for entry in ds:
        en_sent = entry[2]
        ind, tgt_sent = bitext_dict[en_sent]
        new_bitext.append((ind, (en_sent, tgt_sent)))
    return new_bitext

def highlight_words(sentence: Union[list, str], word_indices: list, highlight_char='^'):
    if isinstance(sentence, str):
        words = no_punct(sentence).split()
    elif isinstance(sentence, list):
        words = sentence
        sentence = ' '.join(words)
    else:
        raise TypeError
    word_starts = [0]
    for word in words:
        word_starts.append(word_starts[-1]+len(word)+1)
    lint = [' ']*len(sentence) 
    for word_idx in word_indices:
        start = word_starts[word_idx]
        stop = word_starts[word_idx+1] -1
        for i in range(start, stop):
            lint[i] = highlight_char
    return ''.join(lint)    

def print_prof_alignment(source_sentence, target_sentence, source_idx: list, target_idx: list, file_name: Optional[str]=None):
    if file_name is None:
        out = sys.stdout
    else:
        out = open(file_name, 'a')
    if isinstance(source_sentence, list):
        print(' '.join(source_sentence), file=out)
    else:
        print(source_sentence, file=out)
    print(highlight_words(source_sentence, source_idx), file=out)
    if isinstance(target_sentence, list):
        print(' '.join(target_sentence), file=out)
    else:
        print(target_sentence, file=out)
    print(highlight_words(target_sentence, target_idx), file=out)
    print('='*50 + '\n', file=out)
    out.close()


if __name__ == "__main__":
    # Parse command line arguments
    # args = docopt(__doc__)
    # ds_fn = args["--ds"]
    # bi_fn = args["--bi"]
    # align_fn = args["--align"]
    # out_fn = args["--out"]
    # lang = args["--lang"]
    #
    # probeer alignments in woorden te outputten om alignment te checken/debuggen

    ds_fn = "data\\aggregates\\nl.txt"
    bi_fn = "translations\\deepl\\nl-es.txt"
    align_fn = "src\\forward.nl-es.align"
    out_fn = "output\\debugout.txt"
    lang = "es"

    debug = True # args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    gender_predictor = LANGAUGE_PREDICTOR[lang]()

    ds = [line.strip().split("\t") for line in open(ds_fn, encoding = "utf8")]
    full_bitext = [line.strip().split(" ||| ")
              for line in open(bi_fn, encoding = "utf8")]
    bitext = align_bitext_to_ds(full_bitext, ds)

    translated_profs, tgt_inds = get_translated_professions(align_fn, ds, bitext)
    assert(len(translated_profs) == len(tgt_inds))

    target_sentences = [tgt_sent for (ind, (src_sent, tgt_sent)) in bitext]

    gender_predictions = [gender_predictor.get_gender(prof, translated_sent, entity_index, ds_entry)
                          for prof, translated_sent, entity_index, ds_entry
                          in tqdm(zip(translated_profs,
                                      target_sentences,
                                      map(lambda ls:min(ls, default = -1), tgt_inds), # weirddd min
                                      ds))]

    # Output predictions
    output_predictions(target_sentences, gender_predictions, out_fn)

    d = evaluate_bias(ds, gender_predictions)


    logging.info("DONE")

import os

import pytest

from bertalign import Bertalign
from bertalign.eval import read_alignments
from bertalign.eval import score_multiple
from bertalign.eval import log_final_scores


def align_text_and_berg(filespec, aligner_spec):
    r"""Align Text and Berg using the original aligner."""

    test_alignments = []
    gold_alignments = []

    results = {}

    for test_data in filespec:

        file, src_file, tgt_file, gold_dir  = test_data
        src = open(src_file, 'rt', encoding='utf-8').read()
        tgt = open(tgt_file, 'rt', encoding='utf-8').read()

        print("Start aligning {} to {}".format(src_file, tgt_file))
        # aligner = Bertalign(src, tgt, is_split=True)
        aligner = Bertalign(src, tgt, **aligner_spec)
        aligner.align_sents()
        test_alignments.append(aligner.result)

        gold_file = os.path.join(gold_dir, file)
        gold_alignments.append(read_alignments(gold_file))

        scores = score_multiple(gold_list=gold_alignments, test_list=test_alignments)
        log_final_scores(scores)
        results[file] = scores
    return results


@pytest.mark.skip(reason="is_split is removed at the moment.")
def test_aligner_original(text_and_berg_expected_results, text_and_berg_inputs):
    r"""Test results for the original aligner using is_split."""

    aligner_spec = {"is_split": True}
    result = align_text_and_berg(text_and_berg_inputs, aligner_spec)

    for file in result:
        expected = text_and_berg_expected_results[file]
        calculated = result[file]
        for metric in expected:
            assert expected[metric] == calculated[metric], "Result mismatch"


def test_aligner_altered_parametrization(text_and_berg_expected_results, text_and_berg_inputs):
    r"""Test results for the aligner using input_type and languages."""

    aligner_spec = {"input_type": 'lines', 'src_lang': 'de', 'tgt_lang': 'fr'}
    result = align_text_and_berg(text_and_berg_inputs, aligner_spec)

    for file in result:
        expected = text_and_berg_expected_results[file]
        calculated = result[file]
        for metric in expected:
            assert expected[metric] == calculated[metric], "Result mismatch"

import pytest
import json
import os

def load_json(fpath):
    with open(fpath) as json_file:
        data = json.load(json_file)
    return data


@pytest.fixture
def text_and_berg_expected_results():
    """Fixture for the Text und Berg expected result."""

    cur_dir =  os.path.dirname(os.path.realpath(__file__))
    fname = 'gold_standard_text_und_berg.json'
    fpath = os.path.join(cur_dir, fname)
    data = load_json(fpath)
    yield data



@pytest.fixture
def text_and_berg_inputs():
    r"""Input data for Text and Berg."""

    src_dir = 'text+berg/de'
    tgt_dir = 'text+berg/fr'
    gold_dir = 'text+berg/gold'

    data = []
    for file in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file).replace("\\","/")
        tgt_file = os.path.join(tgt_dir, file).replace("\\","/")
        data.append((file, src_file, tgt_file, gold_dir))

    yield data

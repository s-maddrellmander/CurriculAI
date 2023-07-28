import os
import csv
import json
import pytest
from reformat import JSONFormatter

# Create fixtures for test data

@pytest.fixture(params=[
    {"prose": "Test prose."},
    {"prose": "Another test prose."}
])
def prose_data(request):
    return request.param

@pytest.fixture(params=[
    [{"question": "q1", "answer": "a1"}],
    [{"question": "q2", "answer": "a2"}, {"question": "q3", "answer": "a3"}]
])
def flashcard_data(request):
    return request.param

@pytest.fixture(params=[
    [{"question": "q1", "options": ["o1", "o2"], "answer": "a1"}],
    [{"question": "q2", "options": ["o1", "o2", "o3"], "answer": "a2"}]
])
def mcq_data(request):
    return request.param

def test_load_json():
    data = {"prose": "Test prose.", "flashcards": [{"question": "q1", "answer": "a1"}], "mcq": [{"question": "q2", "options": ["o1", "o2"], "answer": "a2"}]}
    with open('test.json', 'w') as f:
        json.dump(data, f)

    formatter = JSONFormatter('test.json')
    assert formatter.load_json() == data

    os.remove('test.json')

def test_save_as_txt(prose_data):
    formatter = JSONFormatter(None)
    formatter.save_as_txt(prose_data['prose'], 'test.txt')

    with open('test.txt', 'r') as f:
        assert f.read() == prose_data['prose']

    os.remove('test.txt')

def test_save_flashcards_as_csv(flashcard_data):
    formatter = JSONFormatter(None)
    formatter.save_flashcards_as_csv(flashcard_data, 'flashcards.csv')

    with open('flashcards.csv', 'r') as f:
        reader = csv.reader(f)
        assert list(reader) == [['Question', 'Answer']] + [[d['question'], d['answer']] for d in flashcard_data]

    os.remove('flashcards.csv')

def test_save_mcq_as_csv(mcq_data):
    formatter = JSONFormatter(None)
    formatter.save_mcq_as_csv(mcq_data, 'mcq.csv')

    with open('mcq.csv', 'r') as f:
        reader = csv.reader(f)
        assert list(reader) == [['Question', 'Options', 'Answer']] + [[d['question'], ', '.join(d['options']), d['answer']] for d in mcq_data]

    os.remove('mcq.csv')

@pytest.mark.parametrize("prose,flashcards,mcq", [
    ('Test prose.', [{"question": "q1", "answer": "a1"}], [{"question": "q2", "options": ["o1", "o2"], "answer": "a2"}]),
    ('Another prose.', [{"question": "q3", "answer": "a3"}], [{"question": "q4", "options": ["o3", "o4"], "answer": "a4"}]),
])
def test_process_data(prose, flashcards, mcq):
    data = {"prose": prose, "flashcards": flashcards, "mcq": mcq}
    with open('test.json', 'w') as f:
        json.dump(data, f)

    formatter = JSONFormatter('test.json')
    formatter.process_data('test.txt', 'flashcards.csv', 'mcq.csv')

    # Assert files exist
    assert os.path.exists('test.txt')
    assert os.path.exists('flashcards.csv')
    assert os.path.exists('mcq.csv')

    os.remove('test.txt')
    os.remove('flashcards.csv')
    os.remove('mcq.csv')
    os.remove('test.json')

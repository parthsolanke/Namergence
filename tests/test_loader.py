from utils import loader


def test_unicode_to_ascii_strips_accents():
    assert loader.unicode_to_ascii("José") == "Jose"


def test_unicode_to_ascii_drops_unsupported_characters():
    assert loader.unicode_to_ascii("李雷") == ""


def test_unicode_to_ascii_keeps_allowed_punctuation():
    assert loader.unicode_to_ascii("O'Brien, Jr.") == "O'Brien, Jr."


def test_letter_to_index_matches_all_letters():
    for letter in loader.ALL_LETTERS:
        assert loader.letter_to_index(letter) == loader.ALL_LETTERS.index(letter)


def test_letter_to_index_unseen_letter_returns_minus_one():
    assert loader.letter_to_index("李") == -1


def test_letter_to_tensor_builds_one_hot_vector():
    tensor = loader.letter_to_tensor("a")

    assert tensor.shape == (1, loader.N_LETTERS)
    assert tensor.sum().item() == 1
    assert tensor[0][loader.letter_to_index("a")].item() == 1


def test_line_to_tensor_builds_one_hot_vector_per_character():
    line = "Ab"
    tensor = loader.line_to_tensor(line)

    assert tensor.shape == (len(line), 1, loader.N_LETTERS)
    for index, letter in enumerate(line):
        assert tensor[index][0].sum().item() == 1
        assert tensor[index][0][loader.letter_to_index(letter)].item() == 1


def test_random_training_example_returns_matching_category_and_tensors(monkeypatch):
    names = {"English": ["Alice", "Bob"], "French": ["Jean"]}
    categories = ["English", "French"]
    monkeypatch.setattr(loader.random, "choice", lambda options: options[0])

    category, line, category_tensor, line_tensor = loader.random_training_example(
        names, categories
    )

    assert category == "English"
    assert line == "Alice"
    assert category_tensor.item() == 0
    assert line_tensor.shape == (len(line), 1, loader.N_LETTERS)


def test_load_data_reads_and_normalises_name_files(tmp_path, monkeypatch):
    english = tmp_path / "English.txt"
    english.write_text("José\nO'Brien\n", encoding="utf-8")
    french = tmp_path / "French.txt"
    french.write_text("René\n", encoding="utf-8")

    monkeypatch.setattr(loader.glob, "glob", lambda pattern: [str(english), str(french)])

    names, categories = loader.load_data()

    assert categories == ["English", "French"]
    assert names["English"] == ["Jose", "O'Brien"]
    assert names["French"] == ["Rene"]

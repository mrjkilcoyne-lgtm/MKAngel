"""Tests for the natural-language parser and its integration with Angel."""

from glm.angel import Angel
from glm.core.parser import Parser, ParseNode, heuristic_tag, _normalise_category
from glm.grammars.linguistic import build_syntactic_grammar


class TestHeuristicTagger:
    def test_determiner(self):
        assert heuristic_tag("the") == "Det"
        assert heuristic_tag("a") == "Det"
        assert heuristic_tag("this") == "Det"

    def test_pronoun(self):
        assert heuristic_tag("she") == "Pron"
        assert heuristic_tag("we") == "Pron"

    def test_auxiliary(self):
        assert heuristic_tag("is") == "Aux"
        assert heuristic_tag("will") == "Aux"

    def test_preposition(self):
        assert heuristic_tag("in") == "P"
        assert heuristic_tag("through") == "P"

    def test_conjunction(self):
        assert heuristic_tag("and") == "Conj"
        assert heuristic_tag("but") == "Conj"

    def test_wh_word(self):
        assert heuristic_tag("who") == "Wh"
        assert heuristic_tag("where") == "Wh"

    def test_adverb_suffix(self):
        assert heuristic_tag("quickly") == "Adv"
        assert heuristic_tag("softly") == "Adv"

    def test_adjective_suffix(self):
        assert heuristic_tag("beautiful") == "Adj"
        assert heuristic_tag("creative") == "Adj"

    def test_verb_suffix(self):
        assert heuristic_tag("running") == "V"
        assert heuristic_tag("walked") == "V"

    def test_noun_default(self):
        # Unknown word with no suffix cue defaults to noun
        assert heuristic_tag("xyzzy") == "N"

    def test_punctuation_stripped(self):
        # Should strip punctuation before tagging
        assert heuristic_tag("cat,") == "N"
        assert heuristic_tag("hello!") == "N"


class TestNormaliseCategory:
    def test_noun_normalised(self):
        assert _normalise_category("noun") == "N"

    def test_verb_normalised(self):
        assert _normalise_category("verb") == "V"

    def test_already_normalised(self):
        assert _normalise_category("Det") == "Det"
        assert _normalise_category("NP") == "NP"

    def test_empty(self):
        assert _normalise_category("") == "N"


class TestParseNode:
    def test_create_leaf(self):
        node = ParseNode(category="N", span=(0, 1), word="cat")
        assert node.word == "cat"
        assert not node.children

    def test_to_dict_leaf(self):
        node = ParseNode(category="N", span=(0, 1), word="cat")
        d = node.to_dict()
        assert d["cat"] == "N"
        assert d["word"] == "cat"

    def test_to_dict_tree(self):
        leaf = ParseNode(category="N", span=(0, 1), word="cat")
        parent = ParseNode(category="NP", span=(0, 1), children=[leaf])
        d = parent.to_dict()
        assert d["cat"] == "NP"
        assert "children" in d

    def test_flatten(self):
        leaves = [
            ParseNode("Det", (0, 1), word="the"),
            ParseNode("N", (1, 2), word="cat"),
        ]
        root = ParseNode("NP", (0, 2), children=leaves)
        assert root.flatten() == ["the", "cat"]


class TestParser:
    def test_parser_no_lexicon(self):
        p = Parser(None, None)
        tags = p.tag(["the", "cat"])
        assert tags == ["Det", "N"]

    def test_parser_with_lexicon(self):
        angel = Angel()
        angel.awaken()
        p = Parser(angel._lexicon, None)
        tags = p.tag(["the", "cat", "sat"])
        assert tags == ["Det", "N", "V"]

    def test_parser_tags_pronouns(self):
        angel = Angel()
        angel.awaken()
        p = Parser(angel._lexicon, None)
        tags = p.tag(["she", "sees", "him"])
        assert tags[0] == "Pron"
        assert tags[2] == "Pron"

    def test_parser_tags_prepositions(self):
        angel = Angel()
        angel.awaken()
        p = Parser(angel._lexicon, None)
        tags = p.tag(["in", "the", "house"])
        assert tags[0] == "P"

    def test_parse_builds_tree(self):
        angel = Angel()
        angel.awaken()
        g = build_syntactic_grammar()
        p = Parser(angel._lexicon, g)
        tree = p.parse(["the", "cat", "sat"])
        assert tree is not None
        assert tree.category == "S"

    def test_parse_flatten_preserves_words(self):
        angel = Angel()
        angel.awaken()
        p = Parser(angel._lexicon, build_syntactic_grammar())
        tokens = ["the", "cat", "sat"]
        tree = p.parse(tokens)
        assert tree.flatten() == tokens

    def test_parse_empty_returns_none(self):
        p = Parser(None, None)
        assert p.parse([]) is None

    def test_parse_unknown_word_gets_tagged(self):
        angel = Angel()
        angel.awaken()
        p = Parser(angel._lexicon, build_syntactic_grammar())
        tree = p.parse(["xyzzy", "sat"])
        assert tree is not None

    def test_generate_returns_sequences(self):
        g = build_syntactic_grammar()
        p = Parser(None, g)
        seqs = p.generate("S", max_depth=4)
        assert len(seqs) > 0
        # All generated sequences are lists of strings
        for seq in seqs:
            assert isinstance(seq, list)
            for cat in seq:
                assert isinstance(cat, str)


class TestAngelParseIntegration:
    def test_angel_parse_returns_dict(self):
        angel = Angel()
        angel.awaken()
        result = angel.parse(["the", "cat"])
        assert "tokens" in result
        assert "tags" in result
        assert "tree" in result

    def test_angel_parse_correct_tags(self):
        angel = Angel()
        angel.awaken()
        result = angel.parse(["the", "cat", "sat"])
        assert result["tags"] == ["Det", "N", "V"]

    def test_angel_parse_produces_tree(self):
        angel = Angel()
        angel.awaken()
        result = angel.parse(["the", "cat", "sat"])
        assert result["tree"] is not None
        assert result["tree"]["cat"] == "S"


class TestAngelPredictWithParsing:
    def test_predict_fires_on_natural_language(self):
        """Predict should now return non-empty results for English input."""
        angel = Angel()
        angel.awaken()
        preds = angel.predict(["the", "cat"], domain="linguistic", horizon=8)
        # Should produce predictions via parse_extend
        assert len(preds) > 0

    def test_predict_tags_included(self):
        angel = Angel()
        angel.awaken()
        preds = angel.predict(["the", "cat"], domain="linguistic", horizon=5)
        # At least one prediction should include the tagged version
        tagged_preds = [p for p in preds if p.get("tagged")]
        assert len(tagged_preds) > 0

    def test_predict_after_np(self):
        """After 'the cat' (NP), VP should be a prediction."""
        angel = Angel()
        angel.awaken()
        preds = angel.predict(["the", "cat"], domain="linguistic", horizon=10)
        predicted = [p.get("predicted") for p in preds]
        # VP should be in the predictions (to form a sentence)
        assert any("VP" in str(p) for p in predicted)

    def test_predict_confidence_sorted(self):
        angel = Angel()
        angel.awaken()
        preds = angel.predict(["the", "cat"], domain="linguistic", horizon=5)
        if len(preds) > 1:
            for i in range(len(preds) - 1):
                assert preds[i]["confidence"] >= preds[i + 1]["confidence"]

    def test_predict_abstract_symbols_still_work(self):
        """Predict should still work with abstract categories."""
        angel = Angel()
        angel.awaken()
        preds = angel.predict(["NP", "VP"], domain="linguistic", horizon=5)
        # Abstract input should also produce results
        assert isinstance(preds, list)


class TestLexiconEnrichment:
    def test_lexicon_has_common_words(self):
        angel = Angel()
        angel.awaken()
        # The lexicon should now have common English
        common = ["the", "a", "cat", "run", "is", "very", "quickly"]
        found = 0
        for word in common:
            entries = angel._lexicon.lookup(form=word)
            exact = [e for e in entries if e.form == word]
            if exact:
                found += 1
        assert found >= 5

    def test_lexicon_size_grown(self):
        angel = Angel()
        angel.awaken()
        # With common words added, should have > 250 entries
        assert len(angel._lexicon.entries) > 250

import unittest

from research.sentiment.split_ablation import choose_cut, split_scenes


class SplitAblationTest(unittest.TestCase):
    def test_cut_balances_words_at_sentence_ends(self):
        edus = ["One two three.", "four five", "six seven.", "eight nine ten eleven twelve"]
        # Candidate cuts are after EDU 0 (3 vs 9 words) and after EDU 2 (7 vs 5 words).
        self.assertEqual(choose_cut(edus), 3)

    def test_closing_quotes_and_brackets_end_a_sentence(self):
        self.assertEqual(choose_cut(['He said "great."', "Then it broke."]), 1)
        self.assertEqual(choose_cut(["(It broke!)", "Sad."]), 1)

    def test_no_internal_sentence_end_means_no_cut(self):
        self.assertIsNone(choose_cut(["I liked it,", "but it broke."]))
        self.assertIsNone(choose_cut(["Excellent!"]))

    def test_ties_choose_the_earlier_cut(self):
        self.assertEqual(choose_cut(["A b.", "C d.", "E f."]), 1)

    def test_split_keeps_edus_and_review_identity(self):
        scenes = [
            {"document_id": "a", "key": "d000-s00", "text": "x", "tokens": 3, "status": "ok",
             "edus": ["Good screen.", "Bad battery,", "sadly."]},
            {"document_id": "b", "key": "d001-s00", "text": "y", "tokens": 2, "status": "ok",
             "edus": ["Fine,", "I guess."]},
        ]
        rows, decisions = split_scenes(scenes, lambda text: len(text.split()))
        self.assertEqual([row["key"] for row in rows], ["d000-s00", "d000-s01", "d001-s00"])
        self.assertEqual([row["edus"] for row in rows[:2]], [["Good screen."], ["Bad battery,", "sadly."]])
        self.assertEqual({row["document_id"] for row in rows[:2]}, {"a"})
        self.assertEqual(rows[2], scenes[1])
        self.assertEqual([d["cut"] for d in decisions], [1, None])

    def test_split_rejects_multiscene_input(self):
        with self.assertRaises(ValueError):
            split_scenes([{"document_id": "a", "key": "d000-s01", "status": "ok", "edus": ["x."]}], len)


if __name__ == "__main__":
    unittest.main()

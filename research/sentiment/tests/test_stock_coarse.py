import unittest

from research.sentiment.compare_stock_coarse import compare_trees


def tree(inner, root, relation, nuclearity, ids=(1, 2, 3, 4, 5)):
    first, second, third, group, whole = ids
    return {"key": "scene", "document_id": "review", "status": "ok", "root": whole, "nodes": {
        str(first): {"text": "A"}, str(second): {"text": "B"}, str(third): {"text": "C"},
        str(group): {"left": ids[inner[0]], "right": ids[inner[1]],
                     "nuclearity": nuclearity, "relation": relation},
        str(whole): {"left": ids[root[0]], "right": ids[root[1]],
                     "nuclearity": "NS", "relation": relation},
    }}


class StockCoarseTest(unittest.TestCase):
    scene = {"key": "scene", "document_id": "review", "edus": ["A", "B", "C"]}

    def test_uses_spans_and_mapping_instead_of_node_ids_or_fine_names(self):
        stock = tree((0, 1), (3, 2), "elaboration", "SN")
        fine = tree((0, 1), (3, 2), "extra-detail", "SN", (10, 11, 12, 13, 14))
        result = compare_trees([stock], [fine], [self.scene], {"extra-detail": "elaboration"})
        self.assertEqual(result["counts"]["same_split_nuclearity_and_label"], 2)
        self.assertEqual(result["counts"]["identical_scenes"], 1)

    def test_distinguishes_nuclearity_and_root_split_changes(self):
        stock = tree((0, 1), (3, 2), "detail", "SN")
        fine = tree((0, 1), (3, 2), "detail", "NS")
        mapping = {"detail": "detail"}
        counts = compare_trees([stock], [fine], [self.scene], mapping)["counts"]
        self.assertEqual((counts["same_shape_scenes"], counts["same_shape_and_nuclearity_scenes"]), (1, 0))

        fine = tree((1, 2), (0, 3), "detail", "NS")
        result = compare_trees([stock], [fine], [self.scene], mapping)
        self.assertEqual(result["counts"]["same_split"], 0)
        self.assertEqual(result["example"]["stock_root_split_after_edu"], 2)
        self.assertEqual(result["example"]["mapped_root_split_after_edu"], 1)


if __name__ == "__main__":
    unittest.main()

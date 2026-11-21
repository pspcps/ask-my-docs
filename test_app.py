"""Workspace regression tests without an Ollama server."""

import unittest
from unittest.mock import patch

import numpy as np

from workspace import Passage, demo_answer, demo_passages, embed, passages_from_file


class WorkspaceTests(unittest.TestCase):
    def test_passages_overlap_without_repeating_the_final_window(self):
        words = [f"word{index}" for index in range(300)]
        passages = passages_from_file("long.TXT", " ".join(words).encode())
        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0].text.split()[-35:], passages[1].text.split()[:35])
        self.assertEqual(passages[1].text.split()[-1], words[-1])

    def test_csv_locations_and_empty_input(self):
        passage, = passages_from_file("data.csv", b"Name,Role\nSam,Editor\n")
        self.assertEqual(passage, Passage("data.csv", "Row 2", "Name: Sam | Role: Editor"))
        with self.assertRaisesRegex(ValueError, "No readable text"):
            passages_from_file("empty.md", b"\n")

    def test_demo_citation_and_missing_evidence(self):
        passages = demo_passages()
        answer, sources = demo_answer("What is the refund window?", passages)
        self.assertIn("30-day", answer)
        self.assertEqual(sources[0].location, "Refund Process")
        self.assertEqual(demo_answer("neutrino", passages)[1], [])

    @patch("workspace.ollama_request")
    def test_embedding_batch_shape_and_normalization(self, request):
        request.side_effect = lambda path, payload: {"embeddings": [[3, 4]] * len(payload["input"])}
        vectors = embed(["text"] * 33)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(vectors.shape, (33, 2))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)

if __name__ == "__main__":
    unittest.main()

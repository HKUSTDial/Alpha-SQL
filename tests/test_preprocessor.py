import importlib
import os
import unittest


class FakeEmbeddingModel:
    def __init__(self):
        self.embedded_texts = None

    def embed_documents(self, texts):
        self.embedded_texts = texts
        return [[1.0, 0.0] for _ in texts]


class PreprocessorTest(unittest.TestCase):
    def test_embedding_similarity_filters_empty_text_before_embedding(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        preprocessor_module = importlib.import_module("alphasql.runner.preprocessor")
        fake_embedding_model = FakeEmbeddingModel()
        preprocessor_module.EMBEDDING_MODEL_CALLABLE = fake_embedding_model

        candidate_values = [
            {
                "query": "",
                "table_name": "items",
                "column_name": "name",
                "value": "short",
                "edit_similarity": 1.0,
            },
            {
                "query": "short",
                "table_name": "items",
                "column_name": "name",
                "value": "",
                "edit_similarity": 1.0,
            },
            {
                "query": "short",
                "table_name": "items",
                "column_name": "name",
                "value": "short",
                "edit_similarity": 1.0,
            },
        ]

        filtered_values = (
            preprocessor_module.Preprocessor.filter_candidate_values_by_embedding_similarity(
                None,
                candidate_values,
                embedding_similarity_threshold=0.5,
            )
        )

        self.assertEqual(fake_embedding_model.embedded_texts, ["short"])
        self.assertEqual(len(filtered_values), 1)
        self.assertEqual(filtered_values[0]["value"], "short")


if __name__ == "__main__":
    unittest.main()

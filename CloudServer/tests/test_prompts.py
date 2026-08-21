import unittest

from app.prompts import interpreter_instruction, translation_messages


class PromptTests(unittest.TestCase):
    def test_korean_requires_honorifics_and_hangul_names(self):
        prompt = interpreter_instruction("en", "ko")
        self.assertIn("honorific", prompt)
        self.assertIn("Hangul", prompt)

    def test_context_is_bounded_to_three_turns(self):
        context = tuple((f"source {i}", f"target {i}") for i in range(5))
        messages = translation_messages("now", "en", "ko", context)
        self.assertNotIn("source 0", str(messages))
        self.assertIn("source 4", str(messages))
        self.assertEqual(messages[-1]["content"], "now\n/no_think")

    def test_rejects_other_language_pairs(self):
        with self.assertRaises(ValueError):
            interpreter_instruction("ja", "ko")


if __name__ == "__main__":
    unittest.main()


import unittest
import random
from painter.prompt import Prompt, PromptGrid

class TestPromptGrid(unittest.TestCase):
    def setUp(self):
        self.width = 256
        self.height = 256
        self.grid = PromptGrid(self.width, self.height)

    def test_add_single_prompt(self):
        self.grid.add("cat", "ugly", seed=42)
        self.assertEqual(len(self.grid), 1)
        prompt = self.grid.prompts[0]
        self.assertEqual(prompt.positive, "cat")
        self.assertEqual(prompt.negative, "ugly")
        self.assertEqual(prompt.seed, 42)

    def test_add_multiple_positives_pairwise(self):
        self.grid.add(["cat", "dog"], ["ugly", "scary"], seed=[1, 2])
        self.assertEqual(len(self.grid), 2)
        self.assertEqual(self.grid.prompts[1].positive, "dog")
        self.assertEqual(self.grid.prompts[1].seed, 2)

    def test_add_seed_broadcasting(self):
        self.grid.add(["cat", "dog"], "ugly", seed=99)
        self.assertEqual(len(self.grid), 2)
        self.assertEqual(self.grid.prompts[0].seed, 99)
        self.assertEqual(self.grid.prompts[1].seed, 99)

    def test_add_variations(self):
        # Replaces the old cartesian methods
        pos = ["cat", "dog"]
        neg = ["ugly"]
        seeds = [42, 43]
        cfgs = [7.5]
        
        self.grid.add_variations(pos, neg, seeds, cfgs=cfgs)
        # 2 pos * 1 neg * 2 seeds * 1 cfg * 1 step = 4 prompts
        self.assertEqual(len(self.grid), 4)
        self.assertEqual(self.grid.prompts[0].cfg, 7.5)
        
    def test_construct_prompt_dimensions(self):
        # Updated from 'construct_prompt_variations'
        variations = [
            ["cat", "dog"],
            ["cute", "fluffy"]
        ]
        results = self.grid.construct_prompt_dimensions(variations)
        expected = ["cat, cute", "cat, fluffy", "dog, cute", "dog, fluffy"]
        self.assertEqual(results, expected)

    def test_dropout_logic(self):
        # Test 100% dropout to ensure it removes parts
        dropout_grid = PromptGrid(256, 256, pos_dropout=1.0)
        # Input has 2 parts separated by comma
        dropout_grid.add("highly detailed, masterpiece", "bad quality", seed=1)
        
        # Result should be empty string (100% of 2 parts dropped)
        self.assertEqual(dropout_grid.prompts[0].positive, "")

    def test_prompt_name_generation(self):
        p = Prompt("A cat, sitting.", None, 256, 256)
        # Should strip punctuation and spaces
        self.assertEqual(p.name, "IMAGE_Acatsitting")

if __name__ == "__main__":
    unittest.main()
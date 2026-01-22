import unittest
from unittest.mock import patch, MagicMock
from painter.summarizer import LocalSummarizer

class TestLocalSummarizer(unittest.TestCase):

    @patch('ollama.list')
    def setUp(self, mock_list):
        # Mocking ollama.list to simulate that llama3 is already installed
        mock_list.return_value = {
            'models': [{'name': 'llama3:latest'}]
        }
        self.summarizer = LocalSummarizer(model_name='llama3')

    @patch('deep_translator.GoogleTranslator.translate')
    def test_translate_method(self, mock_translate):
        mock_translate.return_value = "Hello"
        result = self.summarizer.translate("Hallo", source="de")
        
        self.assertEqual(result, "Hello")
        mock_translate.assert_called_once_with("Hallo")

    @patch('ollama.generate')
    @patch.object(LocalSummarizer, 'translate')
    def test_summarize_with_translation(self, mock_translate, mock_generate):
        # Setup mocks
        mock_translate.return_value = "The tax reform is good."
        mock_generate.return_value = {'response': 'Summary for business.'}
        
        text = "Die Steuerreform ist gut."
        result = self.summarizer.summarize(text, perspective="business owner", lang="de")
        
        # Verify translation was called because lang="de"
        mock_translate.assert_called_once_with(text, "de")
        # Verify result
        self.assertEqual(result, 'Summary for business.')

    @patch('ollama.generate')
    def test_summarize_english_skips_translation(self, mock_generate):
        mock_generate.return_value = {'response': 'Direct summary.'}
        
        with patch.object(LocalSummarizer, 'translate') as mock_translate:
            self.summarizer.summarize("English text", lang="en")
            # Translation should NOT be called for English
            mock_translate.assert_not_called()

    @patch('ollama.list')
    def test_init_raises_error_if_model_missing(self, mock_list):
        # Simulate an empty model list
        mock_list.return_value = {'models': []}
        
        with self.assertRaises(ValueError):
            LocalSummarizer(model_name='non-existent-model')

if __name__ == '__main__':
    unittest.main()
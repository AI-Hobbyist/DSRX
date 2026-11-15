import json
import random
import tempfile
import unittest
from pathlib import Path

from utils.variance_validation import iter_validation_sources


class ValidationSourceIterationTest(unittest.TestCase):
    def test_each_valid_segment_is_tried_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dictionary = root / 'zh.txt'
            dictionary.write_text('a\ta\n', encoding='utf-8')
            source = root / 'samples.ds'
            source.write_text(json.dumps([
                {'text': 'a', 'note_seq': 'C4', 'note_dur': '1', 'id': index}
                for index in range(3)
            ]), encoding='utf-8')
            items = list(iter_validation_sources(
                {'enable': True, 'zh': [str(source)]},
                {'zh': str(dictionary)},
                rng=random.Random(42),
            ))
        self.assertEqual(sorted(item['id'] for _, item in items), [0, 1, 2])


if __name__ == '__main__':
    unittest.main()

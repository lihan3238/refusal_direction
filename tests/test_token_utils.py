from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.utils.token_utils import decode_token_labels


class FakeTokenizer:
    def __init__(self):
        self.seen_token_ids = []

    def decode(self, token_ids):
        self.seen_token_ids.append(token_ids)
        return f"token-{token_ids[0]}"


def test_decode_token_labels_decodes_each_token_id_individually():
    tokenizer = FakeTokenizer()

    labels = decode_token_labels(tokenizer, [107, 108, 106, 2516, 108])

    assert labels == ["token-107", "token-108", "token-106", "token-2516", "token-108"]
    assert tokenizer.seen_token_ids == [[107], [108], [106], [2516], [108]]

"""Data acquisition integrity checks; all fixtures are synthetic and offline."""

from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from utils.data import load_reviews, obtain_reviews, summarize_reviews
from utils.reproducibility import load_config


def archive_bytes(records):
    text = "\n".join(json.dumps(row) for row in records) + "\n"
    return gzip.compress(text.encode(), mtime=0)


class Response(BytesIO):
    status = 200
    url = "https://example.test/reviews.json.gz"

    def __init__(self, payload, content_type="application/octet-stream", length=None):
        super().__init__(payload)
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(payload) if length is None else length),
        }


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        env = patch.dict(os.environ, {"RECSYS_DATA_DIR": self.directory.name})
        env.start()
        self.addCleanup(env.stop)
        self.config = deepcopy(load_config())
        self.config["dataset"]["sha256"] = None
        self.config["dataset"]["download_url"] = Response.url
        self.records = [
            {"reviewerID": "synthetic-user-1", "asin": "synthetic-item-1", "overall": 5, "unixReviewTime": 1000},
            {"reviewerID": "synthetic-user-2", "asin": "synthetic-item-2", "overall": 2, "unixReviewTime": 2000},
        ]
        self.payload = archive_bytes(self.records)
        self.path = Path(self.directory.name) / self.config["dataset"]["filename"]

    def assert_no_download_leftovers(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(list(Path(self.directory.name).glob("*.part")), [])

    def test_full_download_and_network_free_reuse(self):
        with patch("utils.data.urlopen", return_value=Response(self.payload)) as opener:
            frame, manifest = obtain_reviews(self.config)
            opener.assert_called_once()
        self.assertEqual(manifest["sha256"], hashlib.sha256(self.payload).hexdigest())
        self.assertEqual(manifest["size_bytes"], len(self.payload))
        self.assertEqual(frame["source_line"].tolist(), [1, 2])
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.config["dataset"]["sha256"] = manifest["sha256"]
        with patch("utils.data.urlopen", side_effect=AssertionError("Network used on reuse")):
            cached, cached_manifest = obtain_reviews(self.config)
        self.assertTrue(frame.equals(cached))
        self.assertEqual(manifest["downloaded_at_utc"], cached_manifest["downloaded_at_utc"])

    def test_html_http_200_is_rejected(self):
        with patch("utils.data.urlopen", return_value=Response(b"<html>login</html>", "text/html")):
            with self.assertRaisesRegex(ValueError, "HTML"):
                obtain_reviews(self.config)
        self.assert_no_download_leftovers()

    def test_html_with_binary_content_type_is_rejected(self):
        with patch("utils.data.urlopen", return_value=Response(b"<html>login</html>")):
            with self.assertRaisesRegex(ValueError, "gzip"):
                obtain_reviews(self.config)
        self.assert_no_download_leftovers()

    def test_checksum_mismatch_does_not_publish_file(self):
        self.config["dataset"]["sha256"] = "0" * 64
        with patch("utils.data.urlopen", return_value=Response(self.payload)):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                obtain_reviews(self.config)
        self.assert_no_download_leftovers()

    def test_short_response_does_not_publish_file(self):
        with patch("utils.data.urlopen", return_value=Response(self.payload, length=len(self.payload) + 10)):
            with self.assertRaisesRegex(ValueError, "Content-Length"):
                obtain_reviews(self.config)
        self.assert_no_download_leftovers()

    def test_size_limit_does_not_publish_file(self):
        self.config["dataset"]["download"]["max_bytes"] = 1
        with patch("utils.data.urlopen", return_value=Response(self.payload)):
            with self.assertRaisesRegex(ValueError, "limit"):
                obtain_reviews(self.config)
        self.assert_no_download_leftovers()

    def test_http_404_fails_without_retry(self):
        error = HTTPError(Response.url, 404, "Not found", {}, None)
        with patch("utils.data.urlopen", side_effect=error) as opener:
            with self.assertRaisesRegex(RuntimeError, "404"):
                obtain_reviews(self.config)
            opener.assert_called_once()
        self.assert_no_download_leftovers()

    def test_http_503_is_retried(self):
        error = HTTPError(Response.url, 503, "Unavailable", {}, None)
        with patch("utils.data.urlopen", side_effect=[error, Response(self.payload)]) as opener:
            with patch("utils.data.time.sleep") as sleep:
                frame, _ = obtain_reviews(self.config)
        self.assertEqual(opener.call_count, 2)
        sleep.assert_called_once_with(1)
        self.assertEqual(len(frame), 2)

    def test_existing_changed_archive_is_not_overwritten(self):
        with patch("utils.data.urlopen", return_value=Response(self.payload)):
            obtain_reviews(self.config)
        altered = archive_bytes(self.records[:1])
        self.path.write_bytes(altered)
        with patch("utils.data.urlopen", side_effect=AssertionError("Unexpected download")):
            with self.assertRaisesRegex(ValueError, "acquisition manifest"):
                obtain_reviews(self.config)
        self.assertEqual(self.path.read_bytes(), altered)

    def test_malformed_json_reports_source_line(self):
        self.path.write_bytes(gzip.compress(b'{"valid": true}\nnot-json\n'))
        with self.assertRaisesRegex(ValueError, "source line 2"):
            load_reviews(self.path)

    def test_truncated_gzip_is_rejected(self):
        self.path.write_bytes(self.payload[:-5])
        with self.assertRaisesRegex(ValueError, "Corrupt gzip"):
            load_reviews(self.path)

    def test_missing_required_columns_are_rejected(self):
        self.path.write_bytes(archive_bytes([{"reviewerID": "synthetic-user"}]))
        frame = load_reviews(self.path)
        with self.assertRaisesRegex(ValueError, "Required source columns"):
            summarize_reviews(frame, self.config)

    def test_loader_keeps_duplicates_and_missing_values_for_cleaning(self):
        missing = dict(self.records[0])
        del missing["overall"]
        self.path.write_bytes(archive_bytes([self.records[0], self.records[0], missing]))
        frame = load_reviews(self.path)
        summary = summarize_reviews(frame, self.config)
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame["source_line"].tolist(), [1, 2, 3])
        self.assertEqual(summary["required_fields"]["rating"]["missing_rows"], 1)
        self.assertFalse(summary["cleaned_constraints_verified"])


if __name__ == "__main__":
    unittest.main()

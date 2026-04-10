"""Tests for filesystem and parquet helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from common.io_utils import prepare_dataframe_for_parquet


class IoUtilsTests(unittest.TestCase):
    """Check parquet preparation for blockchain-sized integers."""

    def test_large_python_int_object_column_is_cast_to_string(self) -> None:
        frame = pd.DataFrame(
            {
                "raw_bigint": [2**130, -(2**129)],
                "small_int": [1, 2],
                "label": ["a", "b"],
            }
        ).astype({"raw_bigint": "object"})

        prepared = prepare_dataframe_for_parquet(frame)
        self.assertTrue(pd.api.types.is_string_dtype(prepared["raw_bigint"]))
        self.assertEqual(prepared["raw_bigint"].tolist(), [str(2**130), str(-(2**129))])
        self.assertEqual(prepared["small_int"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()

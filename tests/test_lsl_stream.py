import os
from pathlib import Path
import unittest

import mne_lsl


@unittest.skipUnless(
    os.environ.get("SSPYTHON_RUN_LSL_INTEGRATION") == "1",
    "requires an external FIF recording",
)
class TestLSLStreamIntegration(unittest.TestCase):
    def test_starts_player_for_configured_recording(self):
        raw_path = Path(os.environ["SSPYTHON_LSL_TEST_FILE"])
        self.assertTrue(raw_path.is_file(), raw_path)

        player = mne_lsl.player.PlayerLSL(
            raw_path,
            chunk_size=200,
            name="SSPy-Player",
        ).start()
        try:
            self.assertEqual(player.name, "SSPy-Player")
        finally:
            player.stop()

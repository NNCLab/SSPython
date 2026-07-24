import os
import unittest

from mne_lsl.stream import StreamLSL


@unittest.skipUnless(
    os.environ.get("SSPYTHON_RUN_LSL_INTEGRATION") == "1",
    "requires a running LSL stream",
)
class TestLSLReadIntegration(unittest.TestCase):
    def test_connects_to_configured_stream(self):
        stream_name = os.environ.get("SSPYTHON_LSL_STREAM_NAME", "SSPy-Player")
        stream = StreamLSL(bufsize=10, name=stream_name)
        stream.connect(acquisition_delay=0.1, processing_flags="all")
        try:
            self.assertGreater(stream.info["sfreq"], 0)
        finally:
            stream.disconnect()

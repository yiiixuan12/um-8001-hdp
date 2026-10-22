import json
from pathlib import Path
import tempfile
import unittest
from analyse_grasp_pose import run


class AnalysisGuard(unittest.TestCase):
    def test_partial_collection_not_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"partial.json"
            path.write_text(json.dumps({"protocol":{"scenes":20},"collection_complete":False,"records":[]}))
            with self.assertRaisesRegex(ValueError,"Incomplete"):
                run(path)


if __name__=="__main__":unittest.main()

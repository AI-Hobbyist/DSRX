import pathlib
import subprocess
import sys
import textwrap
import unittest


class DependencyImportTests(unittest.TestCase):
    def test_main_workflow_imports_without_pkg_resources(self):
        repository = pathlib.Path(__file__).resolve().parents[1]
        code = textwrap.dedent(f"""
            import importlib.abc
            import sys
            sys.path.insert(0, {str(repository)!r})
            assert 'pkg_resources' not in sys.modules
            class RejectLegacy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'pkg_resources' or fullname.startswith('pkg_resources.'):
                        raise AssertionError(fullname)
            sys.meta_path.insert(0, RejectLegacy())
            import librosa
            import pyworld
            import lightning.pytorch
            import preprocessing.all_in_one_binarizer
            import training.all_in_one_task
            import utils.training_utils
            assert 'pkg_resources' not in sys.modules
        """)
        subprocess.run([sys.executable, "-I", "-B", "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()

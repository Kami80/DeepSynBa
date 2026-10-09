"""Check integrity protection and preservation of local work during setup."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import setup_reference as setup


class ReferenceSetupTests(unittest.TestCase):
    def test_same_size_tampering_is_rejected_before_install(self):
        original = b"original"
        manifest = {"dataset.py": (setup.digest(original), len(original))}
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("reference/dataset.py", b"modified")
            with patch.object(setup, "REFERENCE_FILES", manifest):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    setup.load_reference(archive)

    def test_conflicting_archive_members_are_rejected(self):
        content = b"valid"
        manifest = {"dataset.py": (setup.digest(content), len(content))}
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("first/dataset.py", content)
                bundle.writestr("second/dataset.py", content)
            with patch.object(setup, "REFERENCE_FILES", manifest):
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    setup.load_reference(archive)

    def test_conflict_preserves_edits_and_prevents_partial_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            edited = output / "dataset.py"
            edited.write_bytes(b"my local edits\n")
            with self.assertRaises(FileExistsError):
                setup.install_reference({"config.yaml": b"config\n", "dataset.py": b"reference\n"}, output)
            self.assertEqual(edited.read_bytes(), b"my local edits\n")
            self.assertFalse((output / "config.yaml").exists())
            self.assertFalse((output / ".reference-cache").exists())

    def test_matching_windows_file_keeps_its_existing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            target = output / "dataset.py"
            target.write_bytes(b"reference\r\n")
            setup.install_reference({"dataset.py": b"reference\n"}, output)
            setup.install_reference({"dataset.py": b"reference\n"}, output)
            self.assertEqual(target.read_bytes(), b"reference\r\n")


if __name__ == "__main__":
    unittest.main()

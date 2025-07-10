import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.manifest_builder import ManifestBuilder

if __name__ == "__main__":
    # test sample path
    root_path = "test/dummy_samples"
    # Create an instance of ManifestBuilder
    manifest_builder = ManifestBuilder(root_path)

    # Build the manifest
    manifest_builder.build()
    # Save manifest
    output_path = "test/outputs/dummy_manifest.json"
    manifest_builder.save(output_path)
    print("Manifest file created")

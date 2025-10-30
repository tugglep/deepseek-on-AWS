#!/usr/bin/env python3
"""Create source zip for CodeBuild without requiring zip command"""

import os
import zipfile
from pathlib import Path

def create_zip(source_dir, output_file, exclude_patterns):
    """Create zip file from source directory"""
    source_path = Path(source_dir)

    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # First, add buildspec.yml at the root if it exists
        buildspec_path = source_path / "buildspec.yml"
        if buildspec_path.exists():
            print(f"Adding: buildspec.yml (at root)")
            zipf.write(buildspec_path, "buildspec.yml")

        for root, dirs, files in os.walk(source_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files:
                file_path = Path(root) / file
                # Skip excluded patterns
                if any(pattern in str(file_path) for pattern in exclude_patterns):
                    continue

                # Skip buildspec.yml since we already added it at root
                if file == "buildspec.yml":
                    continue

                arcname = file_path.relative_to(source_path)
                print(f"Adding: {arcname}")
                zipf.write(file_path, arcname)

    print(f"\n✅ Created {output_file} ({os.path.getsize(output_file) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    exclude = ['.ipynb_checkpoints', '__pycache__', '.git', '.DS_Store']
    create_zip("DeepSeek OCR", "deepseek-ocr-source.zip", exclude)

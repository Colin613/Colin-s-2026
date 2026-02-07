#!/usr/bin/env python3
"""Regenerate protobuf files for compatibility with protobuf 5.x+"""
import subprocess
import sys
from pathlib import Path

# Change to fish-speech directory
fish_speech_dir = Path(__file__).parent
print(f"Working directory: {fish_speech_dir}")

# Run protoc from grpcio-tools
proto_file = fish_speech_dir / "fish_speech" / "datasets" / "protos" / "text-data.proto"
output_dir = fish_speech_dir / "fish_speech" / "datasets" / "protos"

cmd = [
    sys.executable, "-m", "grpc_tools.protoc",
    "--python_out", str(output_dir),
    "--proto_path", str(output_dir),
    str(proto_file)
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True, cwd=fish_speech_dir)

print("STDOUT:", result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    print(f"Failed with return code {result.returncode}")
    sys.exit(1)

print("Proto file regenerated successfully!")

"""Prepare/train one source-only NSF-PINN frozen adaptation; no experiment starts on import."""

from audio_deepfake_detection.sota.frozen_training import main

if __name__ == "__main__":
    main("nsf_pinn")

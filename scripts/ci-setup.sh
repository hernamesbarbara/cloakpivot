#!/bin/bash
set -e

echo "Setting up CI environment..."

# Download required spaCy models based on MODEL_SIZE
case "${MODEL_SIZE:-small}" in
  "small")
    echo "Downloading small spaCy model..."
    python -m spacy download en_core_web_sm
    ;;
  "medium") 
    echo "Downloading small and medium spaCy models..."
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_md
    ;;
  "large")
    echo "Downloading all spaCy models..."
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_md
    python -m spacy download en_core_web_lg
    ;;
  *)
    echo "Unknown MODEL_SIZE: ${MODEL_SIZE}, defaulting to small"
    python -m spacy download en_core_web_sm
    ;;
esac

echo "Verifying model installation..."
python -c "
import spacy
try:
    nlp = spacy.load('en_core_web_sm')
    print('✓ en_core_web_sm loaded successfully')
except OSError:
    print('✗ en_core_web_sm failed to load')
    exit(1)
"

echo "Models downloaded and verified successfully"
#!/usr/bin/env python3
"""
Simple Presidio + Faker test script.

Reads a text file, detects PII (PERSON, PHONE_NUMBER, US_SSN),
and substitutes fake values using Faker.

Usage:
    python presidio_faker_substitute.py input.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from faker import Faker
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


def build_operator_config(fake: Faker) -> Dict[str, OperatorConfig]:
    """Create Faker-backed replacement operators for Presidio."""
    return {
        "PERSON": OperatorConfig(
            operator_name="replace",
            params={"new_value": fake.name()},
        ),
        "PHONE_NUMBER": OperatorConfig(
            operator_name="replace",
            params={"new_value": fake.phone_number()},
        ),
        "US_SSN": OperatorConfig(
            operator_name="replace",
            params={"new_value": fake.ssn()},
        ),
    }


def anonymize_text(text: str) -> str:
    """Detect and replace PII in text using Presidio + Faker."""
    fake = Faker()
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(
        text=text,
        language="en",
        entities=["PERSON", "PHONE_NUMBER", "US_SSN"],
    )

    operators = build_operator_config(fake)

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )

    return anonymized.text


def main() -> None:
    """Entry point."""
    if len(sys.argv) != 2:
        print("Usage: python presidio_faker_substitute.py input.txt")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text()
    output = anonymize_text(text)

    print(output)


if __name__ == "__main__":
    main()

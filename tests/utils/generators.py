"""Test data generators for comprehensive CloakPivot testing."""

import random
import string
from typing import Dict, List, Optional, Tuple

from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem
from presidio_analyzer import RecognizerResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind


class DocumentGenerator:
    """Generates various types of test documents."""
    
    @staticmethod
    def generate_simple_document(
        text: str,
        name: Optional[str] = None
    ) -> DoclingDocument:
        """Generate a simple document with single text item."""
        doc_name = name or f"test_doc_{random.randint(1000, 9999)}"
        doc = DoclingDocument(name=doc_name)
        
        text_item = TextItem(
            text=text,
            self_ref="#/texts/0",
            label="text",
            orig=text
        )
        doc.texts = [text_item]
        return doc
    
    @staticmethod
    def generate_multi_section_document(
        sections: List[str],
        name: Optional[str] = None
    ) -> DoclingDocument:
        """Generate document with multiple text sections."""
        doc_name = name or f"multi_doc_{random.randint(1000, 9999)}"
        doc = DoclingDocument(name=doc_name)
        
        text_items = []
        for i, section in enumerate(sections):
            text_item = TextItem(
                text=section,
                self_ref=f"#/texts/{i}",
                label="text",
                orig=section
            )
            text_items.append(text_item)
        
        doc.texts = text_items
        return doc
    
    @staticmethod
    def generate_document_with_pii(
        pii_types: List[str],
        name: Optional[str] = None
    ) -> Tuple[DoclingDocument, Dict[str, List[str]]]:
        """Generate document containing specified PII types and return locations."""
        pii_patterns = {
            "PHONE_NUMBER": ["555-123-4567", "(555) 987-6543", "555.234.5678"],
            "EMAIL_ADDRESS": ["john@example.com", "alice.smith@company.org", "user123@test.net"],
            "US_SSN": ["123-45-6789", "987-65-4321", "555-44-3333"],
            "CREDIT_CARD": ["4532-1234-5678-9012", "5555-4444-3333-2222", "378282246310005"],
            "PERSON": ["John Smith", "Alice Johnson", "Bob Wilson", "Mary Davis"],
            "LOCATION": ["New York", "123 Main Street", "San Francisco, CA"],
            "US_DRIVER_LICENSE": ["DL123456789", "D123-456-789-012", "AB1234567"],
        }
        
        text_parts = ["This document contains the following information:"]
        pii_locations = {}
        
        for pii_type in pii_types:
            if pii_type in pii_patterns:
                values = random.choices(pii_patterns[pii_type], k=random.randint(1, 3))
                pii_locations[pii_type] = values
                
                for value in values:
                    text_parts.append(f"The {pii_type.lower().replace('_', ' ')} is {value}.")
        
        # Add some non-PII text
        text_parts.extend([
            "This is additional context text.",
            "Processing should handle this document correctly.",
            "End of document content."
        ])
        
        full_text = " ".join(text_parts)
        document = DocumentGenerator.generate_simple_document(full_text, name)
        
        return document, pii_locations


class PolicyGenerator:
    """Generates various masking policies for testing."""
    
    @staticmethod
    def generate_basic_policy(
        privacy_level: str = "medium",
        locale: str = "en"
    ) -> MaskingPolicy:
        """Generate a basic masking policy."""
        return MaskingPolicy(
            locale=locale,
            per_entity={
                "PHONE_NUMBER": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "phone"}),
                "EMAIL_ADDRESS": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "email"}),
                "US_SSN": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "ssn"}),
                "CREDIT_CARD": Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "credit_card"}),
            },
            thresholds={
                "PHONE_NUMBER": 0.7,
                "EMAIL_ADDRESS": 0.8,
                "US_SSN": 0.9,
                "CREDIT_CARD": 0.8,
            }
        )
    
    @staticmethod
    def generate_comprehensive_policy(
        privacy_level: str = "high"
    ) -> MaskingPolicy:
        """Generate a comprehensive policy covering many entity types."""
        all_entities = [
            "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD",
            "PERSON", "LOCATION", "US_DRIVER_LICENSE", "DATE_TIME",
            "US_PASSPORT", "US_BANK_NUMBER", "MEDICAL_LICENSE"
        ]
        
        # Use reversible surrogate strategies for testing
        strategy = Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "custom"})
        
        per_entity = {entity_type: strategy for entity_type in all_entities}
        
        # Special strategies for specific types - use surrogate for reversible testing
        per_entity["PHONE_NUMBER"] = Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "phone"})
        per_entity["EMAIL_ADDRESS"] = Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "email"})
        per_entity["US_SSN"] = Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "ssn"})
        per_entity["CREDIT_CARD"] = Strategy(kind=StrategyKind.SURROGATE, parameters={"format_type": "credit_card"})
        
        # Lower thresholds for high privacy
        base_threshold = 0.5 if privacy_level == "high" else 0.7
        thresholds = {entity_type: base_threshold for entity_type in all_entities}
        
        # Higher thresholds for sensitive data
        sensitive_types = ["US_SSN", "CREDIT_CARD", "US_PASSPORT", "MEDICAL_LICENSE"]
        for entity_type in sensitive_types:
            thresholds[entity_type] = min(base_threshold + 0.2, 0.9)
        
        return MaskingPolicy(
            locale="en",
            per_entity=per_entity,
            thresholds=thresholds
        )
    
    @staticmethod
    def generate_custom_policy(
        entity_strategies: Dict[str, StrategyKind],
        thresholds: Optional[Dict[str, float]] = None
    ) -> MaskingPolicy:
        """Generate a custom policy with specified strategies."""
        per_entity = {}
        for entity_type, strategy_kind in entity_strategies.items():
            # Generate appropriate parameters for each strategy kind
            parameters = {}
            if strategy_kind == StrategyKind.TEMPLATE:
                parameters = {"template": f"[{entity_type}]"}
            elif strategy_kind == StrategyKind.PARTIAL:
                parameters = {"visible_chars": 4}
            elif strategy_kind == StrategyKind.HASH:
                parameters = {"algorithm": "sha256", "truncate": 8}
            elif strategy_kind == StrategyKind.SURROGATE:
                format_types = {
                    "PHONE_NUMBER": "phone",
                    "EMAIL_ADDRESS": "email", 
                    "US_SSN": "ssn",
                    "CREDIT_CARD": "credit_card",
                    "PERSON": "name"
                }
                parameters = {"format_type": format_types.get(entity_type, "custom")}
            
            per_entity[entity_type] = Strategy(kind=strategy_kind, parameters=parameters)
        
        default_thresholds = {entity_type: 0.7 for entity_type in entity_strategies.keys()}
        if thresholds:
            default_thresholds.update(thresholds)
        
        return MaskingPolicy(
            locale="en",
            per_entity=per_entity,
            thresholds=default_thresholds
        )


class EntityGenerator:
    """Generates RecognizerResult entities for testing."""
    
    @staticmethod
    def generate_entities_for_text(
        text: str,
        entity_types: List[str],
        confidence_range: Tuple[float, float] = (0.7, 0.95)
    ) -> List[RecognizerResult]:
        """Generate synthetic entities for given text."""
        entities = []
        
        # Simple pattern matching for common entity types
        patterns = {
            "PHONE_NUMBER": r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
            "EMAIL_ADDRESS": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "US_SSN": r'\b\d{3}-\d{2}-\d{4}\b',
            "CREDIT_CARD": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        }
        
        import re
        for entity_type in entity_types:
            if entity_type in patterns:
                pattern = patterns[entity_type]
                matches = re.finditer(pattern, text)
                
                for match in matches:
                    confidence = random.uniform(*confidence_range)
                    entity = RecognizerResult(
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        score=confidence
                    )
                    entities.append(entity)
        
        return entities
    
    @staticmethod
    def generate_overlapping_entities(
        text_length: int,
        num_entities: int = 5
    ) -> List[RecognizerResult]:
        """Generate overlapping entities for edge case testing."""
        entities = []
        entity_types = ["PERSON", "LOCATION", "ORGANIZATION", "PHONE_NUMBER", "EMAIL_ADDRESS"]
        
        for i in range(num_entities):
            # Create somewhat overlapping ranges
            start = random.randint(0, max(1, text_length - 20))
            length = random.randint(5, 15)
            end = min(start + length, text_length)
            
            entity = RecognizerResult(
                entity_type=random.choice(entity_types),
                start=start,
                end=end,
                score=random.uniform(0.6, 0.95)
            )
            entities.append(entity)
        
        return entities


class TextGenerator:
    """Generates various text content for testing."""
    
    @staticmethod
    def generate_text_with_pii_density(
        word_count: int,
        pii_density: float = 0.1,
        pii_types: Optional[List[str]] = None
    ) -> str:
        """Generate text with specified PII density."""
        if pii_types is None:
            pii_types = ["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "LOCATION"]
        
        pii_examples = {
            "PHONE_NUMBER": ["555-123-4567", "(555) 987-6543", "555.234.5678"],
            "EMAIL_ADDRESS": ["john@example.com", "alice@company.org", "user@test.net"],
            "PERSON": ["John Smith", "Alice Johnson", "Bob Wilson"],
            "LOCATION": ["New York", "Main Street", "California"],
        }
        
        # Generate base words
        base_words = [
            "the", "and", "is", "to", "of", "in", "it", "you", "that", "he",
            "was", "for", "on", "are", "as", "with", "his", "they", "at", "be",
            "this", "have", "from", "or", "one", "had", "but", "not", "what",
            "all", "were", "we", "when", "your", "can", "said", "there", "each"
        ]
        
        words = []
        pii_count = int(word_count * pii_density)
        regular_word_count = word_count - pii_count
        
        # Add regular words
        words.extend(random.choices(base_words, k=regular_word_count))
        
        # Add PII
        for _ in range(pii_count):
            pii_type = random.choice(pii_types)
            if pii_type in pii_examples:
                pii_value = random.choice(pii_examples[pii_type])
                words.append(pii_value)
        
        # Shuffle and join
        random.shuffle(words)
        return " ".join(words)
    
    @staticmethod
    def generate_structured_content() -> str:
        """Generate structured content like forms or reports."""
        template = """
Employee Information Form

Personal Details:
Name: {name}
Phone: {phone}
Email: {email}
SSN: {ssn}

Emergency Contact:
Name: {emergency_name}
Phone: {emergency_phone}
Relationship: {relationship}

Address:
Street: {street}
City: {city}
State: {state}
Zip: {zip_code}

Employment:
Start Date: {start_date}
Department: {department}
Employee ID: {employee_id}
"""
        
        return template.format(
            name="John Doe",
            phone="555-123-4567", 
            email="john.doe@company.com",
            ssn="123-45-6789",
            emergency_name="Jane Doe",
            emergency_phone="555-987-6543",
            relationship="Spouse",
            street="123 Main Street",
            city="New York",
            state="NY",
            zip_code="10001",
            start_date="2023-01-15",
            department="Engineering",
            employee_id="EMP001234"
        )


def generate_test_suite_data(num_documents: int = 10) -> List[Tuple[DoclingDocument, MaskingPolicy]]:
    """Generate a comprehensive test suite with varied documents and policies."""
    test_data = []
    
    for i in range(num_documents):
        # Vary the document types
        if i % 3 == 0:
            # Simple document
            text = TextGenerator.generate_text_with_pii_density(50, 0.2)
            document = DocumentGenerator.generate_simple_document(text, f"simple_doc_{i}")
        elif i % 3 == 1:
            # Structured document
            text = TextGenerator.generate_structured_content()
            document = DocumentGenerator.generate_simple_document(text, f"structured_doc_{i}")
        else:
            # Multi-section document
            sections = [
                TextGenerator.generate_text_with_pii_density(30, 0.15),
                TextGenerator.generate_text_with_pii_density(40, 0.25),
                TextGenerator.generate_structured_content()
            ]
            document = DocumentGenerator.generate_multi_section_document(sections, f"multi_doc_{i}")
        
        # Vary the policies
        privacy_levels = ["low", "medium", "high"]
        privacy_level = privacy_levels[i % len(privacy_levels)]
        
        if i % 4 == 0:
            policy = PolicyGenerator.generate_basic_policy(privacy_level)
        else:
            policy = PolicyGenerator.generate_comprehensive_policy(privacy_level)
        
        test_data.append((document, policy))
    
    return test_data
import re
from typing import Dict, Any
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    source_system: str
    classification: str
    weight: float
    file_path: str
    pii_masked: bool


class TransformRules:

    def __init__(self, source_config: Dict[str, Any]):
        self.source_config = source_config
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'address': re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b',
                                  re.IGNORECASE)
        }

    def apply_transform(self, content: str, source_system: str, file_path: str) -> tuple[str, DocumentMetadata]:
        source_config = self.source_config.get(source_system, {})

        transformed_content = content
        if source_config.get('pii_mask', False):
            transformed_content = self._mask_pii(content)

        metadata = DocumentMetadata(
            source_system=source_system,
            classification=source_config.get('classification', 'PUBLIC'),
            weight=source_config.get('weight', 1.0),
            file_path=file_path,
            pii_masked=source_config.get('pii_mask', False)
        )

        return transformed_content, metadata

    def _mask_pii(self, content: str) -> str:
        masked_content = content

        for pii_type, pattern in self.pii_patterns.items():
            if pii_type == 'email':
                masked_content = pattern.sub('[EMAIL_MASKED]', masked_content)
            elif pii_type == 'phone':
                masked_content = pattern.sub('[PHONE_MASKED]', masked_content)
            elif pii_type == 'ssn':
                masked_content = pattern.sub('[SSN_MASKED]', masked_content)
            elif pii_type == 'address':
                masked_content = pattern.sub('[ADDRESS_MASKED]', masked_content)

        return masked_content

    def get_classification_weight(self, source_system: str) -> float:
        return self.source_config.get(source_system, {}).get('weight', 1.0)
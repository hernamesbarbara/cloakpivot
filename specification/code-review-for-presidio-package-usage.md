# Code Review Requirements to Ensure Best Practices Using Presidio Python Package for PII Anonymization & Deanonymization

### 1. Overview

- **Problem**: CloakPivot leverages Presidio for PII detection and DocPivot for document processing. While the current implementation has good architectural foundations (lazy initialization in AnalyzerEngineWrapper, some session-scoped fixtures), there are opportunities to leverage Presidio more effectively. 

- **Objective**: Perform deep research to understand Presidio's best practices fully. 

Presidio's anonymizer module anonymizes detected PII text entities with desired values. Presidio supports both anonymization and deanonymization by applying different operators. Operators are built-in text manipulation classes which can be easily extended. Anonymizers are used to replace a PII entity text with some other value by applying a certain operator (e.g. replace, mask, redact, encrypt). Deanonymizers are used to revert the anonymization operation. (e.g. to decrypt an encrypted text). 

Use the memo: `sah memo get --id 01K4AEDFF2J2SRVFHSCGR4RAVQ` to learn more about Presidio's built-in `Anonymizers` and `Deanonymizers`.

### 2. Code Review

Evaluate opportunities to better leverage Presidio's built-in features. Some areas that we are likely reinventing the wheel include: `cloakpivot/unmasking/engine.py` and `cloakpivot/masking/engine.py` for example. 

### 3. Summarize your findings in a new report: PRESIDIO_FINDINGS.md. 

## CloakPivot: Full Presidio Integration for PII Anonymization/Deanonymization — Product Requirements Document (PRD)

### 1. Overview

- **Problem**: CloakPivot currently only leverages ~30% of Presidio's capabilities, using just the AnalyzerEngine for PII detection while reimplementing anonymization and deanonymization functionality that Presidio already provides through its AnonymizerEngine and DeanonymizerEngine. This results in ~1,500 lines of redundant code, missed optimization opportunities, and lack of access to battle-tested anonymization operators.

- **Solution**: Fully integrate Presidio's AnonymizerEngine and DeanonymizerEngine to replace custom masking/unmasking implementations, while maintaining CloakPivot's architectural strengths (lazy initialization, document structure preservation, CloakMap reversibility).

- **Outcomes**:
  - Reduce codebase complexity by ~60% through removal of redundant implementations
  - Leverage Presidio's optimized, battle-tested anonymization operators
  - Enable advanced features like native encryption/decryption workflows
  - Improve performance through Presidio's batch processing and C-extensions
  - Maintain backward compatibility with existing CloakPivot APIs

### 2. Goals and Non-Goals

- **Goals**:
  - Replace custom `MaskingEngine` with Presidio's `AnonymizerEngine`
  - Replace custom `UnmaskingEngine` with Presidio's `DeanonymizerEngine`
  - Map existing `StrategyKind` patterns to Presidio's `OperatorConfig`
  - Preserve all current functionality while gaining Presidio's additional capabilities
  - Maintain CloakMap-based reversibility for round-trip operations
  - Keep existing API contracts for backward compatibility
  - Enable access to Presidio's advanced features (operator chaining, ad-hoc recognizers)

- **Non-Goals**:
  - Breaking existing CloakPivot public APIs
  - Removing DocPivot integration or document structure preservation
  - Changing CloakMap format or storage mechanisms
  - Implementing new custom anonymization operators beyond Presidio's offerings

### 3. Users and Use Cases

- **Users**: Current CloakPivot users, data privacy teams, compliance officers, developers
- **Use Cases**:
  - Existing: All current masking/unmasking workflows must continue working
  - New: Leverage Presidio's encrypt/decrypt operators for stronger security
  - New: Use Presidio's operator chaining for complex anonymization workflows
  - New: Access Presidio's batch processing for high-volume operations
  - New: Utilize ad-hoc recognizers for quick pattern-based PII detection

### 4. Key Concepts

- **AnonymizerEngine**: Presidio's core engine for applying anonymization operators to detected PII
- **DeanonymizerEngine**: Presidio's engine for reversing anonymization operations
- **OperatorConfig**: Presidio's configuration for anonymization operators (replace, redact, mask, hash, encrypt, custom)
- **OperatorResult**: Output from anonymization containing transformed text and entity mappings
- **Strategy Adapter**: New component to map CloakPivot's Strategy to Presidio's OperatorConfig
- **Presidio Integration Layer**: Abstraction maintaining CloakPivot's API while using Presidio engines

### 5. System Architecture

- **Current Architecture** (to be replaced):
  ```
  Document → AnalyzerEngine → Custom MaskingEngine → Custom StrategyApplicator → Masked Document
                                        ↓
                                   CloakMap Storage
                                        ↓
  Masked Document → Custom UnmaskingEngine → Custom DocumentUnmasker → Original Document
  ```

- **New Architecture** (with Presidio integration):
  ```
  Document → AnalyzerEngine → PresidioMaskingAdapter → AnonymizerEngine → Masked Document
                                        ↓
                              CloakMap Storage (enhanced)
                                        ↓
  Masked Document → PresidioUnmaskingAdapter → DeanonymizerEngine → Original Document
  ```

- **Integration Components**:
  1. **PresidioMaskingAdapter**: Converts CloakPivot policies to Presidio operators
  2. **PresidioUnmaskingAdapter**: Manages deanonymization with CloakMap data
  3. **StrategyToOperatorMapper**: Maps Strategy enums to OperatorConfig
  4. **CloakMapEnhancer**: Stores Presidio operator results for perfect reversibility

### 6. Product Requirements

- **R1: Full Presidio Engine Integration**
  - Integrate `presidio_anonymizer.AnonymizerEngine` for all masking operations
  - Integrate `presidio_anonymizer.DeanonymizerEngine` for reversible operations
  - Support all Presidio built-in operators: replace, redact, mask, hash, encrypt, custom
  - Enable operator chaining and composition

- **R2: Strategy Mapping**
  - Map `StrategyKind.REDACT` → `OperatorConfig("redact")`
  - Map `StrategyKind.TEMPLATE` → `OperatorConfig("replace", {"new_value": template})`
  - Map `StrategyKind.HASH` → `OperatorConfig("hash", hash_params)`
  - Map `StrategyKind.PARTIAL` → `OperatorConfig("mask", mask_params)`
  - Map `StrategyKind.SURROGATE` → `OperatorConfig("replace", faker_params)`
  - Map `StrategyKind.CUSTOM` → `OperatorConfig("custom", lambda_params)`

- **R3: Backward Compatibility**
  - Maintain existing `MaskingEngine` and `UnmaskingEngine` APIs
  - Support existing `MaskingPolicy` and `Strategy` configurations
  - Preserve CloakMap format and round-trip capabilities
  - Provide migration utilities for existing CloakMaps

- **R4: Enhanced Reversibility**
  - Store Presidio `OperatorResult` data in CloakMap for perfect deanonymization
  - Support encryption/decryption workflows with key management
  - Maintain anchor-based position tracking for document structure

- **R5: Performance Optimization**
  - Leverage Presidio's batch processing for multiple documents
  - Use Presidio's optimized C-extensions for better performance
  - Implement connection pooling for analyzer/anonymizer engines
  - Support parallel processing through Presidio's native capabilities

- **R6: New Presidio Features**
  - Enable ad-hoc recognizers without custom classes
  - Support context enhancement for improved detection
  - Allow confidence score manipulation and normalization
  - Provide result filtering by score, entity type, or location

### 7. API Design

- **Migration Layer** (maintains backward compatibility):
  ```python
  class MaskingEngine:
      """Existing API maintained, internally uses Presidio"""
      def __init__(self, **kwargs):
          self._legacy_mode = kwargs.get('use_legacy', False)
          if not self._legacy_mode:
              self._engine = PresidioMaskingEngine(**kwargs)
          else:
              # Keep legacy implementation during transition
              self._engine = LegacyMaskingEngine(**kwargs)
      
      def mask_document(self, document, entities, policy, text_segments):
          return self._engine.mask_document(document, entities, policy, text_segments)
  ```

- **New Presidio Integration Classes**:
  ```python
  class PresidioMaskingEngine:
      """New implementation using AnonymizerEngine"""
      def __init__(self):
          self.anonymizer = AnonymizerEngine()
          self.operator_mapper = StrategyToOperatorMapper()
      
      def mask_document(self, document, entities, policy, text_segments):
          # Convert policy to operators
          operators = self.operator_mapper.policy_to_operators(policy)
          
          # Apply anonymization
          result = self.anonymizer.anonymize(
              text=self._extract_text(document),
              analyzer_results=entities,
              operators=operators
          )
          
          # Create enhanced CloakMap with operator results
          cloakmap = self._create_enhanced_cloakmap(result, document)
          
          return MaskingResult(
              masked_document=self._apply_to_document(result, document),
              cloakmap=cloakmap
          )
  
  class PresidioUnmaskingEngine:
      """New implementation using DeanonymizerEngine"""
      def __init__(self):
          self.deanonymizer = DeanonymizerEngine()
      
      def unmask_document(self, masked_document, cloakmap):
          # Extract operator results from enhanced CloakMap
          operator_results = self._extract_operator_results(cloakmap)
          
          # Apply deanonymization
          result = self.deanonymizer.deanonymize(
              text=self._extract_text(masked_document),
              entities=operator_results,
              operators=self._get_deanonymize_operators(cloakmap)
          )
          
          return UnmaskingResult(
              restored_document=self._apply_to_document(result, masked_document),
              cloakmap=cloakmap
          )
  ```

- **Strategy to Operator Mapping**:
  ```python
  class StrategyToOperatorMapper:
      """Maps CloakPivot strategies to Presidio operators"""
      
      STRATEGY_MAPPING = {
          StrategyKind.REDACT: lambda s: OperatorConfig(
              "redact",
              {"redact_char": s.get_parameter("redact_char", "*")}
          ),
          StrategyKind.TEMPLATE: lambda s: OperatorConfig(
              "replace",
              {"new_value": s.get_parameter("template", f"[{s.entity_type}]")}
          ),
          StrategyKind.HASH: lambda s: OperatorConfig(
              "hash",
              {
                  "hash_type": s.get_parameter("algorithm", "sha256"),
                  "salt": s.get_parameter("salt", "")
              }
          ),
          StrategyKind.PARTIAL: lambda s: OperatorConfig(
              "mask",
              {
                  "masking_char": s.get_parameter("mask_char", "*"),
                  "chars_to_mask": s.get_parameter("visible_chars", 4),
                  "from_end": s.get_parameter("position", "end") == "end"
              }
          ),
          StrategyKind.SURROGATE: lambda s: OperatorConfig(
              "replace",
              {"new_value": self._generate_fake_value(s)}
          ),
          StrategyKind.CUSTOM: lambda s: OperatorConfig(
              "custom",
              {"lambda": s.get_parameter("callback")}
          )
      }
      
      def strategy_to_operator(self, strategy: Strategy) -> OperatorConfig:
          return self.STRATEGY_MAPPING[strategy.kind](strategy)
      
      def policy_to_operators(self, policy: MaskingPolicy) -> dict:
          operators = {}
          for entity_type, strategy in policy.entity_strategies.items():
              operators[entity_type] = self.strategy_to_operator(strategy)
          return operators
  ```

### 8. Enhanced CloakMap Format

- **Current CloakMap** (preserved):
  ```json
  {
    "version": "1.0",
    "doc_id": "document_name",
    "anchors": [...],
    "policy_snapshot": {...}
  }
  ```

- **Enhanced CloakMap** (with Presidio data):
  ```json
  {
    "version": "2.0",
    "doc_id": "document_name",
    "anchors": [...],
    "policy_snapshot": {...},
    "presidio_metadata": {
      "engine_version": "2.x.x",
      "operator_results": [
        {
          "entity_type": "PHONE_NUMBER",
          "start": 10,
          "end": 22,
          "operator": "encrypt",
          "encrypted_value": "...",
          "key_reference": "key_id_123"
        }
      ],
      "reversible_operators": ["encrypt", "custom_reversible"],
      "batch_id": "optional_batch_tracking"
    }
  }
  ```

### 9. Migration Strategy

- **Phase 1: Parallel Implementation** (Weeks 1-2)
  - Create `PresidioMaskingEngine` alongside existing `MaskingEngine`
  - Create `PresidioUnmaskingEngine` alongside existing `UnmaskingEngine`
  - Implement `StrategyToOperatorMapper` with full mapping coverage
  - Add feature flag for toggling implementations

- **Phase 2: Integration Testing** (Weeks 3-4)
  - Create comprehensive test suite comparing outputs
  - Verify round-trip integrity with both implementations
  - Performance benchmarking: legacy vs Presidio
  - Edge case validation (overlapping entities, special characters)

- **Phase 3: Gradual Rollout** (Weeks 5-6)
  - Enable Presidio engine for new documents by default
  - Provide CLI flag for engine selection
  - Document migration path for existing users
  - Create migration utilities for existing CloakMaps

- **Phase 4: Legacy Deprecation** (Weeks 7-8)
  - Mark legacy engines as deprecated
  - Provide automated migration tools
  - Update all documentation and examples
  - Plan removal in next major version

### 10. Testing Strategy

- **Unit Tests**:
  - Test each operator mapping individually
  - Verify CloakMap enhancement preservation
  - Test backward compatibility layer
  - Validate error handling and fallbacks

- **Integration Tests**:
  - End-to-end masking/unmasking with Presidio engines
  - Cross-engine compatibility (mask with legacy, unmask with Presidio)
  - Performance comparison tests
  - Memory usage profiling

- **Regression Tests**:
  - All existing tests must pass with new implementation
  - Golden file comparisons for output consistency
  - API contract verification
  - CLI behavior validation

### 11. Performance Requirements

- **Target Improvements**:
  - 30-50% faster processing for documents >100 pages
  - 40% reduction in memory usage through Presidio's optimizations
  - Support for processing 10x more documents in batch mode
  - Sub-second response for single-page documents

- **Benchmarks**:
  - Baseline: Current implementation metrics
  - Goal: Match or exceed Presidio's standalone performance
  - Monitoring: Track regression in CI/CD pipeline

### 12. Security Enhancements

- **New Capabilities with Presidio**:
  - Native encryption/decryption operators with key management
  - FIPS-compliant hashing algorithms
  - Secure random generation for surrogates
  - Operator audit trails in CloakMap

- **Key Management Integration**:
  ```python
  class PresidioKeyManager:
      """Manages encryption keys for Presidio operators"""
      def get_encryption_key(self, key_id: str) -> bytes:
          # Integration with KMS, HashiCorp Vault, etc.
          pass
      
      def rotate_keys(self, cloakmap: CloakMap) -> CloakMap:
          # Re-encrypt with new keys
          pass
  ```

### 13. Documentation Updates

- **Migration Guide**:
  - Step-by-step migration instructions
  - Feature comparison table
  - Common migration scenarios
  - Troubleshooting guide

- **API Documentation**:
  - Updated docstrings with Presidio references
  - New operator configuration examples
  - Advanced features tutorial
  - Performance tuning guide

- **Examples**:
  - Basic masking with Presidio engines
  - Encryption/decryption workflows
  - Batch processing examples
  - Custom operator implementation

### 14. Rollback Plan

- **Feature Flags**:
  ```python
  CLOAKPIVOT_USE_PRESIDIO_ENGINE=true  # Toggle implementation
  CLOAKPIVOT_LEGACY_COMPATIBILITY=true  # Maintain old behavior
  ```

- **Rollback Triggers**:
  - Performance regression >20%
  - Round-trip integrity failures
  - Unexpected memory usage increase
  - Critical bug in operator mapping

- **Rollback Procedure**:
  1. Set feature flag to use legacy engine
  2. Deploy hotfix with flag change
  3. Investigate and fix issues
  4. Re-attempt migration with fixes

### 15. Success Metrics

- **Code Quality**:
  - 60% reduction in masking/unmasking code
  - 90% test coverage maintained
  - Zero critical bugs in production

- **Performance**:
  - 40% improvement in processing speed
  - 30% reduction in memory usage
  - 10x improvement in batch processing

- **Adoption**:
  - 100% of new documents use Presidio engine within 3 months
  - 80% of existing users migrated within 6 months
  - Positive feedback from performance improvements

### 16. Dependencies

- **Required**:
  - `presidio-analyzer` (already integrated)
  - `presidio-anonymizer` (to be added)
  - Existing: `docpivot`, `cryptography`

- **Optional**:
  - `faker` for enhanced surrogate generation
  - `hashicorp-vault` for key management
  - `prometheus-client` for metrics

### 17. Timeline

- **Week 1-2**: Implementation of Presidio engines and adapters
- **Week 3-4**: Integration testing and benchmarking
- **Week 5-6**: Documentation and migration tools
- **Week 7-8**: Gradual rollout and monitoring
- **Week 9-10**: Full deployment and legacy deprecation announcement
- **Week 11-12**: Performance optimization and bug fixes

### 18. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking API changes | Low | High | Comprehensive backward compatibility layer |
| Performance regression | Medium | Medium | Extensive benchmarking, feature flags for rollback |
| CloakMap incompatibility | Low | High | Version detection and automatic migration |
| Missing operator features | Low | Medium | Custom operator fallback implementation |
| User adoption resistance | Medium | Low | Clear migration benefits documentation |

### 19. Open Questions

- Should we maintain the custom `StrategyApplicator` as a fallback for edge cases Presidio doesn't handle?
- What is the timeline for deprecating and removing legacy implementations?
- Should we contribute any custom operators back to the Presidio project?
- How do we handle CloakMaps created with legacy engines in perpetuity?

### 20. Deliverables

- **Code**:
  - `PresidioMaskingEngine` and `PresidioUnmaskingEngine` implementations
  - `StrategyToOperatorMapper` with full mapping coverage
  - Enhanced CloakMap format with Presidio metadata
  - Migration utilities and compatibility layer

- **Documentation**:
  - Updated API documentation
  - Migration guide with examples
  - Performance comparison report
  - Security enhancement guide

- **Tools**:
  - CloakMap migration utility
  - Performance benchmarking suite
  - Compatibility verification tool
  - Operator mapping validator

### 21. Success Criteria

The integration will be considered successful when:
1. All existing CloakPivot functionality works with Presidio engines
2. Performance improves by at least 30% for standard use cases
3. Code complexity reduces by at least 50% in masking/unmasking modules
4. Zero regressions in round-trip integrity tests
5. Access to all Presidio advanced features (encryption, operator chaining, etc.)
6. Smooth migration path with no breaking changes for existing users
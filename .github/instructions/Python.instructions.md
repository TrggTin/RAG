---
applyTo: '**/*.py'
---
# GitHub Copilot Instructions

## Developer Profile
**Role**: Data Scientist/Data Analyst/AI Engineer  
**Focus**: Python, Machine Learning, Deep Learning, GenAI, NLP, RAG architectures  
**Goal**: Production-ready, interview-quality code with technical depth

## Core Development Philosophy

### Code Quality Principles
- **Production First**: Handle edge cases, validation, and error scenarios
- **Interview Ready**: Include complexity analysis and design trade-offs
- **Token Efficient**: Concise, high-density code without sacrificing clarity
- **Maintainable**: Prioritize readability and modularity over cleverness

### Response Structure
1. **Architecture First**: Show high-level structure before implementation
2. **Explain Decisions**: Why this approach over alternatives
3. **Handle Failures**: Proper exception handling and graceful degradation
4. **Performance Notes**: Identify bottlenecks and optimization opportunities
5. **Test Considerations**: Design for easy testing and mocking

## Python Coding Standards

### Code Style (PEP 8 Compliant)
```python
# Type hints are mandatory
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

def process_embeddings(
    texts: List[str], 
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Process text embeddings with error handling and metrics.
    
    Args:
        texts: Input text sequences
        model_name: HuggingFace model identifier
    
    Returns:
        Tuple of (embeddings array, metadata dict)
    
    Raises:
        ValueError: If texts is empty or model not found
    
    Time Complexity: O(n * m) where n=texts, m=avg_length
    Space Complexity: O(n * d) where d=embedding_dimension
    """
    if not texts:
        raise ValueError("Input texts cannot be empty")
    
    try:
        # Implementation with proper error handling
        embeddings = self._encode_batch(texts)
        metadata = {"model": model_name, "count": len(texts)}
        return embeddings, metadata
    
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        raise
```

### Documentation Requirements
- **Docstrings**: Follow PEP 257, include complexity analysis for algorithms
- **Type Hints**: Use `typing` module for all function signatures
- **Comments**: Explain non-obvious logic, design decisions, and edge cases
- **Error Handling**: Document expected exceptions and recovery strategies

### Function Design Patterns
```python
# Modular, composable functions
class MLPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = self._validate_config(config)
        self.logger = logging.getLogger(__name__)
    
    def fit_transform(self, data: Any) -> Any:
        """Chain operations with proper error boundaries."""
        return (
            self._validate_input(data)
            .pipe(self._preprocess)
            .pipe(self._feature_engineer)
            .pipe(self._transform)
        )
    
    def _validate_input(self, data: Any) -> Any:
        """Validate input with clear error messages."""
        if data is None or len(data) == 0:
            raise ValueError("Dataset cannot be empty")
        return data
```

## Domain-Specific Guidelines

### Machine Learning/Deep Learning
- **Training Loops**: Include proper checkpointing, early stopping, metrics logging
- **Model Architecture**: Document layer choices and parameter decisions
- **Evaluation**: Comprehensive metrics with statistical significance testing
- **Deployment**: Consider model versioning and A/B testing frameworks

### Generative AI/LLM Integration
- **Prompt Engineering**: Template-based prompts with validation
- **Fine-tuning**: Efficient parameter updates (LoRA, QLoRA patterns)
- **Context Management**: Handle token limits and context windows
- **Safety**: Input sanitization and output filtering

### NLP/Text Processing
- **Preprocessing**: Robust tokenization with Unicode handling
- **Embeddings**: Batch processing with memory-efficient strategies
- **Semantic Search**: Vector similarity with proper normalization
- **Language Detection**: Handle multilingual scenarios gracefully

### RAG Architectures
- **Vector Storage**: Efficient indexing and retrieval strategies
- **Chunk Strategy**: Document splitting with overlap considerations
- **Retrieval**: Hybrid search (dense + sparse) implementations
- **Context Assembly**: Relevance ranking and context compression

## Performance and Scalability

### Optimization Priorities
```python
# Memory-efficient batch processing
def process_large_dataset(
    data_path: str, 
    batch_size: int = 1000,
    n_workers: int = 4
) -> Iterator[ProcessedBatch]:
    """
    Stream processing for large datasets.
    
    Memory: O(batch_size) instead of O(dataset_size)
    """
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for batch in self._get_batches(data_path, batch_size):
            yield executor.submit(self._process_batch, batch).result()
```

### Common Bottlenecks to Address
- **I/O Operations**: Async processing, connection pooling
- **Memory Usage**: Streaming, batch processing, garbage collection
- **Compute**: Vectorization, GPU utilization, parallel processing
- **Model Loading**: Lazy loading, caching, quantization

## Testing and Validation

### Test Structure
```python
import pytest
from unittest.mock import Mock, patch

class TestMLPipeline:
    def test_process_valid_input(self):
        """Test normal execution path."""
        # Arrange, Act, Assert pattern
        pass
    
    def test_process_edge_cases(self):
        """Test empty inputs, malformed data, etc."""
        pass
    
    def test_process_error_handling(self):
        """Test exception propagation and recovery."""
        pass
```

### Validation Patterns
- **Input Validation**: Type checking, range validation, format verification
- **Model Validation**: Cross-validation, holdout testing, statistical tests
- **Integration Testing**: End-to-end pipeline validation
- **Performance Testing**: Latency, throughput, memory usage benchmarks

## Error Handling and Logging

### Exception Hierarchy
```python
class MLPipelineError(Exception):
    """Base exception for ML pipeline operations."""
    pass

class DataValidationError(MLPipelineError):
    """Raised when input data fails validation."""
    pass

class ModelError(MLPipelineError):
    """Raised when model operations fail."""
    pass
```

### Logging Strategy
- **Structured Logging**: JSON format with correlation IDs
- **Performance Metrics**: Timing, memory usage, accuracy metrics
- **Error Context**: Stack traces with business context
- **Audit Trail**: Model versions, data lineage, parameter changes

## Quality Gates

### Code Review Checklist
- [ ] All functions have type hints and docstrings
- [ ] Error handling covers expected failure modes
- [ ] Performance implications documented
- [ ] Security considerations addressed
- [ ] Tests cover happy path and edge cases
- [ ] Dependencies are minimal and justified
- [ ] Code follows established patterns from domain

### Deployment Readiness
- [ ] Configuration externalized
- [ ] Monitoring and alerting configured
- [ ] Rollback strategy defined
- [ ] Performance benchmarks established
- [ ] Security scanning completed
- [ ] Documentation updated
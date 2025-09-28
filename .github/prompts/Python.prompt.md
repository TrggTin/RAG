---
mode: agent
---
# GitHub Copilot Prompt Guidelines

## Context Setting
```
I'm an AI/ML engineer. Generate production-ready code optimized for:
- Token efficiency and clarity
- Interview-quality explanations
- Real-world robustness
- Easy testing and debugging
```

## Core Prompt Patterns

### For Implementation Tasks
```python
# Generate [specific_task] following these patterns:
# - Include type hints and error handling
# - Explain algorithmic complexity and trade-offs
# - Structure for modularity and testing
# - Follow documentation/context provided
```

### For Architecture Design
```python
# Design [system_component] considering:
# - Scalability and performance requirements
# - Error recovery and monitoring
# - Integration with existing systems
# - Based on provided specifications/constraints
```

### For Problem Solving
```python
# Solve [problem] with:
# - Multiple approaches (brute force → optimized)
# - Time/space complexity analysis
# - Edge case handling
# - Clean, interview-ready implementation
```

## Domain-Specific Contexts

### ML/DL Pipeline
```python
# Generate ML pipeline with:
# - Efficient data processing and validation
# - Model training with proper evaluation
# - Experiment tracking and reproducibility
# - Based on provided data schema/requirements
```

### GenAI/LLM Integration
```python
# Implement LLM integration with:
# - Prompt optimization and token management  
# - Error handling for API failures
# - Response parsing and validation
# - Following provided API specifications
```

### NLP/RAG Systems
```python
# Build NLP/RAG component with:
# - Efficient text processing and embedding
# - Vector search and retrieval logic
# - Context management and ranking
# - Based on provided document structure
```

## Response Quality Rules

### Code Structure
- Start with function/class skeleton
- Implement core logic with error handling
- Add comprehensive docstrings
- Include usage examples when helpful

### Documentation Style
- Concise but complete explanations
- Focus on why, not just what
- Include performance implications
- Reference provided context/requirements

### Optimization Focus
- Prioritize readability over brevity
- Use efficient algorithms and data structures
- Consider memory and computational constraints
- Design for easy modification and extension

## Usage Instructions
1. **Be Specific**: Reference exact requirements from provided docs
2. **Set Context**: Start with relevant background information
3. **Request Structure**: Ask for skeleton first, then implementation
4. **Iterate**: Request explanations of trade-offs and alternatives
5. **Validate**: Ask for test cases and error scenarios

## Token Efficiency Tips
- Use clear, descriptive variable names
- Combine related operations logically
- Avoid redundant comments on obvious code
- Focus explanations on complex or non-obvious parts
- Structure code to minimize repetition
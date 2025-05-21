# Project Evolution Summary: CAMEL Extension

## Overall Goal

This project aims to evolve `camel_ext` to implement mutual agent ranking/improvement capabilities and to optimize training for an 8GB NVIDIA RTX 4060 Ti GPU.

## Planned High-Level Steps

### Phase 1: Implementing Mutual Ranking and Improvement

1. **Modifying Workflow and Agent Logic**
   - Enhance [`camel_ext/camel/agents/peer_reviewer.py`](camel_ext/camel/agents/peer_reviewer.py) to create an LLM-based reviewer agent that produces structured feedback
   - Update [`camel_ext/camel/agents/proposer.py`](camel_ext/camel/agents/proposer.py) to incorporate feedback loops
   - Implement new interaction loop between agents
   - Update agent configuration in `agents.yaml`

2. **Data Logging**
   - Modify [`camel_ext/plugins/logging.py`](camel_ext/plugins/logging.py) to capture agent interactions, rankings, and improvement metrics
   - Create structures for storing feedback data

3. **Dual DPO Training**
   - Enhance [`camel_ext/scripts/train_dpo.py`](camel_ext/scripts/train_dpo.py) to implement training using the feedback data
   - Create evaluation metrics for agent improvement

### Phase 2: Optimizing for an 8GB RTX 4060 Ti

1. **Key Parameter Tuning**
   - Modify [`camel_ext/scripts/train_dpo.py`](camel_ext/scripts/train_dpo.py) to adjust batch sizes, model precision, and memory usage parameters
   - Implement efficient gradient accumulation

2. **Additional Optimizations**
   - Implement gradient checkpointing
   - Configure mixed precision training
   - Optimize memory usage through model pruning or quantization
   - Implement efficient data loading pipelines

## Work Undertaken and Changes Made

Work has been initiated on Phase 1 of the project, with focus on the following:

### Subtask 1: Enhance `PeerReviewer` Agent

- **File:** [`camel_ext/camel/agents/peer_reviewer.py`](camel_ext/camel/agents/peer_reviewer.py)
- **Objective:** Modify the `PeerReviewer` agent to be LLM-based and produce structured feedback for agent responses, enabling quantifiable improvement metrics.
- **Status:** This subtask was initiated but its completion status and specific changes are pending confirmation due to a previous interruption in the development process.

### Changes to the original CAMEL repo

- Changes to the original CAMEL repository are planned to be made incrementally.
- Currently, the only changes initiated relate to the `PeerReviewer` agent.
- Further changes will be documented as more subtasks are completed.
- The `camel_ext` directory structure will mirror the main CAMEL repository structure while adding new functionalities.
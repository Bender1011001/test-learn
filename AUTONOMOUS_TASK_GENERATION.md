# Autonomous Task Generation System

This document describes the autonomous task generation capabilities implemented for the CAMEL project's Proposer agent.

## Overview

The autonomous task generation system enables the Proposer agent to continuously create diverse, challenging tasks across multiple domains without manual intervention. The system includes:

- **Task complexity scoring and progressive difficulty**
- **Multiple task categories/domains**
- **Task validation and quality assessment**
- **Task queue management with prioritization**
- **Integration with existing workflow system**

## Architecture

### Core Components

1. **ProposerAgent** (`camel/agents/proposer.py`)
   - Autonomous task generation engine
   - Task quality validation
   - Task queue management
   - Progressive difficulty adjustment

2. **ExecutorAgent** (`camel/agents/executor.py`)
   - Task execution engine
   - Multi-domain task handling
   - Result analysis and scoring

3. **Database Models** (`backend/db/models/tasks.py`)
   - Task storage and tracking
   - Execution history
   - Feedback and performance metrics

4. **API Endpoints** (`backend/api/routers/tasks.py`)
   - RESTful task management
   - Queue status monitoring
   - Execution control

5. **Workflow Integration** (`backend/core/services/workflow_manager.py`)
   - Autonomous generation workflows
   - Task execution workflows
   - Performance monitoring

## Task Categories

The system supports six main task categories:

### 1. Coding Tasks
- Function implementation
- Algorithm development
- Data structure creation
- Code optimization
- **Languages**: Python, JavaScript, Java, C++, Go

### 2. Reasoning Tasks
- Logic puzzles
- Strategic analysis
- Problem decomposition
- Decision making

### 3. Creative Tasks
- Creative writing
- Design challenges
- Brainstorming exercises
- Innovation problems

### 4. Analytical Tasks
- Data interpretation
- Pattern recognition
- Statistical analysis
- Research synthesis

### 5. Problem Solving Tasks
- Process optimization
- System design
- Troubleshooting
- Root cause analysis

### 6. Data Analysis Tasks
- Dataset exploration
- Predictive modeling
- Visualization creation
- Pipeline development

## Difficulty Levels

Tasks are classified into three difficulty levels:

- **Beginner**: Simple, foundational tasks (Complexity: 1-3)
- **Intermediate**: Moderate complexity tasks (Complexity: 4-7)
- **Advanced**: Complex, challenging tasks (Complexity: 8-10)

## Task Generation Features

### Quality Controls
- Minimum complexity score: 2.0
- Maximum complexity score: 9.0
- Diversity threshold: 0.7
- Success criteria validation
- Prerequisites checking

### Rate Limiting
- Configurable generation rate (default: 10 tasks/minute)
- Queue size limits (default: 100 tasks)
- Adaptive throttling based on queue status

### Progressive Difficulty
- Automatic difficulty adjustment based on success rates
- Feedback-driven complexity scaling
- Performance-based task selection

## API Endpoints

### Task Generation
```http
POST /api/v1/tasks/generate
Content-Type: application/json

{
  "category": "coding",
  "difficulty": "intermediate",
  "priority": "medium",
  "count": 5
}
```

### Task Execution
```http
POST /api/v1/tasks/{task_id}/execute
Content-Type: application/json

{
  "executor_agent": "ExecutorAgent"
}
```

### Queue Status
```http
GET /api/v1/tasks/queue/status
```

### Task Feedback
```http
POST /api/v1/tasks/{task_id}/feedback
Content-Type: application/json

{
  "overall_rating": 8.5,
  "strengths": ["Clear requirements", "Good complexity"],
  "areas_for_improvement": ["More context needed"],
  "effectiveness_score": 8.0,
  "correctness_score": 9.0
}
```

## Workflow Integration

### Autonomous Generation Workflow
```python
# Start autonomous task generation
run_id = await workflow_manager.start_autonomous_task_generation({
    "categories": ["coding", "reasoning"],
    "difficulties": ["beginner", "intermediate", "advanced"],
    "generation_rate": 5,  # tasks per minute
    "max_queue_size": 50
})

# Monitor status
status = workflow_manager.get_autonomous_generation_status(run_id)

# Stop generation
await workflow_manager.stop_autonomous_task_generation(run_id)
```

### Task Execution Workflow
```python
# Execute a specific task
run_id = await workflow_manager.execute_task_workflow(
    task_id="task_12345",
    executor_settings={"timeout": 300}
)
```

## Configuration

### Agent Configuration (`configs/agents.yaml`)
```yaml
agents:
  Proposer:
    parameters:
      generation_rate_limit: 10
      min_complexity_score: 2.0
      max_complexity_score: 9.0
      diversity_threshold: 0.7
      preferred_categories: ["coding", "reasoning", "creative"]
      enable_adaptive_difficulty: true
      enable_feedback_learning: true

  Executor:
    parameters:
      max_execution_time: 300
      supported_languages: ["python", "javascript", "java"]
      quality_scoring_enabled: true
```

### Workflow Configuration
```yaml
workflows:
  autonomous_task_generation:
    description: "Continuous autonomous task generation"
    settings:
      generation_rate: 5
      max_queue_size: 50
      quality_threshold: 6.0
      adaptive_difficulty: true

  autonomous_learning_loop:
    description: "Complete learning loop with generation, execution, and feedback"
    agent_sequence: ["Proposer", "Executor", "PeerReviewer"]
    settings:
      continuous_operation: true
      feedback_learning_enabled: true
      auto_dpo_training: true
```

## Database Schema

### Tasks Table
- `task_id`: Unique task identifier
- `title`: Task title
- `description`: Detailed task description
- `category`: Task category (enum)
- `difficulty`: Difficulty level (enum)
- `complexity_score`: Calculated complexity (1-10)
- `success_criteria`: JSON array of success criteria
- `evaluation_metrics`: JSON object with evaluation metrics

### Task Executions Table
- `execution_id`: Unique execution identifier
- `task_id`: Reference to task
- `executor_agent`: Agent that executed the task
- `execution_output`: JSON with execution results
- `quality_score`: Overall quality score
- `execution_time`: Time taken to execute

### Task Feedback Table
- `feedback_id`: Unique feedback identifier
- `task_id`: Reference to task
- `overall_rating`: Rating (1-10)
- `strengths`: JSON array of strengths
- `areas_for_improvement`: JSON array of improvements
- `detailed_feedback`: Text feedback

## Usage Examples

### 1. Generate Tasks Programmatically
```python
from camel.agents.proposer import ProposerAgent, TaskCategory, TaskDifficulty
from camel.messages import SystemMessage

# Create proposer agent
system_message = SystemMessage(
    role_name="Proposer",
    content="Generate diverse, challenging tasks"
)
proposer = ProposerAgent(system_message=system_message)

# Generate a coding task
task = proposer._generate_autonomous_task({
    "category": TaskCategory.CODING,
    "difficulty": TaskDifficulty.INTERMEDIATE
})

# Validate and add to queue
if proposer._validate_task_quality(task):
    proposer.task_queue.add_task(task)
```

### 2. Execute Tasks
```python
from camel.agents.executor import ExecutorAgent
from camel.messages import SystemMessage, UserMessage

# Create executor agent
system_message = SystemMessage(
    role_name="Executor",
    content="Execute tasks with detailed results"
)
executor = ExecutorAgent(system_message=system_message)

# Execute a task
task_message = UserMessage(role_name="user", content=task_description)
results = executor.step(task_message)
```

### 3. Monitor Queue Status
```python
# Get queue status
status = proposer.get_queue_status()
print(f"Total tasks: {status['total_tasks']}")
print(f"Difficulty distribution: {status['difficulty_distribution']}")
print(f"Average complexity: {status['average_complexity']}")
```

## Testing

Run the test script to verify the implementation:

```bash
python scripts/test_autonomous_task_generation.py
```

This will test:
- Task generation capabilities
- Task execution functionality
- Workflow integration
- API endpoints (if server is running)

## Performance Metrics

The system tracks various performance metrics:

- **Generation Rate**: Tasks generated per minute
- **Queue Utilization**: Current queue size vs. maximum
- **Completion Rate**: Percentage of successfully completed tasks
- **Average Quality Score**: Mean quality score across all executions
- **Complexity Distribution**: Distribution of task complexities
- **Category Balance**: Distribution across task categories

## Future Enhancements

1. **Machine Learning Integration**
   - Task difficulty prediction
   - Success rate optimization
   - Personalized task generation

2. **Advanced Quality Controls**
   - Semantic similarity checking
   - Automated task validation
   - Duplicate detection

3. **Enhanced Feedback Loop**
   - Real-time difficulty adjustment
   - Performance-based task selection
   - Automated DPO training integration

4. **Multi-Agent Collaboration**
   - Collaborative task generation
   - Peer review integration
   - Consensus-based quality assessment

## Troubleshooting

### Common Issues

1. **Task Generation Fails**
   - Check agent configuration
   - Verify template availability
   - Review quality thresholds

2. **Queue Fills Up**
   - Increase queue size limit
   - Reduce generation rate
   - Check task execution rate

3. **Low Quality Scores**
   - Adjust complexity thresholds
   - Review success criteria
   - Update task templates

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check workflow status:
```python
status = workflow_manager.get_workflow_status(run_id)
print(f"Status: {status}")
```

## Contributing

When contributing to the autonomous task generation system:

1. Follow the existing code patterns
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Test with multiple task categories

## License

This implementation is part of the CAMEL project and follows the same licensing terms.
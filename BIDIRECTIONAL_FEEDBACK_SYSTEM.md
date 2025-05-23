# Bidirectional Feedback System

This document describes the comprehensive bidirectional feedback system implemented for the CAMEL project, which enables all agents to evaluate each other's performance and creates a complete feedback loop for continuous improvement.

## Overview

The bidirectional feedback system addresses the critical gap in the original design where feedback was unidirectional. Now all agents can evaluate each other, creating a rich feedback ecosystem that drives autonomous improvement through machine learning.

### Key Features

- **Cross-Agent Evaluation**: Every agent can evaluate every other agent's performance
- **Structured Feedback Collection**: Standardized feedback format with quantitative and qualitative metrics
- **Performance Tracking**: Historical performance monitoring with trend analysis
- **Autonomous Learning Loop**: Complete cycle of generation → execution → feedback → improvement
- **DPO Training Integration**: Automatic export of feedback data for Direct Preference Optimization
- **Real-time Monitoring**: Live performance dashboards and insights

## Architecture

### Core Components

1. **BidirectionalFeedbackManager** (`camel/agents/bidirectional_feedback.py`)
   - Central coordinator for all feedback operations
   - Manages feedback collection, storage, and analysis
   - Provides performance insights and trend analysis

2. **BidirectionalFeedbackAgent** (`camel/agents/bidirectional_feedback.py`)
   - Wrapper that adds feedback capabilities to any agent
   - Enables agents to both give and receive feedback
   - Tracks interaction context for feedback collection

3. **Workflow Integration** (`backend/core/services/workflow_manager.py`)
   - Bidirectional feedback workflows
   - Autonomous learning loop implementation
   - Performance monitoring and training triggers

4. **API Endpoints** (`backend/api/routers/feedback.py`)
   - RESTful interface for feedback management
   - Real-time status monitoring
   - Data export capabilities

## Feedback Types

The system supports multiple types of feedback based on agent roles:

### Task Quality Feedback
- **Evaluator**: Executor, PeerReviewer
- **Evaluated**: Proposer
- **Focus**: Task clarity, feasibility, scope, context, success criteria

### Execution Quality Feedback
- **Evaluator**: Proposer, PeerReviewer
- **Evaluated**: Executor
- **Focus**: Correctness, methodology, explanation quality, code quality, innovation

### Review Quality Feedback
- **Evaluator**: Proposer, Executor
- **Evaluated**: PeerReviewer
- **Focus**: Accuracy, constructiveness, specificity, fairness, comprehensiveness

### Additional Feedback Types
- Communication Clarity
- Problem Solving
- Creativity
- Efficiency

## Feedback Structure

Each feedback entry contains:

```python
@dataclass
class FeedbackEntry:
    feedback_id: str
    evaluator_agent: str
    evaluated_agent: str
    feedback_type: FeedbackType
    overall_rating: float  # 1-10 scale
    specific_scores: Dict[str, float]
    strengths: List[str]
    areas_for_improvement: List[str]
    detailed_feedback: str
    context: Dict[str, Any]
    timestamp: datetime
    confidence_score: float  # How confident the evaluator is
```

## Usage Examples

### 1. Basic Feedback Collection

```python
from camel.agents.bidirectional_feedback import BidirectionalFeedbackManager, FeedbackType
from camel.agents.proposer import ProposerAgent
from camel.messages import SystemMessage

# Create feedback manager
feedback_manager = BidirectionalFeedbackManager()

# Create agent
proposer = ProposerAgent(system_message=SystemMessage(
    role_name="Proposer",
    content="You are a task proposer agent."
))

# Collect feedback
interaction_context = {
    "task_description": "Write a sorting algorithm",
    "executor_response": "def bubble_sort(arr): ..."
}

feedback_entry = await feedback_manager.collect_feedback(
    evaluator_agent=proposer,
    evaluated_agent_name="Executor",
    interaction_context=interaction_context,
    feedback_type=FeedbackType.EXECUTION_QUALITY
)
```

### 2. Workflow Integration

```python
# Start bidirectional feedback workflow
run_id = await workflow_manager.start_bidirectional_feedback_workflow(
    interaction_context={
        "task_description": "Create a web scraper",
        "workflow_type": "standard",
        "agents_involved": ["Proposer", "Executor", "PeerReviewer"]
    },
    agents_involved=["Proposer", "Executor", "PeerReviewer"]
)

# Monitor progress
status = workflow_manager.get_workflow_status(run_id)
print(f"Feedback collected: {status['feedback_collected']}")
```

### 3. Autonomous Learning Loop

```python
# Start complete learning loop
loop_run_id = await workflow_manager.start_autonomous_learning_loop({
    "generation_rate": 5,  # tasks per minute
    "max_concurrent_tasks": 3,
    "feedback_frequency": "after_each_task",
    "auto_dpo_training": True,
    "performance_threshold": 7.0,
    "continuous_operation": True
})
```

### 4. Performance Analysis

```python
# Get agent performance summary
summary = feedback_manager.get_agent_performance_summary("Executor")
print(f"Average rating: {summary['overall_performance']['average_rating']}")
print(f"Trend: {summary['overall_performance']['trend_description']}")

# Get system insights
insights = feedback_manager.get_feedback_insights()
print(f"System average: {insights['system_overview']['average_rating']}")
print(f"Top strength: {insights['common_strengths'][0]}")
```

## API Endpoints

### Feedback Collection
```http
POST /api/v1/feedback/collect
Content-Type: application/json

{
  "interaction_context": {
    "task_description": "Write a function to parse JSON",
    "timestamp": "2024-01-01T12:00:00Z"
  },
  "agents_involved": ["Proposer", "Executor", "PeerReviewer"]
}
```

### Autonomous Learning Loop
```http
POST /api/v1/feedback/learning-loop
Content-Type: application/json

{
  "generation_rate": 3,
  "max_concurrent_tasks": 5,
  "auto_dpo_training": true,
  "performance_threshold": 7.0,
  "continuous_operation": true
}
```

### Performance Summaries
```http
GET /api/v1/feedback/summary/Executor
GET /api/v1/feedback/summary
```

### DPO Data Export
```http
GET /api/v1/feedback/export/dpo?min_rating=6.0
```

### Workflow Management
```http
GET /api/v1/feedback/workflow/{run_id}/status
POST /api/v1/feedback/workflow/{run_id}/stop
```

### System Health
```http
GET /api/v1/feedback/health
```

### Interaction Simulation
```http
POST /api/v1/feedback/simulate-interaction?task_description=Write a hello world function
```

## Autonomous Learning Loop

The autonomous learning loop is the crown jewel of the system, implementing a complete cycle:

### Phase 1: Task Generation
- Autonomous task creation by Proposer agent
- Diverse task categories and difficulty levels
- Quality validation and queue management

### Phase 2: Task Execution
- Automatic task execution by Executor agent
- Multi-domain task handling
- Result analysis and scoring

### Phase 3: Bidirectional Feedback
- Cross-agent evaluation of all interactions
- Structured feedback collection
- Performance metric calculation

### Phase 4: Learning and Improvement
- Performance analysis and trend detection
- Automatic DPO training triggers
- Model improvement and adaptation

### Configuration

```yaml
autonomous_learning_loop:
  generation_rate: 5  # tasks per minute
  max_concurrent_tasks: 3
  feedback_frequency: "after_each_task"
  auto_dpo_training: true
  performance_threshold: 7.0
  max_iterations: 100
  continuous_operation: true
```

## Performance Metrics

### Agent-Level Metrics
- **Average Overall Rating**: Mean rating across all feedback received
- **Performance by Type**: Ratings broken down by feedback type
- **Improvement Trend**: Direction and rate of performance change
- **Feedback Activity**: How much feedback the agent provides
- **Feedback Quality Score**: How good the agent is at giving feedback

### System-Level Metrics
- **Total Feedback Entries**: Overall feedback volume
- **Average System Rating**: Mean performance across all agents
- **Agent Rankings**: Comparative performance rankings
- **Common Strengths/Improvements**: Frequently mentioned patterns
- **Feedback Coverage**: How well the system covers agent interactions

## Integration with DPO Training

The feedback system seamlessly integrates with Direct Preference Optimization:

### Data Export Format
```python
{
    "id": "feedback_12345",
    "context": "Original task context",
    "evaluator": "Proposer",
    "evaluated": "Executor", 
    "feedback_type": "execution_quality",
    "rating": 8.5,
    "scores": {"correctness": 9, "efficiency": 8},
    "strengths": ["Clear implementation", "Good error handling"],
    "improvements": ["Could optimize for speed"],
    "detailed_feedback": "The solution is correct and well-structured...",
    "confidence": 0.9,
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Automatic Training Triggers
- Performance drops below threshold
- Significant trend changes detected
- Feedback quality degradation
- Configurable training schedules

## Testing

Run the comprehensive test suite:

```bash
python scripts/test_bidirectional_feedback.py
```

This tests:
- Feedback manager functionality
- Workflow integration
- API endpoints
- Performance analysis
- DPO data export

## Monitoring and Debugging

### Health Checks
```http
GET /api/v1/feedback/health
```

Returns:
- Active feedback workflows
- System performance metrics
- Error status and diagnostics

### Performance Trends
```http
GET /api/v1/feedback/agents/{agent_name}/performance-trend?days=7
```

Returns historical performance data with trend analysis.

### Workflow Status
```http
GET /api/v1/feedback/workflow/{run_id}/status
```

Real-time workflow progress and results.

## Configuration

### Agent Configuration (`configs/agents.yaml`)
```yaml
agents:
  Proposer:
    parameters:
      feedback_enabled: true
      evaluation_criteria: ["task_clarity", "feasibility", "scope"]
      
  Executor:
    parameters:
      feedback_enabled: true
      evaluation_criteria: ["correctness", "efficiency", "explanation"]
      
  PeerReviewer:
    parameters:
      feedback_enabled: true
      evaluation_criteria: ["accuracy", "constructiveness", "fairness"]
```

### Feedback System Configuration
```yaml
feedback_system:
  collection_rate_limit: 10  # feedback per minute per agent
  min_confidence_threshold: 0.6
  auto_aggregation: true
  performance_window_days: 30
  trend_analysis_enabled: true
```

## Best Practices

### 1. Feedback Quality
- Provide specific, actionable feedback
- Include both strengths and improvements
- Use appropriate confidence scores
- Maintain consistent evaluation criteria

### 2. Performance Monitoring
- Regular health checks
- Trend analysis review
- Performance threshold tuning
- Feedback quality assessment

### 3. Training Integration
- Regular DPO training cycles
- Performance-based training triggers
- Feedback data quality validation
- Model performance tracking

### 4. System Optimization
- Feedback collection rate limiting
- Efficient workflow scheduling
- Resource usage monitoring
- Error handling and recovery

## Troubleshooting

### Common Issues

1. **Low Feedback Quality**
   - Check agent evaluation prompts
   - Verify feedback templates
   - Review confidence thresholds

2. **Performance Degradation**
   - Analyze trend data
   - Check for feedback bias
   - Verify training data quality

3. **Workflow Failures**
   - Check agent initialization
   - Verify database connections
   - Review error logs

4. **API Errors**
   - Validate request formats
   - Check authentication
   - Verify endpoint availability

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features
1. **Multi-Modal Feedback**: Support for different feedback modalities
2. **Federated Learning**: Distributed feedback collection
3. **Advanced Analytics**: ML-powered insight generation
4. **Custom Feedback Types**: User-defined evaluation criteria
5. **Real-time Adaptation**: Dynamic feedback adjustment
6. **Cross-Domain Learning**: Knowledge transfer between domains

### Research Directions
- Feedback quality optimization
- Bias detection and mitigation
- Automated evaluation criteria generation
- Meta-learning for feedback systems
- Collaborative filtering for agent improvement

## Contributing

When contributing to the bidirectional feedback system:

1. Follow existing code patterns and structure
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure backward compatibility
5. Test with multiple agent configurations

## License

This implementation is part of the CAMEL project and follows the same licensing terms.
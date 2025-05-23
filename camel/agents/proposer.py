"""
CAMEL Extensions - Proposer Agent with Autonomous Task Generation

This module implements the ProposerAgent, which autonomously generates diverse,
challenging tasks for the Executor agent to solve. The agent includes:

1. Task complexity scoring and progressive difficulty
2. Multiple task categories/domains  
3. Task validation and quality assessment
4. Task queue management with prioritization
5. Integration with the existing workflow system

The ProposerAgent serves as the task creation component in the CAMEL workflow by:
- Generating tasks across different domains (coding, reasoning, creative, analytical)
- Assessing task difficulty levels (beginner, intermediate, advanced)
- Managing task queues with intelligent prioritization
- Providing success criteria and evaluation metrics
- Implementing quality controls and rate limiting
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

# CAMEL library imports
from camel.agents import BaseAgent
from camel.messages import BaseMessage, AssistantMessage, UserMessage, SystemMessage
from camel.types import AgentType


class TaskDifficulty(Enum):
    """Task difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"


class TaskCategory(Enum):
    """Task categories/domains"""
    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    PROBLEM_SOLVING = "problem_solving"
    DATA_ANALYSIS = "data_analysis"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Task:
    """Represents a generated task with metadata"""
    
    def __init__(
        self,
        task_id: str,
        title: str,
        description: str,
        category: TaskCategory,
        difficulty: TaskDifficulty,
        priority: TaskPriority = TaskPriority.MEDIUM,
        success_criteria: List[str] = None,
        evaluation_metrics: Dict[str, Any] = None,
        estimated_duration: int = 30,  # minutes
        prerequisites: List[str] = None,
        tags: List[str] = None
    ):
        self.task_id = task_id
        self.title = title
        self.description = description
        self.category = category
        self.difficulty = difficulty
        self.priority = priority
        self.success_criteria = success_criteria or []
        self.evaluation_metrics = evaluation_metrics or {}
        self.estimated_duration = estimated_duration
        self.prerequisites = prerequisites or []
        self.tags = tags or []
        self.created_at = datetime.utcnow()
        self.complexity_score = self._calculate_complexity_score()
        
    def _calculate_complexity_score(self) -> float:
        """Calculate task complexity score (0-10)"""
        base_score = {
            TaskDifficulty.BEGINNER: 2.0,
            TaskDifficulty.INTERMEDIATE: 5.0,
            TaskDifficulty.ADVANCED: 8.0
        }[self.difficulty]
        
        # Adjust based on category
        category_multiplier = {
            TaskCategory.CODING: 1.2,
            TaskCategory.REASONING: 1.1,
            TaskCategory.CREATIVE: 0.9,
            TaskCategory.ANALYTICAL: 1.3,
            TaskCategory.PROBLEM_SOLVING: 1.1,
            TaskCategory.DATA_ANALYSIS: 1.4
        }[self.category]
        
        # Adjust based on prerequisites and success criteria
        complexity_factors = (
            len(self.prerequisites) * 0.2 +
            len(self.success_criteria) * 0.1 +
            (self.estimated_duration / 60) * 0.5
        )
        
        final_score = min(10.0, base_score * category_multiplier + complexity_factors)
        return round(final_score, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "priority": self.priority.value,
            "success_criteria": self.success_criteria,
            "evaluation_metrics": self.evaluation_metrics,
            "estimated_duration": self.estimated_duration,
            "prerequisites": self.prerequisites,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "complexity_score": self.complexity_score
        }


class TaskQueue:
    """Manages task queue with prioritization and filtering"""
    
    def __init__(self, max_size: int = 100):
        self.tasks: List[Task] = []
        self.max_size = max_size
        self.completed_tasks: List[str] = []
        
    def add_task(self, task: Task) -> bool:
        """Add task to queue with priority ordering"""
        if len(self.tasks) >= self.max_size:
            # Remove lowest priority task if queue is full
            self.tasks.sort(key=lambda t: (t.priority.value, t.complexity_score))
            self.tasks.pop(0)
            
        self.tasks.append(task)
        self._sort_by_priority()
        return True
        
    def get_next_task(self, difficulty_filter: Optional[TaskDifficulty] = None) -> Optional[Task]:
        """Get next task from queue with optional difficulty filtering"""
        if not self.tasks:
            return None
            
        if difficulty_filter:
            filtered_tasks = [t for t in self.tasks if t.difficulty == difficulty_filter]
            if filtered_tasks:
                task = filtered_tasks[0]
                self.tasks.remove(task)
                return task
                
        # Return highest priority task
        return self.tasks.pop(0) if self.tasks else None
        
    def _sort_by_priority(self):
        """Sort tasks by priority and complexity"""
        self.tasks.sort(key=lambda t: (-t.priority.value, -t.complexity_score))
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        difficulty_counts = {}
        category_counts = {}
        
        for task in self.tasks:
            difficulty_counts[task.difficulty.value] = difficulty_counts.get(task.difficulty.value, 0) + 1
            category_counts[task.category.value] = category_counts.get(task.category.value, 0) + 1
            
        return {
            "total_tasks": len(self.tasks),
            "difficulty_distribution": difficulty_counts,
            "category_distribution": category_counts,
            "average_complexity": sum(t.complexity_score for t in self.tasks) / len(self.tasks) if self.tasks else 0,
            "completed_tasks": len(self.completed_tasks)
        }


class ProposerAgent(BaseAgent):
    """
    ProposerAgent with autonomous task generation capabilities.
    
    This agent can autonomously create diverse, challenging tasks across
    multiple domains with appropriate difficulty levels and success criteria.
    """
    
    def __init__(
        self,
        system_message: SystemMessage,
        **kwargs
    ):
        """
        Initialize the ProposerAgent.
        
        Args:
            system_message: SystemMessage defining the proposer's role and behavior
            **kwargs: Additional arguments passed to the BaseAgent
        """
        super().__init__(system_message=system_message, **kwargs)
        self.agent_type = AgentType.ASSISTANT
        
        # Task generation settings
        self.task_queue = TaskQueue()
        self.generation_rate_limit = 10  # max tasks per minute
        self.last_generation_time = datetime.utcnow()
        self.generation_count = 0
        
        # Task templates for different categories
        self.task_templates = self._initialize_task_templates()
        
        # Quality control settings
        self.min_complexity_score = 2.0
        self.max_complexity_score = 9.0
        self.diversity_threshold = 0.7  # minimum diversity in task queue
        
    def step(self, message: BaseMessage, chat_history: Optional[List[BaseMessage]] = None) -> List[BaseMessage]:
        """
        Process input and generate appropriate response.
        
        Args:
            message: Input message (could be request for task or feedback)
            chat_history: Optional chat history for context
            
        Returns:
            List of message(s) containing task or response
        """
        # Check if this is a request for autonomous task generation
        if self._is_task_generation_request(message):
            return self._handle_task_generation_request(message)
        
        # Check if this is feedback on a previous task
        if self._is_task_feedback(message):
            return self._handle_task_feedback(message, chat_history)
            
        # Default: generate a task based on the input
        return self._generate_contextual_task(message, chat_history)
    
    def _is_task_generation_request(self, message: BaseMessage) -> bool:
        """Check if message is requesting autonomous task generation"""
        content = message.content.lower()
        keywords = ["generate task", "create task", "new task", "autonomous task", "task generation"]
        return any(keyword in content for keyword in keywords)
    
    def _is_task_feedback(self, message: BaseMessage) -> bool:
        """Check if message contains feedback on a previous task"""
        content = message.content.lower()
        feedback_keywords = ["completed", "finished", "result", "feedback", "evaluation"]
        return any(keyword in content for keyword in keywords)
    
    def _handle_task_generation_request(self, message: BaseMessage) -> List[BaseMessage]:
        """Handle autonomous task generation request"""
        # Check rate limiting
        if not self._check_rate_limit():
            return [AssistantMessage(
                role_name="Proposer",
                content="Task generation rate limit exceeded. Please wait before requesting more tasks."
            )]
        
        # Parse any specific requirements from the message
        requirements = self._parse_task_requirements(message.content)
        
        # Generate task based on requirements
        task = self._generate_autonomous_task(requirements)
        
        if task and self._validate_task_quality(task):
            self.task_queue.add_task(task)
            response_content = self._format_task_response(task)
            
            return [AssistantMessage(role_name="Proposer", content=response_content)]
        else:
            return [AssistantMessage(
                role_name="Proposer", 
                content="Failed to generate a valid task. Please try again with different requirements."
            )]
    
    def _handle_task_feedback(self, message: BaseMessage, chat_history: Optional[List[BaseMessage]]) -> List[BaseMessage]:
        """Handle feedback on completed tasks"""
        # Extract task performance data
        feedback_data = self._parse_feedback(message.content)
        
        # Update task generation strategy based on feedback
        self._update_generation_strategy(feedback_data)
        
        # Generate follow-up task if appropriate
        if feedback_data.get("success", False):
            next_task = self._generate_progressive_task(feedback_data)
            if next_task:
                self.task_queue.add_task(next_task)
                response_content = f"Great work! Here's your next challenge:\n\n{self._format_task_response(next_task)}"
                return [AssistantMessage(role_name="Proposer", content=response_content)]
        
        return [AssistantMessage(
            role_name="Proposer",
            content="Thank you for the feedback. I'll use this to improve future task generation."
        )]
    
    def _generate_contextual_task(self, message: BaseMessage, chat_history: Optional[List[BaseMessage]]) -> List[BaseMessage]:
        """Generate a task based on the input context"""
        # Analyze the input to determine appropriate task type
        context_analysis = self._analyze_context(message.content, chat_history)
        
        # Generate task based on context
        task = self._create_task_from_context(context_analysis)
        
        if task and self._validate_task_quality(task):
            self.task_queue.add_task(task)
            response_content = self._format_task_response(task)
            return [AssistantMessage(role_name="Proposer", content=response_content)]
        
        # Fallback to default response
        return [AssistantMessage(
            role_name="Proposer",
            content="I understand your request. Let me suggest an appropriate task for you to work on."
        )]
    
    def _generate_autonomous_task(self, requirements: Dict[str, Any]) -> Optional[Task]:
        """Generate a task autonomously based on requirements"""
        # Determine task category
        category = requirements.get("category")
        if not category:
            category = random.choice(list(TaskCategory))
        elif isinstance(category, str):
            try:
                category = TaskCategory(category.lower())
            except ValueError:
                category = random.choice(list(TaskCategory))
        
        # Determine difficulty
        difficulty = requirements.get("difficulty")
        if not difficulty:
            difficulty = self._select_progressive_difficulty()
        elif isinstance(difficulty, str):
            try:
                difficulty = TaskDifficulty(difficulty.lower())
            except ValueError:
                difficulty = TaskDifficulty.INTERMEDIATE
        
        # Get task template
        template = self._get_task_template(category, difficulty)
        if not template:
            return None
        
        # Generate unique task ID
        task_id = f"task_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Create task instance
        task = Task(
            task_id=task_id,
            title=template["title"],
            description=template["description"],
            category=category,
            difficulty=difficulty,
            priority=requirements.get("priority", TaskPriority.MEDIUM),
            success_criteria=template.get("success_criteria", []),
            evaluation_metrics=template.get("evaluation_metrics", {}),
            estimated_duration=template.get("estimated_duration", 30),
            prerequisites=template.get("prerequisites", []),
            tags=template.get("tags", [])
        )
        
        return task
    
    def _get_task_template(self, category: TaskCategory, difficulty: TaskDifficulty) -> Optional[Dict[str, Any]]:
        """Get task template for given category and difficulty"""
        templates = self.task_templates.get(category, {}).get(difficulty, [])
        if not templates:
            return None
        
        # Select random template and customize it
        base_template = random.choice(templates)
        
        # Add some randomization to make tasks unique
        customized_template = self._customize_template(base_template, category, difficulty)
        
        return customized_template
    
    def _customize_template(self, template: Dict[str, Any], category: TaskCategory, difficulty: TaskDifficulty) -> Dict[str, Any]:
        """Customize a task template to make it unique"""
        customized = template.copy()
        
        # Add category-specific customizations
        if category == TaskCategory.CODING:
            languages = ["Python", "JavaScript", "Java", "C++", "Go"]
            language = random.choice(languages)
            customized["description"] = customized["description"].replace("{language}", language)
            customized["tags"].append(language.lower())
            
        elif category == TaskCategory.DATA_ANALYSIS:
            datasets = ["sales data", "user behavior data", "financial data", "sensor data"]
            dataset = random.choice(datasets)
            customized["description"] = customized["description"].replace("{dataset}", dataset)
            
        # Add difficulty-specific adjustments
        if difficulty == TaskDifficulty.ADVANCED:
            customized["estimated_duration"] = int(customized["estimated_duration"] * 1.5)
            customized["success_criteria"].append("Provide detailed analysis and optimization suggestions")
            
        return customized
    
    def _validate_task_quality(self, task: Task) -> bool:
        """Validate task meets quality standards"""
        # Check complexity score is within acceptable range
        if not (self.min_complexity_score <= task.complexity_score <= self.max_complexity_score):
            logger.warning(f"Task {task.task_id} complexity score {task.complexity_score} outside acceptable range")
            return False
        
        # Check task has sufficient success criteria
        if len(task.success_criteria) < 1:
            logger.warning(f"Task {task.task_id} lacks sufficient success criteria")
            return False
        
        # Check for task diversity in queue
        if not self._check_task_diversity(task):
            logger.warning(f"Task {task.task_id} doesn't meet diversity requirements")
            return False
        
        return True
    
    def _check_task_diversity(self, new_task: Task) -> bool:
        """Check if new task adds sufficient diversity to queue"""
        if len(self.task_queue.tasks) < 3:
            return True  # Always accept if queue is small
        
        # Check category diversity
        recent_categories = [t.category for t in self.task_queue.tasks[-3:]]
        if recent_categories.count(new_task.category) >= 2:
            return False
        
        # Check difficulty diversity
        recent_difficulties = [t.difficulty for t in self.task_queue.tasks[-3:]]
        if recent_difficulties.count(new_task.difficulty) >= 3:
            return False
        
        return True
    
    def _check_rate_limit(self) -> bool:
        """Check if task generation is within rate limits"""
        now = datetime.utcnow()
        
        # Reset counter if more than a minute has passed
        if now - self.last_generation_time > timedelta(minutes=1):
            self.generation_count = 0
            self.last_generation_time = now
        
        if self.generation_count >= self.generation_rate_limit:
            return False
        
        self.generation_count += 1
        return True
    
    def _select_progressive_difficulty(self) -> TaskDifficulty:
        """Select difficulty based on recent task completion patterns"""
        # Simple progressive difficulty selection
        # In a real implementation, this would analyze completion rates
        
        queue_status = self.task_queue.get_queue_status()
        difficulty_dist = queue_status.get("difficulty_distribution", {})
        
        # Balance difficulty distribution
        total_tasks = sum(difficulty_dist.values())
        if total_tasks == 0:
            return TaskDifficulty.BEGINNER
        
        beginner_ratio = difficulty_dist.get("beginner", 0) / total_tasks
        intermediate_ratio = difficulty_dist.get("intermediate", 0) / total_tasks
        advanced_ratio = difficulty_dist.get("advanced", 0) / total_tasks
        
        # Aim for 40% beginner, 40% intermediate, 20% advanced
        if beginner_ratio < 0.4:
            return TaskDifficulty.BEGINNER
        elif intermediate_ratio < 0.4:
            return TaskDifficulty.INTERMEDIATE
        else:
            return TaskDifficulty.ADVANCED
    
    def _parse_task_requirements(self, content: str) -> Dict[str, Any]:
        """Parse task requirements from user input"""
        requirements = {}
        content_lower = content.lower()
        
        # Parse difficulty
        if "beginner" in content_lower or "easy" in content_lower:
            requirements["difficulty"] = TaskDifficulty.BEGINNER
        elif "advanced" in content_lower or "hard" in content_lower or "difficult" in content_lower:
            requirements["difficulty"] = TaskDifficulty.ADVANCED
        elif "intermediate" in content_lower or "medium" in content_lower:
            requirements["difficulty"] = TaskDifficulty.INTERMEDIATE
        
        # Parse category
        for category in TaskCategory:
            if category.value in content_lower:
                requirements["category"] = category
                break
        
        # Parse priority
        if "urgent" in content_lower or "critical" in content_lower:
            requirements["priority"] = TaskPriority.CRITICAL
        elif "high priority" in content_lower:
            requirements["priority"] = TaskPriority.HIGH
        elif "low priority" in content_lower:
            requirements["priority"] = TaskPriority.LOW
        
        return requirements
    
    def _parse_feedback(self, content: str) -> Dict[str, Any]:
        """Parse feedback from task completion"""
        feedback = {}
        content_lower = content.lower()
        
        # Determine success
        success_indicators = ["completed", "finished", "success", "done", "solved"]
        failure_indicators = ["failed", "error", "couldn't", "unable", "stuck"]
        
        if any(indicator in content_lower for indicator in success_indicators):
            feedback["success"] = True
        elif any(indicator in content_lower for indicator in failure_indicators):
            feedback["success"] = False
        
        # Extract difficulty feedback
        if "too easy" in content_lower or "too simple" in content_lower:
            feedback["difficulty_feedback"] = "increase"
        elif "too hard" in content_lower or "too difficult" in content_lower:
            feedback["difficulty_feedback"] = "decrease"
        
        return feedback
    
    def _update_generation_strategy(self, feedback_data: Dict[str, Any]):
        """Update task generation strategy based on feedback"""
        # Adjust difficulty preferences based on feedback
        if feedback_data.get("difficulty_feedback") == "increase":
            # User wants harder tasks - adjust selection weights
            logger.info("Adjusting task generation to prefer higher difficulty")
        elif feedback_data.get("difficulty_feedback") == "decrease":
            # User wants easier tasks - adjust selection weights
            logger.info("Adjusting task generation to prefer lower difficulty")
    
    def _generate_progressive_task(self, feedback_data: Dict[str, Any]) -> Optional[Task]:
        """Generate a follow-up task based on completion feedback"""
        if not feedback_data.get("success", False):
            return None
        
        # Generate a slightly more challenging task in the same category
        # This is a simplified implementation
        requirements = {
            "difficulty": TaskDifficulty.INTERMEDIATE,  # Default progression
            "category": random.choice(list(TaskCategory))
        }
        
        return self._generate_autonomous_task(requirements)
    
    def _analyze_context(self, content: str, chat_history: Optional[List[BaseMessage]]) -> Dict[str, Any]:
        """Analyze input context to determine appropriate task type"""
        analysis = {
            "suggested_category": TaskCategory.PROBLEM_SOLVING,
            "suggested_difficulty": TaskDifficulty.INTERMEDIATE,
            "keywords": [],
            "context_type": "general"
        }
        
        content_lower = content.lower()
        
        # Analyze for category hints
        if any(word in content_lower for word in ["code", "program", "function", "algorithm"]):
            analysis["suggested_category"] = TaskCategory.CODING
        elif any(word in content_lower for word in ["analyze", "data", "statistics", "chart"]):
            analysis["suggested_category"] = TaskCategory.DATA_ANALYSIS
        elif any(word in content_lower for word in ["creative", "design", "write", "story"]):
            analysis["suggested_category"] = TaskCategory.CREATIVE
        elif any(word in content_lower for word in ["logic", "reasoning", "puzzle", "think"]):
            analysis["suggested_category"] = TaskCategory.REASONING
        
        return analysis
    
    def _create_task_from_context(self, context_analysis: Dict[str, Any]) -> Optional[Task]:
        """Create a task based on context analysis"""
        return self._generate_autonomous_task({
            "category": context_analysis["suggested_category"],
            "difficulty": context_analysis["suggested_difficulty"]
        })
    
    def _format_task_response(self, task: Task) -> str:
        """Format task as a response message"""
        response = f"# {task.title}\n\n"
        response += f"**Category:** {task.category.value.title()}\n"
        response += f"**Difficulty:** {task.difficulty.value.title()}\n"
        response += f"**Estimated Duration:** {task.estimated_duration} minutes\n"
        response += f"**Complexity Score:** {task.complexity_score}/10\n\n"
        
        response += f"## Description\n{task.description}\n\n"
        
        if task.prerequisites:
            response += f"## Prerequisites\n"
            for prereq in task.prerequisites:
                response += f"- {prereq}\n"
            response += "\n"
        
        response += f"## Success Criteria\n"
        for criteria in task.success_criteria:
            response += f"- {criteria}\n"
        response += "\n"
        
        if task.evaluation_metrics:
            response += f"## Evaluation Metrics\n"
            for metric, description in task.evaluation_metrics.items():
                response += f"- **{metric}:** {description}\n"
            response += "\n"
        
        if task.tags:
            response += f"**Tags:** {', '.join(task.tags)}\n\n"
        
        response += f"**Task ID:** {task.task_id}\n"
        
        return response
    
    def _initialize_task_templates(self) -> Dict[TaskCategory, Dict[TaskDifficulty, List[Dict[str, Any]]]]:
        """Initialize task templates for different categories and difficulties"""
        return {
            TaskCategory.CODING: {
                TaskDifficulty.BEGINNER: [
                    {
                        "title": "Simple {language} Function",
                        "description": "Write a {language} function that takes two numbers and returns their sum. Include proper error handling for invalid inputs.",
                        "success_criteria": [
                            "Function correctly adds two numbers",
                            "Handles invalid input gracefully",
                            "Includes basic documentation"
                        ],
                        "evaluation_metrics": {
                            "correctness": "Function produces correct output for all test cases",
                            "code_quality": "Code is readable and follows basic conventions"
                        },
                        "estimated_duration": 15,
                        "tags": ["function", "basic", "arithmetic"]
                    }
                ],
                TaskDifficulty.INTERMEDIATE: [
                    {
                        "title": "Data Structure Implementation",
                        "description": "Implement a basic data structure (stack, queue, or linked list) in {language} with all standard operations.",
                        "success_criteria": [
                            "All standard operations implemented correctly",
                            "Proper error handling for edge cases",
                            "Unit tests included",
                            "Time complexity documented"
                        ],
                        "evaluation_metrics": {
                            "correctness": "All operations work correctly",
                            "efficiency": "Operations have appropriate time complexity",
                            "testing": "Comprehensive test coverage"
                        },
                        "estimated_duration": 45,
                        "tags": ["data-structure", "algorithms", "testing"]
                    }
                ],
                TaskDifficulty.ADVANCED: [
                    {
                        "title": "Algorithm Optimization Challenge",
                        "description": "Optimize a given algorithm to improve its time or space complexity. Analyze the trade-offs and document your approach.",
                        "success_criteria": [
                            "Significant performance improvement achieved",
                            "Trade-offs clearly documented",
                            "Benchmarking results provided",
                            "Code is production-ready"
                        ],
                        "evaluation_metrics": {
                            "performance": "Measurable improvement in time/space complexity",
                            "analysis": "Thorough analysis of optimization approach",
                            "documentation": "Clear explanation of changes and trade-offs"
                        },
                        "estimated_duration": 90,
                        "prerequisites": ["Understanding of algorithm complexity", "Proficiency in chosen language"],
                        "tags": ["optimization", "algorithms", "performance"]
                    }
                ]
            },
            TaskCategory.DATA_ANALYSIS: {
                TaskDifficulty.BEGINNER: [
                    {
                        "title": "Basic Data Exploration",
                        "description": "Analyze a dataset of {dataset} to identify basic patterns and trends. Create simple visualizations and summary statistics.",
                        "success_criteria": [
                            "Dataset successfully loaded and explored",
                            "Basic statistics calculated",
                            "At least 3 visualizations created",
                            "Key findings summarized"
                        ],
                        "evaluation_metrics": {
                            "completeness": "All required analyses completed",
                            "visualization": "Charts are clear and informative",
                            "insights": "Meaningful patterns identified"
                        },
                        "estimated_duration": 30,
                        "tags": ["data", "visualization", "statistics"]
                    }
                ],
                TaskDifficulty.INTERMEDIATE: [
                    {
                        "title": "Predictive Model Development",
                        "description": "Build a predictive model using {dataset}. Compare multiple algorithms and evaluate their performance.",
                        "success_criteria": [
                            "Data properly preprocessed",
                            "Multiple models trained and compared",
                            "Model performance evaluated with appropriate metrics",
                            "Best model selected with justification"
                        ],
                        "evaluation_metrics": {
                            "methodology": "Sound approach to model development",
                            "performance": "Models achieve reasonable accuracy",
                            "comparison": "Fair comparison between different approaches"
                        },
                        "estimated_duration": 60,
                        "prerequisites": ["Basic machine learning knowledge"],
                        "tags": ["machine-learning", "prediction", "evaluation"]
                    }
                ],
                TaskDifficulty.ADVANCED: [
                    {
                        "title": "Advanced Analytics Pipeline",
                        "description": "Design and implement a complete analytics pipeline for {dataset} including data ingestion, processing, modeling, and deployment considerations.",
                        "success_criteria": [
                            "End-to-end pipeline implemented",
                            "Scalability considerations addressed",
                            "Model monitoring and updating strategy defined",
                            "Production deployment plan created"
                        ],
                        "evaluation_metrics": {
                            "architecture": "Pipeline design is scalable and maintainable",
                            "implementation": "Code is production-ready",
                            "strategy": "Comprehensive deployment and monitoring plan"
                        },
                        "estimated_duration": 120,
                        "prerequisites": ["Advanced ML knowledge", "Pipeline development experience"],
                        "tags": ["pipeline", "production", "scalability"]
                    }
                ]
            },
            TaskCategory.REASONING: {
                TaskDifficulty.BEGINNER: [
                    {
                        "title": "Logic Puzzle Solving",
                        "description": "Solve a series of logic puzzles that require deductive reasoning and pattern recognition.",
                        "success_criteria": [
                            "All puzzles solved correctly",
                            "Solution approach explained",
                            "Reasoning steps documented"
                        ],
                        "evaluation_metrics": {
                            "accuracy": "Correct solutions provided",
                            "reasoning": "Clear explanation of logical steps"
                        },
                        "estimated_duration": 25,
                        "tags": ["logic", "puzzles", "deduction"]
                    }
                ],
                TaskDifficulty.INTERMEDIATE: [
                    {
                        "title": "Strategic Problem Analysis",
                        "description": "Analyze a complex business scenario and develop strategic recommendations using structured reasoning frameworks.",
                        "success_criteria": [
                            "Problem clearly defined and analyzed",
                            "Multiple solution approaches considered",
                            "Recommendations supported by logical reasoning",
                            "Potential risks and benefits identified"
                        ],
                        "evaluation_metrics": {
                            "analysis": "Thorough problem analysis",
                            "reasoning": "Sound logical framework applied",
                            "recommendations": "Practical and well-justified solutions"
                        },
                        "estimated_duration": 50,
                        "tags": ["strategy", "analysis", "business"]
                    }
                ]
            },
            TaskCategory.CREATIVE: {
                TaskDifficulty.BEGINNER: [
                    {
                        "title": "Creative Writing Exercise",
                        "description": "Write a short story or creative piece based on a given prompt. Focus on originality and engaging narrative.",
                        "success_criteria": [
                            "Story follows the given prompt",
                            "Demonstrates creativity and originality",
                            "Has clear beginning, middle, and end",
                            "Engaging and well-written"
                        ],
                        "evaluation_metrics": {
                            "creativity": "Original and imaginative content",
                            "structure": "Well-organized narrative",
                            "engagement": "Compelling and interesting to read"
                        },
                        "estimated_duration": 35,
                        "tags": ["writing", "creativity", "storytelling"]
                    }
                ]
            },
            TaskCategory.PROBLEM_SOLVING: {
                TaskDifficulty.BEGINNER: [
                    {
                        "title": "Process Optimization",
                        "description": "Identify inefficiencies in a given process and propose improvements. Document your analysis and recommendations.",
                        "success_criteria": [
                            "Current process thoroughly analyzed",
                            "Inefficiencies clearly identified",
                            "Practical improvements proposed",
                            "Implementation plan provided"
                        ],
                        "evaluation_metrics": {
                            "analysis": "Comprehensive process analysis",
                            "solutions": "Practical and implementable improvements",
                            "impact": "Clear benefits and expected outcomes"
                        },
                        "estimated_duration": 40,
                        "tags": ["optimization", "process", "improvement"]
                    }
                ]
            }
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current task queue status"""
        return self.task_queue.get_queue_status()
    
    def get_next_task(self, difficulty_filter: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get next task from queue"""
        difficulty = None
        if difficulty_filter:
            try:
                difficulty = TaskDifficulty(difficulty_filter.lower())
            except ValueError:
                pass
        
        task = self.task_queue.get_next_task(difficulty)
        return task.to_dict() if task else None
    
    def add_custom_task(self, task_data: Dict[str, Any]) -> bool:
        """Add a custom task to the queue"""
        try:
            task = Task(
                task_id=task_data.get("task_id", f"custom_{int(time.time())}"),
                title=task_data["title"],
                description=task_data["description"],
                category=TaskCategory(task_data["category"]),
                difficulty=TaskDifficulty(task_data["difficulty"]),
                priority=TaskPriority(task_data.get("priority", "medium")),
                success_criteria=task_data.get("success_criteria", []),
                evaluation_metrics=task_data.get("evaluation_metrics", {}),
                estimated_duration=task_data.get("estimated_duration", 30),
                prerequisites=task_data.get("prerequisites", []),
                tags=task_data.get("tags", [])
            )
            
            if self._validate_task_quality(task):
                return self.task_queue.add_task(task)
            return False
            
        except Exception as e:
            logger.error(f"Failed to add custom task: {str(e)}")
            return False
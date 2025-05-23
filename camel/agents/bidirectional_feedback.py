"""
CAMEL Extensions - Bidirectional Feedback System

This module implements a comprehensive bidirectional feedback system where all agents
can evaluate each other's performance, creating a complete feedback loop for continuous
improvement and learning.

Key Features:
- Cross-agent evaluation (Proposer ↔ Executor ↔ PeerReviewer)
- Performance tracking and metrics
- Feedback aggregation and analysis
- Integration with DPO training pipeline
- Adaptive learning based on feedback patterns

The system enables:
1. Proposer evaluates Executor's task execution quality
2. Executor evaluates Proposer's task clarity and feasibility
3. PeerReviewer evaluates both and receives evaluation from both
4. All feedback is aggregated for training and improvement
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import json
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

# CAMEL library imports
from camel.agents import BaseAgent
from camel.messages import BaseMessage, AssistantMessage, UserMessage, SystemMessage
from camel.types import AgentType


class FeedbackType(Enum):
    """Types of feedback that can be provided"""
    TASK_QUALITY = "task_quality"
    EXECUTION_QUALITY = "execution_quality"
    REVIEW_QUALITY = "review_quality"
    COMMUNICATION_CLARITY = "communication_clarity"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVITY = "creativity"
    EFFICIENCY = "efficiency"


class AgentRole(Enum):
    """Agent roles in the feedback system"""
    PROPOSER = "Proposer"
    EXECUTOR = "Executor"
    PEER_REVIEWER = "PeerReviewer"


@dataclass
class FeedbackEntry:
    """Structured feedback entry"""
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
    confidence_score: float  # How confident the evaluator is in their assessment


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an agent"""
    agent_name: str
    total_evaluations_received: int
    average_overall_rating: float
    average_scores_by_type: Dict[str, float]
    improvement_trend: float  # Positive = improving, negative = declining
    feedback_given_count: int
    feedback_quality_score: float  # How good this agent is at giving feedback
    last_updated: datetime


class BidirectionalFeedbackManager:
    """
    Manages bidirectional feedback between all agents in the system
    """
    
    def __init__(self):
        self.feedback_history: List[FeedbackEntry] = []
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.feedback_templates = self._initialize_feedback_templates()
        
    def _initialize_feedback_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize feedback templates for different agent combinations"""
        return {
            "Proposer_to_Executor": {
                "prompt": """
                Evaluate the Executor's performance on the task you proposed.
                
                Original Task: {task_description}
                Executor's Response: {executor_response}
                
                Rate the following aspects (1-10):
                1. Task Understanding: How well did the Executor understand your task?
                2. Solution Quality: How good was the solution provided?
                3. Communication: How clear and well-structured was the response?
                4. Completeness: Did the Executor address all aspects of the task?
                5. Efficiency: Was the solution efficient and well-optimized?
                
                Provide specific feedback on what the Executor did well and what could be improved.
                """,
                "focus_areas": ["task_understanding", "solution_quality", "communication", "completeness", "efficiency"]
            },
            "Executor_to_Proposer": {
                "prompt": """
                Evaluate the Proposer's task that you just executed.
                
                Task Received: {task_description}
                Your Execution: {executor_response}
                
                Rate the following aspects (1-10):
                1. Task Clarity: How clear and well-defined was the task?
                2. Feasibility: Was the task realistic and achievable?
                3. Scope: Was the task scope appropriate (not too broad/narrow)?
                4. Context: Was sufficient context provided?
                5. Success Criteria: Were the success criteria clear and measurable?
                
                Provide specific feedback on how the Proposer could improve task formulation.
                """,
                "focus_areas": ["task_clarity", "feasibility", "scope", "context", "success_criteria"]
            },
            "PeerReviewer_to_Proposer": {
                "prompt": """
                Evaluate the Proposer's task generation and formulation.
                
                Task Generated: {task_description}
                Context: {context}
                
                Rate the following aspects (1-10):
                1. Task Quality: Overall quality of the task design
                2. Creativity: How creative and engaging is the task?
                3. Educational Value: How much learning potential does the task have?
                4. Difficulty Appropriateness: Is the difficulty level appropriate?
                5. Clarity: How clear are the instructions and requirements?
                
                Provide feedback on the task generation process and suggestions for improvement.
                """,
                "focus_areas": ["task_quality", "creativity", "educational_value", "difficulty", "clarity"]
            },
            "PeerReviewer_to_Executor": {
                "prompt": """
                Evaluate the Executor's task execution and solution.
                
                Original Task: {task_description}
                Executor's Solution: {executor_response}
                
                Rate the following aspects (1-10):
                1. Correctness: Is the solution correct and accurate?
                2. Methodology: Was the approach/methodology sound?
                3. Explanation Quality: How well was the solution explained?
                4. Code Quality: If applicable, how good is the code quality?
                5. Innovation: Did the solution show creative problem-solving?
                
                Provide detailed feedback on the execution quality and areas for improvement.
                """,
                "focus_areas": ["correctness", "methodology", "explanation", "code_quality", "innovation"]
            },
            "Proposer_to_PeerReviewer": {
                "prompt": """
                Evaluate the PeerReviewer's feedback on your task.
                
                Your Task: {task_description}
                PeerReviewer's Feedback: {reviewer_feedback}
                
                Rate the following aspects (1-10):
                1. Feedback Accuracy: How accurate was the review?
                2. Constructiveness: How constructive and helpful was the feedback?
                3. Specificity: How specific and actionable were the suggestions?
                4. Fairness: Was the evaluation fair and unbiased?
                5. Comprehensiveness: Did the review cover all important aspects?
                
                Provide feedback on the quality of the peer review process.
                """,
                "focus_areas": ["accuracy", "constructiveness", "specificity", "fairness", "comprehensiveness"]
            },
            "Executor_to_PeerReviewer": {
                "prompt": """
                Evaluate the PeerReviewer's assessment of your execution.
                
                Your Execution: {executor_response}
                PeerReviewer's Assessment: {reviewer_feedback}
                
                Rate the following aspects (1-10):
                1. Assessment Accuracy: How accurate was the peer review?
                2. Insight Quality: Did the review provide valuable insights?
                3. Balance: Was the feedback balanced (strengths and improvements)?
                4. Technical Understanding: Did the reviewer understand the technical aspects?
                5. Actionability: How actionable were the improvement suggestions?
                
                Provide feedback on how the peer review process could be improved.
                """,
                "focus_areas": ["accuracy", "insight", "balance", "technical_understanding", "actionability"]
            }
        }
    
    async def collect_feedback(
        self,
        evaluator_agent: BaseAgent,
        evaluated_agent_name: str,
        interaction_context: Dict[str, Any],
        feedback_type: FeedbackType = FeedbackType.TASK_QUALITY
    ) -> Optional[FeedbackEntry]:
        """
        Collect feedback from one agent about another agent's performance
        
        Args:
            evaluator_agent: The agent providing the feedback
            evaluated_agent_name: Name of the agent being evaluated
            interaction_context: Context of the interaction being evaluated
            feedback_type: Type of feedback being collected
            
        Returns:
            FeedbackEntry if successful, None otherwise
        """
        try:
            # Get the appropriate feedback template
            template_key = f"{evaluator_agent.system_message.role_name}_to_{evaluated_agent_name}"
            template = self.feedback_templates.get(template_key)
            
            if not template:
                logger.warning(f"No feedback template found for {template_key}")
                return None
            
            # Build the feedback prompt
            feedback_prompt = self._build_feedback_prompt(template, interaction_context)
            
            # Get feedback from the evaluator agent
            feedback_message = UserMessage(role_name="user", content=feedback_prompt)
            response = evaluator_agent.step(feedback_message)
            
            if not response:
                logger.error("No response received from evaluator agent")
                return None
            
            # Parse the structured feedback
            parsed_feedback = self._parse_feedback_response(response[0].content)
            
            if not parsed_feedback:
                logger.error("Failed to parse feedback response")
                return None
            
            # Create feedback entry
            feedback_entry = FeedbackEntry(
                feedback_id=f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(parsed_feedback))}",
                evaluator_agent=evaluator_agent.system_message.role_name,
                evaluated_agent=evaluated_agent_name,
                feedback_type=feedback_type,
                overall_rating=parsed_feedback.get("overall_rating", 0),
                specific_scores=parsed_feedback.get("specific_scores", {}),
                strengths=parsed_feedback.get("strengths", []),
                areas_for_improvement=parsed_feedback.get("areas_for_improvement", []),
                detailed_feedback=parsed_feedback.get("detailed_feedback", ""),
                context=interaction_context,
                timestamp=datetime.now(),
                confidence_score=parsed_feedback.get("confidence_score", 0.8)
            )
            
            # Store the feedback
            self.feedback_history.append(feedback_entry)
            
            # Update agent metrics
            self._update_agent_metrics(feedback_entry)
            
            logger.info(f"Collected feedback from {evaluator_agent.system_message.role_name} about {evaluated_agent_name}")
            return feedback_entry
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {str(e)}")
            return None
    
    def _build_feedback_prompt(self, template: Dict[str, str], context: Dict[str, Any]) -> str:
        """Build a feedback prompt using the template and context"""
        try:
            prompt = template["prompt"].format(**context)
            
            # Add structured response format
            prompt += """
            
Please provide your evaluation in the following JSON format:
```json
{
    "overall_rating": <score 1-10>,
    "specific_scores": {
        "aspect1": <score>,
        "aspect2": <score>,
        ...
    },
    "strengths": ["strength1", "strength2", ...],
    "areas_for_improvement": ["improvement1", "improvement2", ...],
    "detailed_feedback": "<detailed text feedback>",
    "confidence_score": <0.0-1.0 how confident you are in this assessment>
}
```
"""
            return prompt
            
        except KeyError as e:
            logger.error(f"Missing context key for feedback template: {e}")
            return template["prompt"]
    
    def _parse_feedback_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """Parse structured feedback from agent response"""
        try:
            # Extract JSON from response
            json_content = response_content
            if "```json" in response_content:
                json_content = response_content.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in response_content:
                json_content = response_content.split("```", 1)[1].split("```", 1)[0]
            
            parsed = json.loads(json_content.strip())
            
            # Validate required fields
            required_fields = ["overall_rating", "specific_scores", "strengths", "areas_for_improvement", "detailed_feedback"]
            for field in required_fields:
                if field not in parsed:
                    logger.warning(f"Missing required field in feedback: {field}")
                    return None
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse feedback response: {str(e)}")
            return None
    
    def _update_agent_metrics(self, feedback: FeedbackEntry) -> None:
        """Update performance metrics for the evaluated agent"""
        agent_name = feedback.evaluated_agent
        
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentPerformanceMetrics(
                agent_name=agent_name,
                total_evaluations_received=0,
                average_overall_rating=0.0,
                average_scores_by_type={},
                improvement_trend=0.0,
                feedback_given_count=0,
                feedback_quality_score=0.0,
                last_updated=datetime.now()
            )
        
        metrics = self.agent_metrics[agent_name]
        
        # Update evaluation count and average rating
        old_total = metrics.total_evaluations_received
        new_total = old_total + 1
        
        # Calculate new average rating
        old_avg = metrics.average_overall_rating
        new_avg = (old_avg * old_total + feedback.overall_rating) / new_total
        
        metrics.total_evaluations_received = new_total
        metrics.average_overall_rating = new_avg
        
        # Update scores by type
        feedback_type_str = feedback.feedback_type.value
        if feedback_type_str not in metrics.average_scores_by_type:
            metrics.average_scores_by_type[feedback_type_str] = feedback.overall_rating
        else:
            # Simple moving average for now
            old_score = metrics.average_scores_by_type[feedback_type_str]
            metrics.average_scores_by_type[feedback_type_str] = (old_score + feedback.overall_rating) / 2
        
        # Calculate improvement trend (simplified)
        recent_feedback = [f for f in self.feedback_history[-10:] if f.evaluated_agent == agent_name]
        if len(recent_feedback) >= 2:
            recent_scores = [f.overall_rating for f in recent_feedback]
            metrics.improvement_trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        metrics.last_updated = datetime.now()
        
        # Also update metrics for the evaluator (feedback given count)
        evaluator_name = feedback.evaluator_agent
        if evaluator_name not in self.agent_metrics:
            self.agent_metrics[evaluator_name] = AgentPerformanceMetrics(
                agent_name=evaluator_name,
                total_evaluations_received=0,
                average_overall_rating=0.0,
                average_scores_by_type={},
                improvement_trend=0.0,
                feedback_given_count=0,
                feedback_quality_score=0.0,
                last_updated=datetime.now()
            )
        
        self.agent_metrics[evaluator_name].feedback_given_count += 1
    
    def get_agent_performance_summary(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get performance summary for a specific agent"""
        if agent_name not in self.agent_metrics:
            return None
        
        metrics = self.agent_metrics[agent_name]
        
        # Get recent feedback
        recent_feedback = [f for f in self.feedback_history[-20:] if f.evaluated_agent == agent_name]
        
        return {
            "agent_name": agent_name,
            "overall_performance": {
                "average_rating": round(metrics.average_overall_rating, 2),
                "total_evaluations": metrics.total_evaluations_received,
                "improvement_trend": round(metrics.improvement_trend, 2),
                "trend_description": self._get_trend_description(metrics.improvement_trend)
            },
            "performance_by_type": metrics.average_scores_by_type,
            "feedback_activity": {
                "feedback_given": metrics.feedback_given_count,
                "feedback_quality": round(metrics.feedback_quality_score, 2)
            },
            "recent_feedback": [
                {
                    "from": f.evaluator_agent,
                    "rating": f.overall_rating,
                    "type": f.feedback_type.value,
                    "timestamp": f.timestamp.isoformat(),
                    "key_strengths": f.strengths[:2],
                    "key_improvements": f.areas_for_improvement[:2]
                }
                for f in recent_feedback[-5:]
            ],
            "last_updated": metrics.last_updated.isoformat()
        }
    
    def _get_trend_description(self, trend: float) -> str:
        """Get human-readable description of performance trend"""
        if trend > 0.5:
            return "Significantly improving"
        elif trend > 0.1:
            return "Improving"
        elif trend > -0.1:
            return "Stable"
        elif trend > -0.5:
            return "Declining"
        else:
            return "Significantly declining"
    
    def get_feedback_insights(self) -> Dict[str, Any]:
        """Get insights from all collected feedback"""
        if not self.feedback_history:
            return {"message": "No feedback data available"}
        
        # Calculate overall system metrics
        total_feedback = len(self.feedback_history)
        avg_rating = sum(f.overall_rating for f in self.feedback_history) / total_feedback
        
        # Feedback by agent pairs
        agent_pairs = {}
        for feedback in self.feedback_history:
            pair = f"{feedback.evaluator_agent} → {feedback.evaluated_agent}"
            if pair not in agent_pairs:
                agent_pairs[pair] = []
            agent_pairs[pair].append(feedback.overall_rating)
        
        # Most common strengths and improvements
        all_strengths = []
        all_improvements = []
        for feedback in self.feedback_history:
            all_strengths.extend(feedback.strengths)
            all_improvements.extend(feedback.areas_for_improvement)
        
        # Count occurrences
        strength_counts = {}
        improvement_counts = {}
        
        for strength in all_strengths:
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        for improvement in all_improvements:
            improvement_counts[improvement] = improvement_counts.get(improvement, 0) + 1
        
        return {
            "system_overview": {
                "total_feedback_entries": total_feedback,
                "average_rating": round(avg_rating, 2),
                "active_agents": len(self.agent_metrics),
                "feedback_coverage": len(agent_pairs)
            },
            "agent_pair_performance": {
                pair: {
                    "count": len(ratings),
                    "average": round(sum(ratings) / len(ratings), 2)
                }
                for pair, ratings in agent_pairs.items()
            },
            "common_strengths": sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "common_improvements": sorted(improvement_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "agent_rankings": sorted(
                [
                    {
                        "agent": name,
                        "rating": metrics.average_overall_rating,
                        "evaluations": metrics.total_evaluations_received
                    }
                    for name, metrics in self.agent_metrics.items()
                ],
                key=lambda x: x["rating"],
                reverse=True
            )
        }
    
    def export_feedback_for_dpo(self, min_rating_threshold: float = 6.0) -> List[Dict[str, Any]]:
        """Export feedback data in format suitable for DPO training"""
        dpo_data = []
        
        for feedback in self.feedback_history:
            if feedback.overall_rating >= min_rating_threshold:
                dpo_entry = {
                    "id": feedback.feedback_id,
                    "context": feedback.context,
                    "evaluator": feedback.evaluator_agent,
                    "evaluated": feedback.evaluated_agent,
                    "feedback_type": feedback.feedback_type.value,
                    "rating": feedback.overall_rating,
                    "scores": feedback.specific_scores,
                    "strengths": feedback.strengths,
                    "improvements": feedback.areas_for_improvement,
                    "detailed_feedback": feedback.detailed_feedback,
                    "confidence": feedback.confidence_score,
                    "timestamp": feedback.timestamp.isoformat()
                }
                dpo_data.append(dpo_entry)
        
        return dpo_data


class BidirectionalFeedbackAgent(BaseAgent):
    """
    Agent wrapper that enables bidirectional feedback capabilities
    """
    
    def __init__(self, base_agent: BaseAgent, feedback_manager: BidirectionalFeedbackManager):
        super().__init__(system_message=base_agent.system_message)
        self.base_agent = base_agent
        self.feedback_manager = feedback_manager
        self.agent_type = base_agent.agent_type
    
    def step(self, message: BaseMessage, chat_history: Optional[List[BaseMessage]] = None) -> List[BaseMessage]:
        """Enhanced step method that includes feedback collection"""
        # Execute the base agent's step
        response = self.base_agent.step(message, chat_history)
        
        # Store interaction context for potential feedback
        self._last_interaction = {
            "input_message": message.content,
            "response": response[0].content if response else "",
            "timestamp": datetime.now(),
            "chat_history": chat_history
        }
        
        return response
    
    async def provide_feedback_on(
        self,
        other_agent_name: str,
        interaction_context: Dict[str, Any],
        feedback_type: FeedbackType = FeedbackType.TASK_QUALITY
    ) -> Optional[FeedbackEntry]:
        """Provide feedback on another agent's performance"""
        return await self.feedback_manager.collect_feedback(
            evaluator_agent=self,
            evaluated_agent_name=other_agent_name,
            interaction_context=interaction_context,
            feedback_type=feedback_type
        )
    
    def get_performance_summary(self) -> Optional[Dict[str, Any]]:
        """Get this agent's performance summary"""
        return self.feedback_manager.get_agent_performance_summary(
            self.system_message.role_name
        )
"""
CAMEL Extensions - Peer Reviewer Agent

This module implements the PeerReviewer agent, which evaluates the quality of the
Proposer agent's suggestions and the Executor agent's executions, providing
structured feedback for DPO training.

The PeerReviewer serves as a critical component in the CAMEL workflow by:
1. Analyzing the quality of interactions between agents
2. Providing structured feedback with quantitative ratings
3. Highlighting strengths and areas for improvement
4. Generating feedback in a format suitable for DPO training

Integration with DPO Training:
------------------------------
The PeerReviewer generates feedback that can be directly used for training models
through Direct Preference Optimization (DPO). This feedback helps create paired
examples of preferred and non-preferred responses based on quality assessments.

The review process follows these steps:
1. Extract the Proposer suggestion and Executor execution from the interaction
2. Generate a detailed evaluation with numeric scores and specific feedback
3. Structure the feedback in both human-readable and machine-readable formats
4. Provide annotations that can be used directly for DPO training

Sample feedback structure:
```json
{
  "overall_rating": 8,
  "strengths": ["Clear instructions", "Efficient solution"],
  "areas_for_improvement": ["Could provide more context"],
  "effectiveness_score": 7,
  "correctness_score": 9,
  "detailed_feedback": "The suggestion was well-structured but..."
}
```
"""

from typing import Dict, List, Optional, Any, Tuple
import json
from loguru import logger

# CAMEL library imports
from camel.agents import BaseAgent
from camel.messages import BaseMessage, AssistantMessage, UserMessage, SystemMessage
from camel.types import AgentType


class PeerReviewer(BaseAgent):
    """
    PeerReviewer Agent that evaluates the quality of other agents' outputs
    and provides structured feedback for DPO training.
    """
    
    def __init__(
        self,
        system_message: SystemMessage,
        **kwargs
    ):
        """
        Initialize the PeerReviewer agent.
        
        Args:
            system_message: SystemMessage defining the reviewer's role and behavior
            **kwargs: Additional arguments passed to the BaseAgent
        """
        super().__init__(system_message=system_message, **kwargs)
        self.agent_type = AgentType.ASSISTANT
    
    def step(self, message: BaseMessage, chat_history: Optional[List[BaseMessage]] = None) -> List[BaseMessage]:
        """
        Process the input message and generate a review response.
        
        Args:
            message: Input message to review (typically from Executor agent)
            chat_history: Optional chat history to provide context
            
        Returns:
            List of message(s) containing the evaluation and feedback
        """
        # Full chat history provides context on the interaction
        full_history = chat_history if chat_history else []
        
        # Extract proposer's suggestion and executor's execution
        proposer_suggestion, executor_execution = self._extract_proposer_executor_interaction(full_history, message)
        
        # Build review prompt
        review_prompt = self._build_review_prompt(proposer_suggestion, executor_execution, message)
        
        # Get the evaluation from the model
        review_message = self._get_response(review_prompt, full_history)
        
        # Parse structured feedback if possible
        parsed_feedback = self._parse_structured_feedback(review_message.content)
        
        # If parsing was successful, format the response
        if parsed_feedback:
            formatted_content = self._format_structured_feedback(parsed_feedback)
            return [AssistantMessage(role_name="PeerReviewer", content=formatted_content)]
        
        # If parsing failed, return the original model response
        return [review_message]
    
    def _extract_proposer_executor_interaction(
        self, 
        history: List[BaseMessage], 
        current_message: BaseMessage
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Extract the most recent proposer suggestion and executor execution from history.
        
        Args:
            history: Chat history
            current_message: Current message being processed
            
        Returns:
            Tuple of (proposer_suggestion, executor_execution)
        """
        proposer_suggestion = None
        executor_execution = None
        
        # Look for the most recent proposer message in history
        for msg in reversed(history):
            if msg.role_name == "Proposer" and msg.role_type == "assistant":
                proposer_suggestion = {
                    "content": msg.content,
                    "role": msg.role_name
                }
                break
        
        # Consider the current message as the executor's execution
        if current_message.role_name == "Executor":
            executor_execution = {
                "content": current_message.content,
                "role": current_message.role_name
            }
        # If the current message isn't from Executor, check history
        else:
            for msg in reversed(history):
                if msg.role_name == "Executor" and msg.role_type == "assistant":
                    executor_execution = {
                        "content": msg.content,
                        "role": msg.role_name
                    }
                    break
        
        return proposer_suggestion, executor_execution
    
    def _build_review_prompt(
        self, 
        proposer_suggestion: Optional[Dict], 
        executor_execution: Optional[Dict],
        current_message: BaseMessage
    ) -> str:
        """
        Build a prompt for the reviewer to evaluate the interaction.
        
        Args:
            proposer_suggestion: The suggestion from the Proposer agent
            executor_execution: The execution from the Executor agent
            current_message: Current message being processed
            
        Returns:
            Review prompt string
        """
        prompt = "You are a peer reviewer evaluating an AI interaction. "
        prompt += "Provide a structured evaluation of the quality and effectiveness "
        prompt += "of the following agent interactions.\n\n"
        
        # Include proposer suggestion if available
        if proposer_suggestion:
            prompt += f"## Proposer Agent Suggestion\n\n{proposer_suggestion['content']}\n\n"
        else:
            prompt += "## Proposer Agent Suggestion\n\nNo suggestion available\n\n"
        
        # Include executor execution if available
        if executor_execution:
            prompt += f"## Executor Agent Execution\n\n{executor_execution['content']}\n\n"
        else:
            prompt += f"## Current Message\n\n{current_message.content}\n\n"
        
        # Add evaluation guidelines
        prompt += """## Evaluation Instructions
        
Please provide a structured evaluation with the following sections:
1. Overall Rating: Numeric score from 1-10 where 10 is excellent
2. Strengths: Key strengths of the interaction
3. Areas for Improvement: Suggestions for how the interaction could be improved
4. Effectiveness: How effective was the Proposer's suggestion in addressing the task
5. Correctness: Evaluate the correctness of both the suggestion and execution
6. Detailed Feedback: Specific feedback on both agents' performance

Format your response as a JSON object with these keys:
```json
{
  "overall_rating": <score>,
  "strengths": ["<strength1>", "<strength2>", ...],
  "areas_for_improvement": ["<improvement1>", "<improvement2>", ...],
  "effectiveness_score": <score>,
  "correctness_score": <score>,
  "detailed_feedback": "<text>"
}
```
"""
        return prompt
    
    def _parse_structured_feedback(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse structured feedback from the model output.
        
        Args:
            content: Model output content
            
        Returns:
            Parsed feedback dictionary or None if parsing failed
        """
        try:
            # Look for JSON content between triple backticks
            json_content = content
            if "```json" in content and "```" in content.split("```json", 1)[1]:
                json_content = content.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in content and "```" in content.split("```", 1)[1]:
                json_content = content.split("```", 1)[1].split("```", 1)[0]
            
            # Try to parse the JSON
            parsed = json.loads(json_content)
            
            # Validate required fields
            required_fields = [
                "overall_rating", "strengths", "areas_for_improvement", 
                "effectiveness_score", "correctness_score", "detailed_feedback"
            ]
            
            for field in required_fields:
                if field not in parsed:
                    logger.warning(f"Missing required field in review feedback: {field}")
                    return None
                    
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse structured feedback: {str(e)}")
            return None
    
    def _format_structured_feedback(self, feedback: Dict[str, Any]) -> str:
        """
        Format structured feedback into a readable message.
        
        Args:
            feedback: Parsed feedback dictionary
            
        Returns:
            Formatted feedback string
        """
        formatted = f"# Peer Review Evaluation\n\n"
        formatted += f"## Overall Rating: {feedback['overall_rating']}/10\n\n"
        
        formatted += "## Strengths\n"
        for strength in feedback["strengths"]:
            formatted += f"- {strength}\n"
        formatted += "\n"
        
        formatted += "## Areas for Improvement\n"
        for improvement in feedback["areas_for_improvement"]:
            formatted += f"- {improvement}\n"
        formatted += "\n"
        
        formatted += f"## Effectiveness Score: {feedback['effectiveness_score']}/10\n"
        formatted += f"## Correctness Score: {feedback['correctness_score']}/10\n\n"
        
        formatted += "## Detailed Feedback\n"
        formatted += feedback["detailed_feedback"]
        
        # Add machine-readable version at the end for DPO training
        formatted += "\n\n```json\n"
        formatted += json.dumps(feedback, indent=2)
        formatted += "\n```"
        
        return formatted
    
    def get_dpo_annotations(
        self,
        proposer_suggestion: Dict,
        executor_execution: Dict,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format the peer review feedback into a structure suitable for DPO training.
        
        Args:
            proposer_suggestion: The suggestion from the Proposer agent
            executor_execution: The execution from the Executor agent
            feedback: Parsed feedback dictionary
            
        Returns:
            Dictionary formatted for DPO training with context, chosen, and rejected
        """
        # Determine if the proposer suggestion was good or needs improvement
        is_good_suggestion = feedback.get("effectiveness_score", 0) >= 7
        
        # Get the original task context from history if available
        task_context = "Task context not available"
        if hasattr(self, '_chat_history') and self._chat_history:
            for msg in self._chat_history:
                if msg.role_type == "user" and msg.role_name == "user":
                    task_context = msg.content
                    break
        
        # Format the annotation for DPO
        annotation = {
            "id": f"peer_review_{hash(str(feedback))}",
            "dpo_context": task_context,
            "proposer_suggestion": proposer_suggestion.get("content", ""),
            "executor_execution": executor_execution.get("content", ""),
            "feedback": feedback,
            # The higher-rated response should be "chosen" and lower-rated should be "rejected"
            # Here we're just providing an example - in a real implementation you'd
            # compare multiple responses or create synthetic improvements
            "chosen": proposer_suggestion.get("content", "") if is_good_suggestion else "Improved version would go here",
            "rejected": "Lower quality version would go here" if is_good_suggestion else proposer_suggestion.get("content", ""),
            "scores": {
                "overall": feedback.get("overall_rating", 0),
                "effectiveness": feedback.get("effectiveness_score", 0),
                "correctness": feedback.get("correctness_score", 0)
            },
            "timestamp": None  # Would be filled in when actually used
        }
        
        return annotation
    
    def update_system_message(self, new_content: str) -> None:
        """
        Update the system message to adjust the reviewer's behavior.
        
        Args:
            new_content: New system message content
        """
        self.system_message = SystemMessage(role_name="PeerReviewer", content=new_content)
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from camel.agents.base import BaseAgent
from camel.models import ModelFactory, BaseModelBackend
from camel.types import ModelPlatformType, ModelType
from camel.messages import OpenAIMessage


class ReviewRating(BaseModel):
    """Rating model for specific aspects of a proposal"""
    score: float = Field(..., ge=0.0, le=10.0, description="Rating score from 0 to 10")
    justification: str = Field(..., description="Justification for the rating")


class StructuredReview(BaseModel):
    """Structured review output model for fine-tunable peer review agent"""
    relevance: ReviewRating = Field(
        ..., 
        description="How relevant the proposal is to the task"
    )
    correctness: ReviewRating = Field(
        ...,
        description="How technically correct the proposal is"
    )
    safety: ReviewRating = Field(
        ...,
        description="How safe and appropriate the proposal is"
    )
    clarity: ReviewRating = Field(
        ..., 
        description="How clear and understandable the proposal is"
    )
    overall_score: float = Field(
        ..., 
        ge=0.0, 
        le=10.0, 
        description="Overall score from 0 to 10"
    )
    summary_critique: str = Field(
        ...,
        description="Summary critique of the proposal"
    )
    improvement_suggestions: List[str] = Field(
        ...,
        description="Specific suggestions for improvement"
    )


class PeerReviewer(BaseAgent):
    """
    LLM-based peer reviewer agent that evaluates proposals and generates structured feedback.
    This agent is designed to be fine-tunable for Direct Preference Optimization (DPO).
    """

    def __init__(
        self, 
        name: str = 'peer_reviewer',
        model_platform: Union[ModelPlatformType, str] = ModelPlatformType.OPENAI,
        model_name: Union[ModelType, str] = 'gpt-4o',
        api_key: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the PeerReviewer agent with an LLM.

        Args:
            name: The name of the agent
            model_platform: The platform of the LLM (e.g., OPENAI, ANTHROPIC)
            model_name: The specific model to use
            api_key: Optional API key for the LLM service
            model_config: Optional configuration dictionary for the model
        """
        super().__init__(name=name)
        
        # Default configuration if none provided
        if model_config is None:
            model_config = {
                "temperature": 0.2,  # Lower temperature for more consistent evaluations
                "max_tokens": 2048,   # Sufficient for detailed reviews
                "top_p": 0.95        # Slightly constrained sampling
            }
        
        # Initialize the LLM for review
        self.model = ModelFactory.create(
            model_platform=model_platform,
            model_type=model_name,
            model_config_dict=model_config,
            api_key=api_key
        )

    def reset(self, *args, **kwargs) -> None:
        """Reset the agent's state"""
        return None
        
    def step(self, *args, **kwargs) -> Any:
        """Perform a reasoning step - not needed for this reviewer agent"""
        return None

    async def review(
        self, 
        transcript_a: str, 
        transcript_b: str
    ) -> Dict[str, Any]:
        """
        Review and compare two transcripts, generating structured feedback.
        Designed with fine-tunable input-output patterns for DPO training.

        Args:
            transcript_a: The first transcript to review
            transcript_b: The second transcript to review (for comparison)

        Returns:
            Dict containing structured review data
        """
        # Create the prompt for the review
        system_message = {
            "role": "system", 
            "content": """You are a peer reviewer tasked with evaluating proposed solutions.
Provide a detailed, structured assessment of the given transcripts based on relevance, correctness,
safety, and clarity. Your evaluation should be fair, thorough, and constructive.

You must provide numerical ratings (0.0-10.0) with justifications for each aspect,
and specific suggestions for improvement. Format your response as a structured review
following the required output schema.
"""
        }
        
        user_message = {
            "role": "user",
            "content": f"""Please review the following transcripts and provide a structured evaluation.

TRANSCRIPT A:
{transcript_a}

TRANSCRIPT B:
{transcript_b}

Provide a detailed structured review focused on Transcript A, using Transcript B for comparison where relevant.
"""
        }
        
        messages = [system_message, user_message]
        
        try:
            # Call the LLM for review
            response = await self.model.arun(
                messages=messages,
                response_format=StructuredReview
            )
            
            # Extract and return the structured review
            if hasattr(response.choices[0].message, 'parsed'):
                structured_review = response.choices[0].message.parsed
                
                # Convert to dict for compatibility with existing systems
                review_dict = structured_review.model_dump()
                
                # Add backwards compatibility field
                review_dict['score'] = structured_review.overall_score
                review_dict['critique'] = structured_review.summary_critique
                
                return review_dict
            else:
                # Fallback if parsing fails
                content = response.choices[0].message.get('content', '')
                return {
                    'score': 5.0,  # Neutral score
                    'critique': f"Failed to parse structured response. Raw content: {content[:100]}...",
                    'overall_score': 5.0,
                    'summary_critique': "Parsing error occurred."
                }
                
        except Exception as e:
            # Handle errors gracefully
            return {
                'score': 0.0,
                'critique': f"Error during review: {str(e)}",
                'overall_score': 0.0,
                'summary_critique': f"Error: {str(e)}"
            }

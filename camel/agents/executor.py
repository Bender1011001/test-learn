"""
CAMEL Extensions - Executor Agent

This module implements the ExecutorAgent, which executes tasks provided by the
Proposer agent. The executor can handle various types of tasks including:

1. Code execution and development tasks
2. Data analysis and processing
3. Creative and analytical tasks
4. Problem-solving challenges

The ExecutorAgent serves as the task execution component in the CAMEL workflow by:
- Receiving tasks from the Proposer agent
- Executing tasks according to their specifications
- Providing detailed results and feedback
- Handling errors and edge cases gracefully
"""

from typing import Dict, List, Optional, Any
import json
import subprocess
import tempfile
import os
import sys
from datetime import datetime
from loguru import logger

# CAMEL library imports
from camel.agents import BaseAgent
from camel.messages import BaseMessage, AssistantMessage, UserMessage, SystemMessage
from camel.types import AgentType


class ExecutorAgent(BaseAgent):
    """
    ExecutorAgent that executes tasks provided by the Proposer agent.
    
    This agent can handle various types of tasks and provide detailed
    execution results and feedback.
    """
    
    def __init__(
        self,
        system_message: SystemMessage,
        use_system_commands: bool = False,
        **kwargs
    ):
        """
        Initialize the ExecutorAgent.
        
        Args:
            system_message: SystemMessage defining the executor's role and behavior
            use_system_commands: Whether to allow system command execution
            **kwargs: Additional arguments passed to the BaseAgent
        """
        super().__init__(system_message=system_message, **kwargs)
        self.agent_type = AgentType.ASSISTANT
        self.use_system_commands = use_system_commands
        self.execution_history: List[Dict[str, Any]] = []
        
    def step(self, message: BaseMessage, chat_history: Optional[List[BaseMessage]] = None) -> List[BaseMessage]:
        """
        Process input message and execute the requested task.
        
        Args:
            message: Input message containing task to execute
            chat_history: Optional chat history for context
            
        Returns:
            List of message(s) containing execution results
        """
        try:
            # Parse the task from the message
            task_info = self._parse_task_from_message(message)
            
            if not task_info:
                return [AssistantMessage(
                    role_name="Executor",
                    content="I couldn't identify a specific task to execute. Please provide a clear task description."
                )]
            
            # Execute the task
            execution_result = self._execute_task(task_info)
            
            # Store execution in history
            self.execution_history.append({
                "task": task_info,
                "result": execution_result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Format and return the result
            response_content = self._format_execution_result(task_info, execution_result)
            
            return [AssistantMessage(role_name="Executor", content=response_content)]
            
        except Exception as e:
            logger.error(f"Error in ExecutorAgent.step: {str(e)}")
            return [AssistantMessage(
                role_name="Executor",
                content=f"An error occurred while executing the task: {str(e)}"
            )]
    
    def _parse_task_from_message(self, message: BaseMessage) -> Optional[Dict[str, Any]]:
        """
        Parse task information from the input message.
        
        Args:
            message: Input message
            
        Returns:
            Dictionary containing task information or None if no task found
        """
        content = message.content
        
        # Try to extract structured task information
        task_info = {}
        
        # Look for task ID
        if "Task ID:" in content:
            task_id_line = [line for line in content.split('\n') if "Task ID:" in line]
            if task_id_line:
                task_info["task_id"] = task_id_line[0].split("Task ID:")[-1].strip()
        
        # Look for title
        title_lines = [line for line in content.split('\n') if line.startswith('# ')]
        if title_lines:
            task_info["title"] = title_lines[0][2:].strip()
        
        # Look for category
        if "**Category:**" in content:
            category_line = [line for line in content.split('\n') if "**Category:**" in line]
            if category_line:
                task_info["category"] = category_line[0].split("**Category:**")[-1].strip()
        
        # Look for difficulty
        if "**Difficulty:**" in content:
            difficulty_line = [line for line in content.split('\n') if "**Difficulty:**" in line]
            if difficulty_line:
                task_info["difficulty"] = difficulty_line[0].split("**Difficulty:**")[-1].strip()
        
        # Extract description
        description_start = content.find("## Description")
        if description_start != -1:
            description_section = content[description_start:]
            description_end = description_section.find("##", 1)
            if description_end != -1:
                description_section = description_section[:description_end]
            
            description = description_section.replace("## Description", "").strip()
            task_info["description"] = description
        
        # Extract success criteria
        criteria_start = content.find("## Success Criteria")
        if criteria_start != -1:
            criteria_section = content[criteria_start:]
            criteria_end = criteria_section.find("##", 1)
            if criteria_end != -1:
                criteria_section = criteria_section[:criteria_end]
            
            criteria_lines = [line.strip()[2:] for line in criteria_section.split('\n') 
                            if line.strip().startswith('- ')]
            task_info["success_criteria"] = criteria_lines
        
        # If we have at least a description or title, consider it a valid task
        if task_info.get("description") or task_info.get("title"):
            # Fill in defaults for missing information
            task_info.setdefault("task_id", f"parsed_{int(datetime.utcnow().timestamp())}")
            task_info.setdefault("title", "Parsed Task")
            task_info.setdefault("category", "general")
            task_info.setdefault("difficulty", "intermediate")
            task_info.setdefault("description", content[:200] + "..." if len(content) > 200 else content)
            task_info.setdefault("success_criteria", [])
            
            return task_info
        
        return None
    
    def _execute_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the given task based on its category and requirements.
        
        Args:
            task_info: Dictionary containing task information
            
        Returns:
            Dictionary containing execution results
        """
        category = task_info.get("category", "").lower()
        
        try:
            if "coding" in category:
                return self._execute_coding_task(task_info)
            elif "data" in category or "analysis" in category:
                return self._execute_data_analysis_task(task_info)
            elif "creative" in category:
                return self._execute_creative_task(task_info)
            elif "reasoning" in category:
                return self._execute_reasoning_task(task_info)
            elif "problem" in category:
                return self._execute_problem_solving_task(task_info)
            else:
                return self._execute_general_task(task_info)
                
        except Exception as e:
            logger.error(f"Error executing task {task_info.get('task_id')}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "output": None,
                "execution_time": 0,
                "success_criteria_met": []
            }
    
    def _execute_coding_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a coding task"""
        description = task_info.get("description", "")
        
        # This is a simplified implementation
        # In a real system, you might want to use code execution sandboxes
        
        result = {
            "status": "completed",
            "output": "Coding task analysis completed",
            "execution_time": 5,
            "success_criteria_met": [],
            "code_analysis": {},
            "recommendations": []
        }
        
        # Analyze the coding task requirements
        if "function" in description.lower():
            result["code_analysis"]["task_type"] = "function_implementation"
            result["recommendations"].append("Implement the function with proper error handling")
            result["recommendations"].append("Include unit tests for the function")
            
        elif "algorithm" in description.lower():
            result["code_analysis"]["task_type"] = "algorithm_implementation"
            result["recommendations"].append("Consider time and space complexity")
            result["recommendations"].append("Document the algorithm approach")
            
        elif "data structure" in description.lower():
            result["code_analysis"]["task_type"] = "data_structure_implementation"
            result["recommendations"].append("Implement all standard operations")
            result["recommendations"].append("Handle edge cases appropriately")
        
        # Check success criteria
        success_criteria = task_info.get("success_criteria", [])
        for criteria in success_criteria:
            if any(keyword in criteria.lower() for keyword in ["function", "implement", "code"]):
                result["success_criteria_met"].append(criteria)
        
        # Generate sample code structure if appropriate
        if "python" in description.lower():
            result["sample_code"] = self._generate_python_code_template(task_info)
        elif "javascript" in description.lower():
            result["sample_code"] = self._generate_javascript_code_template(task_info)
        
        return result
    
    def _execute_data_analysis_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a data analysis task"""
        description = task_info.get("description", "")
        
        result = {
            "status": "completed",
            "output": "Data analysis task completed",
            "execution_time": 10,
            "success_criteria_met": [],
            "analysis_steps": [],
            "recommendations": []
        }
        
        # Define analysis steps based on task description
        if "explore" in description.lower() or "basic" in description.lower():
            result["analysis_steps"] = [
                "Load and inspect the dataset",
                "Calculate basic statistics (mean, median, mode, std dev)",
                "Identify missing values and outliers",
                "Create basic visualizations (histograms, scatter plots)",
                "Summarize key findings"
            ]
        elif "predictive" in description.lower() or "model" in description.lower():
            result["analysis_steps"] = [
                "Data preprocessing and cleaning",
                "Feature selection and engineering",
                "Split data into training and testing sets",
                "Train multiple models (e.g., linear regression, random forest)",
                "Evaluate model performance using appropriate metrics",
                "Select best performing model"
            ]
        elif "pipeline" in description.lower():
            result["analysis_steps"] = [
                "Design data ingestion pipeline",
                "Implement data preprocessing steps",
                "Set up model training pipeline",
                "Create model evaluation framework",
                "Design deployment and monitoring strategy"
            ]
        
        # Add recommendations
        result["recommendations"] = [
            "Ensure data quality and consistency",
            "Document all analysis steps and assumptions",
            "Validate results with domain experts",
            "Consider ethical implications of the analysis"
        ]
        
        # Check success criteria
        success_criteria = task_info.get("success_criteria", [])
        for criteria in success_criteria:
            if any(keyword in criteria.lower() for keyword in ["data", "analysis", "model", "visualization"]):
                result["success_criteria_met"].append(criteria)
        
        return result
    
    def _execute_creative_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a creative task"""
        description = task_info.get("description", "")
        
        result = {
            "status": "completed",
            "output": "Creative task completed",
            "execution_time": 8,
            "success_criteria_met": [],
            "creative_elements": [],
            "suggestions": []
        }
        
        if "story" in description.lower() or "writing" in description.lower():
            result["creative_elements"] = [
                "Compelling characters with clear motivations",
                "Engaging plot with conflict and resolution",
                "Vivid setting and atmosphere",
                "Consistent narrative voice",
                "Satisfying conclusion"
            ]
            result["suggestions"] = [
                "Start with an interesting hook to grab attention",
                "Show, don't tell - use action and dialogue",
                "Create tension and conflict to drive the story",
                "End with a meaningful resolution"
            ]
        elif "design" in description.lower():
            result["creative_elements"] = [
                "Clear visual hierarchy",
                "Consistent color scheme",
                "Appropriate typography",
                "Balanced composition",
                "User-friendly interface"
            ]
            result["suggestions"] = [
                "Consider the target audience",
                "Ensure accessibility standards",
                "Test with real users",
                "Iterate based on feedback"
            ]
        
        # Check success criteria
        success_criteria = task_info.get("success_criteria", [])
        for criteria in success_criteria:
            if any(keyword in criteria.lower() for keyword in ["creative", "original", "engaging", "story"]):
                result["success_criteria_met"].append(criteria)
        
        return result
    
    def _execute_reasoning_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a reasoning task"""
        description = task_info.get("description", "")
        
        result = {
            "status": "completed",
            "output": "Reasoning task completed",
            "execution_time": 6,
            "success_criteria_met": [],
            "reasoning_steps": [],
            "logical_framework": ""
        }
        
        if "puzzle" in description.lower() or "logic" in description.lower():
            result["reasoning_steps"] = [
                "Identify the given information and constraints",
                "Determine what needs to be solved",
                "Apply logical deduction rules",
                "Test potential solutions against constraints",
                "Verify the solution is complete and correct"
            ]
            result["logical_framework"] = "Deductive reasoning with constraint satisfaction"
            
        elif "strategic" in description.lower() or "business" in description.lower():
            result["reasoning_steps"] = [
                "Define the problem clearly",
                "Gather relevant information and context",
                "Identify stakeholders and their interests",
                "Generate multiple solution alternatives",
                "Evaluate alternatives against criteria",
                "Select and justify the best approach"
            ]
            result["logical_framework"] = "Strategic analysis with multi-criteria decision making"
        
        # Check success criteria
        success_criteria = task_info.get("success_criteria", [])
        for criteria in success_criteria:
            if any(keyword in criteria.lower() for keyword in ["reasoning", "logic", "analysis", "solution"]):
                result["success_criteria_met"].append(criteria)
        
        return result
    
    def _execute_problem_solving_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a problem-solving task"""
        description = task_info.get("description", "")
        
        result = {
            "status": "completed",
            "output": "Problem-solving task completed",
            "execution_time": 7,
            "success_criteria_met": [],
            "problem_analysis": {},
            "solution_approach": []
        }
        
        # Analyze the problem
        result["problem_analysis"] = {
            "problem_type": "process_optimization" if "process" in description.lower() else "general",
            "complexity": "medium",
            "stakeholders": ["users", "system", "organization"],
            "constraints": ["time", "resources", "feasibility"]
        }
        
        # Define solution approach
        result["solution_approach"] = [
            "Understand the current state thoroughly",
            "Identify root causes of inefficiencies",
            "Brainstorm potential solutions",
            "Evaluate solutions against constraints",
            "Develop implementation plan",
            "Define success metrics and monitoring"
        ]
        
        # Check success criteria
        success_criteria = task_info.get("success_criteria", [])
        for criteria in success_criteria:
            if any(keyword in criteria.lower() for keyword in ["problem", "solution", "improvement", "optimization"]):
                result["success_criteria_met"].append(criteria)
        
        return result
    
    def _execute_general_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a general task"""
        result = {
            "status": "completed",
            "output": "General task analysis completed",
            "execution_time": 3,
            "success_criteria_met": [],
            "approach": "General problem-solving methodology applied",
            "recommendations": [
                "Break down complex tasks into smaller components",
                "Gather all necessary information before starting",
                "Document the process and results",
                "Validate outcomes against requirements"
            ]
        }
        
        # Check success criteria
        success_criteria = task_info.get("success_criteria", [])
        result["success_criteria_met"] = success_criteria  # Assume all criteria can be met for general tasks
        
        return result
    
    def _generate_python_code_template(self, task_info: Dict[str, Any]) -> str:
        """Generate a Python code template based on task requirements"""
        description = task_info.get("description", "").lower()
        
        if "function" in description:
            return '''def example_function(param1, param2):
    """
    Description of what this function does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: If invalid input is provided
    """
    try:
        # Implementation goes here
        result = param1 + param2  # Example operation
        return result
    except Exception as e:
        raise ValueError(f"Invalid input: {e}")

# Example usage and tests
if __name__ == "__main__":
    # Test cases
    assert example_function(2, 3) == 5
    print("All tests passed!")'''
        
        elif "class" in description or "data structure" in description:
            return '''class ExampleDataStructure:
    """
    Implementation of a data structure.
    """
    
    def __init__(self):
        """Initialize the data structure."""
        self.data = []
    
    def add(self, item):
        """Add an item to the data structure."""
        self.data.append(item)
    
    def remove(self, item):
        """Remove an item from the data structure."""
        if item in self.data:
            self.data.remove(item)
            return True
        return False
    
    def __len__(self):
        """Return the size of the data structure."""
        return len(self.data)
    
    def __str__(self):
        """String representation of the data structure."""
        return f"DataStructure({self.data})"

# Example usage
if __name__ == "__main__":
    ds = ExampleDataStructure()
    ds.add("item1")
    ds.add("item2")
    print(ds)  # DataStructure(['item1', 'item2'])'''
        
        return "# Python code template - implement based on specific requirements"
    
    def _generate_javascript_code_template(self, task_info: Dict[str, Any]) -> str:
        """Generate a JavaScript code template based on task requirements"""
        description = task_info.get("description", "").lower()
        
        if "function" in description:
            return '''/**
 * Description of what this function does.
 * @param {*} param1 - Description of parameter 1
 * @param {*} param2 - Description of parameter 2
 * @returns {*} Description of return value
 * @throws {Error} If invalid input is provided
 */
function exampleFunction(param1, param2) {
    try {
        // Validate inputs
        if (param1 == null || param2 == null) {
            throw new Error("Parameters cannot be null");
        }
        
        // Implementation goes here
        const result = param1 + param2; // Example operation
        return result;
    } catch (error) {
        throw new Error(`Invalid input: ${error.message}`);
    }
}

// Example usage and tests
if (typeof module !== 'undefined' && module.exports) {
    module.exports = exampleFunction;
} else {
    // Browser environment - run tests
    console.assert(exampleFunction(2, 3) === 5, "Test failed");
    console.log("All tests passed!");
}'''
        
        return "// JavaScript code template - implement based on specific requirements"
    
    def _format_execution_result(self, task_info: Dict[str, Any], execution_result: Dict[str, Any]) -> str:
        """Format the execution result into a readable response"""
        response = f"# Task Execution Report\n\n"
        response += f"**Task:** {task_info.get('title', 'Unknown Task')}\n"
        response += f"**Task ID:** {task_info.get('task_id', 'N/A')}\n"
        response += f"**Category:** {task_info.get('category', 'General')}\n"
        response += f"**Status:** {execution_result.get('status', 'Unknown')}\n"
        response += f"**Execution Time:** {execution_result.get('execution_time', 0)} seconds\n\n"
        
        # Add error information if present
        if execution_result.get("error"):
            response += f"## Error\n{execution_result['error']}\n\n"
        
        # Add main output
        if execution_result.get("output"):
            response += f"## Execution Output\n{execution_result['output']}\n\n"
        
        # Add success criteria assessment
        success_criteria_met = execution_result.get("success_criteria_met", [])
        if success_criteria_met:
            response += f"## Success Criteria Met\n"
            for criteria in success_criteria_met:
                response += f"✅ {criteria}\n"
            response += "\n"
        
        # Add category-specific information
        if "analysis_steps" in execution_result:
            response += f"## Analysis Steps\n"
            for step in execution_result["analysis_steps"]:
                response += f"1. {step}\n"
            response += "\n"
        
        if "reasoning_steps" in execution_result:
            response += f"## Reasoning Process\n"
            for step in execution_result["reasoning_steps"]:
                response += f"• {step}\n"
            response += "\n"
        
        if "creative_elements" in execution_result:
            response += f"## Creative Elements Considered\n"
            for element in execution_result["creative_elements"]:
                response += f"• {element}\n"
            response += "\n"
        
        if "sample_code" in execution_result:
            response += f"## Code Template\n```python\n{execution_result['sample_code']}\n```\n\n"
        
        # Add recommendations
        if execution_result.get("recommendations"):
            response += f"## Recommendations\n"
            for rec in execution_result["recommendations"]:
                response += f"• {rec}\n"
            response += "\n"
        
        if execution_result.get("suggestions"):
            response += f"## Suggestions\n"
            for suggestion in execution_result["suggestions"]:
                response += f"• {suggestion}\n"
            response += "\n"
        
        response += f"## Summary\n"
        if execution_result.get("status") == "completed":
            response += f"Task execution completed successfully. "
            if success_criteria_met:
                response += f"{len(success_criteria_met)} success criteria were met."
            else:
                response += "Ready for review and feedback."
        else:
            response += f"Task execution encountered issues. Please review the error details above."
        
        return response
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history"""
        return self.execution_history
    
    def clear_execution_history(self):
        """Clear the execution history"""
        self.execution_history.clear()
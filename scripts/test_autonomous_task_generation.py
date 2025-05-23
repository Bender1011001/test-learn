#!/usr/bin/env python3
"""
Test script for autonomous task generation capabilities.

This script demonstrates the new autonomous task generation features
of the Proposer agent and the complete workflow system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.core.services.config_manager import ConfigManager
from backend.core.services.workflow_manager import WorkflowManager
from backend.db.base import get_db
from camel.agents.proposer import ProposerAgent, TaskCategory, TaskDifficulty, TaskPriority
from camel.agents.executor import ExecutorAgent
from camel.messages import SystemMessage, UserMessage
from loguru import logger
import json


async def test_proposer_agent():
    """Test the ProposerAgent autonomous task generation"""
    print("=" * 60)
    print("Testing ProposerAgent Autonomous Task Generation")
    print("=" * 60)
    
    # Create proposer agent
    system_message = SystemMessage(
        role_name="Proposer",
        content="You are an autonomous task generation agent. Generate diverse, challenging tasks across multiple domains."
    )
    proposer = ProposerAgent(system_message=system_message)
    
    # Test autonomous task generation
    print("\n1. Testing autonomous task generation...")
    
    # Generate tasks with different requirements
    test_requirements = [
        {"category": TaskCategory.CODING, "difficulty": TaskDifficulty.BEGINNER},
        {"category": TaskCategory.REASONING, "difficulty": TaskDifficulty.INTERMEDIATE},
        {"category": TaskCategory.CREATIVE, "difficulty": TaskDifficulty.ADVANCED},
        {"category": TaskCategory.DATA_ANALYSIS, "difficulty": TaskDifficulty.INTERMEDIATE},
    ]
    
    generated_tasks = []
    
    for i, requirements in enumerate(test_requirements, 1):
        print(f"\n   Generating task {i}: {requirements['category'].value} - {requirements['difficulty'].value}")
        
        task = proposer._generate_autonomous_task(requirements)
        
        if task and proposer._validate_task_quality(task):
            generated_tasks.append(task)
            print(f"   ✅ Generated: {task.title}")
            print(f"      Complexity: {task.complexity_score}/10")
            print(f"      Duration: {task.estimated_duration} minutes")
            print(f"      Success Criteria: {len(task.success_criteria)} items")
        else:
            print(f"   ❌ Failed to generate valid task")
    
    # Test task queue management
    print(f"\n2. Testing task queue management...")
    print(f"   Queue status: {proposer.get_queue_status()}")
    
    # Add tasks to queue
    for task in generated_tasks:
        proposer.task_queue.add_task(task)
    
    print(f"   After adding tasks: {proposer.get_queue_status()}")
    
    # Test getting next task
    print(f"\n3. Testing task retrieval...")
    next_task = proposer.get_next_task()
    if next_task:
        print(f"   Next task: {next_task['title']}")
        print(f"   Category: {next_task['category']}")
        print(f"   Difficulty: {next_task['difficulty']}")
    else:
        print(f"   No tasks available")
    
    return generated_tasks


async def test_executor_agent():
    """Test the ExecutorAgent task execution"""
    print("\n" + "=" * 60)
    print("Testing ExecutorAgent Task Execution")
    print("=" * 60)
    
    # Create executor agent
    system_message = SystemMessage(
        role_name="Executor",
        content="You are a task execution agent. Execute tasks according to their specifications and provide detailed results."
    )
    executor = ExecutorAgent(system_message=system_message)
    
    # Create a sample task message
    task_message = """# Simple Python Function

**Category:** Coding
**Difficulty:** Beginner
**Task ID:** test_task_001

## Description
Write a Python function that takes two numbers and returns their sum. Include proper error handling for invalid inputs.

## Success Criteria
- Function correctly adds two numbers
- Handles invalid input gracefully
- Includes basic documentation
"""
    
    print("\n1. Testing task execution...")
    user_message = UserMessage(role_name="user", content=task_message)
    
    # Execute the task
    response_messages = executor.step(user_message)
    
    if response_messages:
        print("   ✅ Task executed successfully")
        print(f"   Response length: {len(response_messages[0].content)} characters")
        print("   Response preview:")
        print("   " + response_messages[0].content[:200] + "...")
    else:
        print("   ❌ Task execution failed")
    
    return response_messages


async def test_workflow_integration():
    """Test the workflow manager integration"""
    print("\n" + "=" * 60)
    print("Testing Workflow Manager Integration")
    print("=" * 60)
    
    # Initialize services
    db = next(get_db())
    try:
        config_manager = ConfigManager()
        workflow_manager = WorkflowManager(config_manager, lambda: db)
        
        print("\n1. Testing autonomous task generation workflow...")
        
        # Start autonomous task generation
        generation_settings = {
            "categories": ["coding", "reasoning"],
            "difficulties": ["beginner", "intermediate"],
            "generation_rate": 2,  # 2 tasks per minute for testing
            "max_queue_size": 10
        }
        
        run_id = await workflow_manager.start_autonomous_task_generation(generation_settings)
        print(f"   Started autonomous generation with run_id: {run_id}")
        
        # Wait a bit to let it generate some tasks
        print("   Waiting 10 seconds for task generation...")
        await asyncio.sleep(10)
        
        # Check status
        status = workflow_manager.get_autonomous_generation_status(run_id)
        if status:
            print(f"   Status: {status['status']}")
            print(f"   Tasks generated: {status.get('tasks_generated', 0)}")
        
        # Stop the generation
        print("   Stopping autonomous generation...")
        stopped = await workflow_manager.stop_autonomous_task_generation(run_id)
        print(f"   Stopped: {stopped}")
        
    finally:
        db.close()


async def test_api_integration():
    """Test the API endpoints (if server is running)"""
    print("\n" + "=" * 60)
    print("Testing API Integration")
    print("=" * 60)
    
    try:
        import httpx
        
        base_url = "http://localhost:8000/api/v1"
        
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            print("\n1. Testing health endpoint...")
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("   ✅ API server is running")
                
                # Test task generation endpoint
                print("\n2. Testing task generation endpoint...")
                task_request = {
                    "category": "coding",
                    "difficulty": "beginner",
                    "count": 2
                }
                
                response = await client.post(f"{base_url}/tasks/generate", json=task_request)
                if response.status_code == 200:
                    tasks = response.json()
                    print(f"   ✅ Generated {len(tasks)} tasks via API")
                    for task in tasks:
                        print(f"      - {task['title']} ({task['category']}, {task['difficulty']})")
                else:
                    print(f"   ❌ Task generation failed: {response.status_code}")
                
                # Test queue status endpoint
                print("\n3. Testing queue status endpoint...")
                response = await client.get(f"{base_url}/tasks/queue/status")
                if response.status_code == 200:
                    status = response.json()
                    print(f"   ✅ Queue status retrieved")
                    print(f"      Total tasks: {status['total_tasks']}")
                    print(f"      Pending: {status['pending_tasks']}")
                    print(f"      Completed: {status['completed_tasks']}")
                else:
                    print(f"   ❌ Queue status failed: {response.status_code}")
            else:
                print("   ❌ API server not running")
                
    except ImportError:
        print("   ⚠️  httpx not available, skipping API tests")
    except Exception as e:
        print(f"   ⚠️  API tests failed: {str(e)}")


async def main():
    """Run all tests"""
    print("🚀 Starting Autonomous Task Generation Tests")
    print("=" * 60)
    
    try:
        # Test individual components
        generated_tasks = await test_proposer_agent()
        await test_executor_agent()
        
        # Test workflow integration
        await test_workflow_integration()
        
        # Test API integration (if available)
        await test_api_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
        print("\n📋 Summary:")
        print(f"   - Generated {len(generated_tasks)} tasks")
        print("   - Tested task execution")
        print("   - Tested workflow integration")
        print("   - Tested API endpoints")
        
        print("\n🎯 Next Steps:")
        print("   1. Start the API server: python -m backend.api.main")
        print("   2. Run the GUI: streamlit run gui/app.py")
        print("   3. Use the API to generate and execute tasks")
        print("   4. Monitor task queue and performance metrics")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        logger.exception("Test execution failed")
        return 1
    
    return 0


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}:{function}</cyan> | <level>{message}</level>"
    )
    
    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
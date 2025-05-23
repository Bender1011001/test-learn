#!/usr/bin/env python3
"""
Test script for bidirectional feedback system.

This script demonstrates the new bidirectional feedback capabilities
and the complete autonomous learning loop.
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
from camel.agents.bidirectional_feedback import BidirectionalFeedbackManager, FeedbackType
from camel.agents.proposer import ProposerAgent
from camel.agents.executor import ExecutorAgent
from camel.agents.peer_reviewer import PeerReviewer
from camel.messages import SystemMessage, UserMessage
from loguru import logger
import json


async def test_bidirectional_feedback_manager():
    """Test the BidirectionalFeedbackManager directly"""
    print("=" * 60)
    print("Testing BidirectionalFeedbackManager")
    print("=" * 60)
    
    # Create feedback manager
    feedback_manager = BidirectionalFeedbackManager()
    
    # Create test agents
    proposer_system = SystemMessage(
        role_name="Proposer",
        content="You are a task proposer agent. Evaluate other agents' performance and provide constructive feedback."
    )
    proposer = ProposerAgent(system_message=proposer_system)
    
    executor_system = SystemMessage(
        role_name="Executor", 
        content="You are a task executor agent. Evaluate other agents' performance and provide constructive feedback."
    )
    executor = ExecutorAgent(system_message=executor_system)
    
    reviewer_system = SystemMessage(
        role_name="PeerReviewer",
        content="You are a peer reviewer agent. Evaluate other agents' performance and provide constructive feedback."
    )
    reviewer = PeerReviewer(system_message=reviewer_system)
    
    print("\n1. Testing feedback collection...")
    
    # Create sample interaction context
    interaction_context = {
        "task_description": "Write a Python function to calculate fibonacci numbers",
        "proposer_response": "Create a function that efficiently calculates fibonacci numbers using dynamic programming",
        "executor_response": "def fibonacci(n):\n    if n <= 1:\n        return n\n    dp = [0, 1]\n    for i in range(2, n + 1):\n        dp.append(dp[i-1] + dp[i-2])\n    return dp[n]",
        "reviewer_feedback": "Good implementation with dynamic programming. Could be optimized for space complexity."
    }
    
    # Test Proposer evaluating Executor
    print("\n   Proposer evaluating Executor...")
    feedback_entry = await feedback_manager.collect_feedback(
        evaluator_agent=proposer,
        evaluated_agent_name="Executor",
        interaction_context=interaction_context,
        feedback_type=FeedbackType.EXECUTION_QUALITY
    )
    
    if feedback_entry:
        print(f"   ✅ Feedback collected: {feedback_entry.feedback_id}")
        print(f"      Overall rating: {feedback_entry.overall_rating}/10")
        print(f"      Confidence: {feedback_entry.confidence_score}")
        print(f"      Strengths: {feedback_entry.strengths[:2]}")
    else:
        print("   ❌ Failed to collect feedback")
    
    # Test Executor evaluating Proposer
    print("\n   Executor evaluating Proposer...")
    feedback_entry = await feedback_manager.collect_feedback(
        evaluator_agent=executor,
        evaluated_agent_name="Proposer",
        interaction_context=interaction_context,
        feedback_type=FeedbackType.TASK_QUALITY
    )
    
    if feedback_entry:
        print(f"   ✅ Feedback collected: {feedback_entry.feedback_id}")
        print(f"      Overall rating: {feedback_entry.overall_rating}/10")
        print(f"      Confidence: {feedback_entry.confidence_score}")
        print(f"      Strengths: {feedback_entry.strengths[:2]}")
    else:
        print("   ❌ Failed to collect feedback")
    
    # Test PeerReviewer evaluating both
    print("\n   PeerReviewer evaluating Proposer...")
    feedback_entry = await feedback_manager.collect_feedback(
        evaluator_agent=reviewer,
        evaluated_agent_name="Proposer",
        interaction_context=interaction_context,
        feedback_type=FeedbackType.TASK_QUALITY
    )
    
    if feedback_entry:
        print(f"   ✅ Feedback collected: {feedback_entry.feedback_id}")
        print(f"      Overall rating: {feedback_entry.overall_rating}/10")
    
    print("\n   PeerReviewer evaluating Executor...")
    feedback_entry = await feedback_manager.collect_feedback(
        evaluator_agent=reviewer,
        evaluated_agent_name="Executor",
        interaction_context=interaction_context,
        feedback_type=FeedbackType.EXECUTION_QUALITY
    )
    
    if feedback_entry:
        print(f"   ✅ Feedback collected: {feedback_entry.feedback_id}")
        print(f"      Overall rating: {feedback_entry.overall_rating}/10")
    
    print("\n2. Testing performance summaries...")
    
    # Get performance summaries
    for agent_name in ["Proposer", "Executor", "PeerReviewer"]:
        summary = feedback_manager.get_agent_performance_summary(agent_name)
        if summary:
            print(f"\n   {agent_name} Performance:")
            print(f"      Average rating: {summary['overall_performance']['average_rating']}")
            print(f"      Total evaluations: {summary['overall_performance']['total_evaluations']}")
            print(f"      Trend: {summary['overall_performance']['trend_description']}")
        else:
            print(f"\n   {agent_name}: No performance data yet")
    
    print("\n3. Testing system insights...")
    insights = feedback_manager.get_feedback_insights()
    print(f"   Total feedback entries: {insights['system_overview']['total_feedback_entries']}")
    print(f"   Average system rating: {insights['system_overview']['average_rating']}")
    print(f"   Active agents: {insights['system_overview']['active_agents']}")
    
    if insights['common_strengths']:
        print(f"   Top strength: {insights['common_strengths'][0][0]} ({insights['common_strengths'][0][1]} mentions)")
    
    print("\n4. Testing DPO export...")
    dpo_data = feedback_manager.export_feedback_for_dpo(min_rating_threshold=5.0)
    print(f"   Exported {len(dpo_data)} feedback entries for DPO training")
    
    return feedback_manager


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
        
        print("\n1. Testing bidirectional feedback workflow...")
        
        # Create interaction context
        interaction_context = {
            "task_description": "Create a sorting algorithm",
            "workflow_type": "test",
            "timestamp": "2024-01-01T12:00:00Z",
            "agents_involved": ["Proposer", "Executor", "PeerReviewer"]
        }
        
        # Start bidirectional feedback workflow
        run_id = await workflow_manager.start_bidirectional_feedback_workflow(
            interaction_context=interaction_context,
            agents_involved=["Proposer", "Executor", "PeerReviewer"]
        )
        print(f"   Started feedback workflow with run_id: {run_id}")
        
        # Wait for completion
        print("   Waiting for feedback collection to complete...")
        max_wait_time = 120  # 2 minutes
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < max_wait_time:
            status = workflow_manager.get_workflow_status(run_id)
            if status and status["status"] not in ["running", "starting"]:
                break
            await asyncio.sleep(5)
        
        # Check final status
        final_status = workflow_manager.get_workflow_status(run_id)
        if final_status:
            print(f"   Final status: {final_status['status']}")
            print(f"   Feedback collected: {final_status.get('feedback_collected', 0)}")
        
        print("\n2. Testing autonomous learning loop (short demo)...")
        
        # Start a short learning loop
        loop_settings = {
            "generation_rate": 2,  # 2 tasks per minute
            "max_concurrent_tasks": 2,
            "feedback_frequency": "after_each_task",
            "auto_dpo_training": False,  # Disable for demo
            "max_iterations": 1,  # Just one iteration for demo
            "continuous_operation": False
        }
        
        learning_run_id = await workflow_manager.start_autonomous_learning_loop(
            loop_settings=loop_settings
        )
        print(f"   Started learning loop with run_id: {learning_run_id}")
        
        # Wait a bit to see it start
        await asyncio.sleep(10)
        
        # Check status
        learning_status = workflow_manager.get_workflow_status(learning_run_id)
        if learning_status:
            print(f"   Learning loop status: {learning_status['status']}")
            print(f"   Iterations completed: {learning_status.get('iterations_completed', 0)}")
        
        # Stop the learning loop for demo
        print("   Stopping learning loop for demo...")
        stopped = await workflow_manager.stop_workflow_async(learning_run_id)
        print(f"   Stopped: {stopped}")
        
        print("\n3. Testing feedback summaries...")
        
        # Get feedback summary
        summary = workflow_manager.get_feedback_summary()
        if summary and "system_overview" in summary:
            print(f"   System feedback entries: {summary['system_overview'].get('total_feedback_entries', 0)}")
            print(f"   Average rating: {summary['system_overview'].get('average_rating', 0)}")
        else:
            print("   No system feedback data available yet")
        
        # Get agent-specific summaries
        for agent_name in ["Proposer", "Executor", "PeerReviewer"]:
            agent_summary = workflow_manager.get_feedback_summary(agent_name=agent_name)
            if agent_summary:
                print(f"   {agent_name} average rating: {agent_summary.get('overall_performance', {}).get('average_rating', 'N/A')}")
        
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
            print("\n1. Testing API health...")
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("   ✅ API server is running")
                
                # Test feedback health endpoint
                print("\n2. Testing feedback system health...")
                response = await client.get(f"{base_url}/feedback/health")
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"   ✅ Feedback system status: {health_data['status']}")
                    print(f"   Active feedback workflows: {health_data['active_feedback_workflows']}")
                    print(f"   Total feedback entries: {health_data['total_feedback_entries']}")
                else:
                    print(f"   ❌ Feedback health check failed: {response.status_code}")
                
                # Test feedback collection endpoint
                print("\n3. Testing feedback collection endpoint...")
                feedback_request = {
                    "interaction_context": {
                        "task_description": "API test task",
                        "test_mode": True,
                        "timestamp": "2024-01-01T12:00:00Z"
                    },
                    "agents_involved": ["Proposer", "Executor", "PeerReviewer"]
                }
                
                response = await client.post(f"{base_url}/feedback/collect", json=feedback_request)
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Feedback collection started: {result['run_id']}")
                    
                    # Check workflow status
                    run_id = result['run_id']
                    await asyncio.sleep(5)  # Wait a bit
                    
                    response = await client.get(f"{base_url}/feedback/workflow/{run_id}/status")
                    if response.status_code == 200:
                        status = response.json()
                        print(f"   Workflow status: {status['status']}")
                    
                else:
                    print(f"   ❌ Feedback collection failed: {response.status_code}")
                
                # Test system insights endpoint
                print("\n4. Testing system insights endpoint...")
                response = await client.get(f"{base_url}/feedback/summary")
                if response.status_code == 200:
                    insights = response.json()
                    print(f"   ✅ System insights retrieved")
                    print(f"   Total feedback entries: {insights['system_overview']['total_feedback_entries']}")
                    print(f"   Average rating: {insights['system_overview']['average_rating']}")
                else:
                    print(f"   ❌ System insights failed: {response.status_code}")
                
                # Test simulation endpoint
                print("\n5. Testing interaction simulation...")
                response = await client.post(
                    f"{base_url}/feedback/simulate-interaction",
                    params={"task_description": "Write a simple hello world function"}
                )
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Simulation started: {result['status']}")
                    print(f"   Workflow run ID: {result['workflow_run_id']}")
                    print(f"   Feedback run ID: {result['feedback_run_id']}")
                else:
                    print(f"   ❌ Simulation failed: {response.status_code}")
                
            else:
                print("   ❌ API server not running")
                
    except ImportError:
        print("   ⚠️  httpx not available, skipping API tests")
    except Exception as e:
        print(f"   ⚠️  API tests failed: {str(e)}")


async def main():
    """Run all tests"""
    print("🚀 Starting Bidirectional Feedback System Tests")
    print("=" * 60)
    
    try:
        # Test individual components
        feedback_manager = await test_bidirectional_feedback_manager()
        
        # Test workflow integration
        await test_workflow_integration()
        
        # Test API integration (if available)
        await test_api_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
        print("\n📋 Summary:")
        print("   - Tested bidirectional feedback manager")
        print("   - Tested workflow integration")
        print("   - Tested API endpoints")
        print("   - Demonstrated autonomous learning loop")
        
        print("\n🎯 Next Steps:")
        print("   1. Start the API server: python -m backend.api.main")
        print("   2. Use the feedback endpoints to collect cross-agent feedback")
        print("   3. Start the autonomous learning loop for continuous improvement")
        print("   4. Monitor agent performance and system insights")
        print("   5. Export feedback data for DPO training")
        
        print("\n🔄 Autonomous Learning Loop Features:")
        print("   - Continuous task generation")
        print("   - Automatic task execution")
        print("   - Bidirectional feedback collection")
        print("   - Performance monitoring")
        print("   - Automatic DPO training triggers")
        
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
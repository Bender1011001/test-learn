#!/usr/bin/env python3
"""
Load demo data into the database for development.

This script creates sample workflows, logs, and annotations for development and testing.
Run it with: python backend/scripts/load_demo_data.py
"""
import os
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import random
import uuid
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import models and database
from backend.db.base import SessionLocal, engine, Base
from backend.db.models.logs import InteractionLog, DPOAnnotation


def create_demo_workflows(db, count=5):
    """Create demo workflow runs."""
    workflow_runs = []
    workflow_ids = ["proposer_executor_review_loop"]
    agent_names = ["Proposer", "Executor", "PeerReviewer", "system"]
    
    print(f"Creating {count} demo workflow runs...")
    
    for i in range(count):
        # Generate a workflow run ID
        run_id = str(uuid.uuid4())
        workflow_runs.append(run_id)
        
        # Start time decreases as we go back in time (older workflows first)
        start_time = datetime.utcnow() - timedelta(days=i, hours=random.randint(0, 12))
        
        # Demo goals
        goals = [
            "Create a comprehensive report on renewable energy trends",
            "Optimize the database query performance",
            "Develop a testing strategy for the new authentication module",
            "Design a responsive UI for the mobile app",
            "Implement a caching strategy for the API"
        ]
        
        # Create start log
        start_log = InteractionLog(
            workflow_run_id=run_id,
            timestamp=start_time,
            agent_name="system",
            agent_type="system",
            input_data={"workflow_id": workflow_ids[0]},
            output_data={"status": "started", "initial_goal": goals[i % len(goals)]}
        )
        db.add(start_log)
        
        # Create agent interaction logs (20 per workflow)
        for j in range(20):
            agent_idx = j % (len(agent_names) - 1)  # Skip "system" for regular logs
            agent_name = agent_names[agent_idx]
            agent_type = agent_name
            log_time = start_time + timedelta(seconds=j*30)  # 30 seconds between logs
            
            # Create realistic input data based on agent type
            if agent_name == "Proposer":
                input_data = {
                    "goal": goals[i % len(goals)],
                    "step": j,
                    "context": f"Current iteration: {j//3 + 1}"
                }
                output_data = {
                    "proposal": f"I propose to {random.choice(['research', 'analyze', 'examine', 'investigate'])} {random.choice(['the latest', 'current', 'emerging', 'relevant'])} {random.choice(['trends', 'developments', 'patterns', 'data'])} related to {goals[i % len(goals)]}.",
                    "confidence": random.uniform(0.7, 0.95)
                }
            elif agent_name == "Executor":
                input_data = {
                    "command": f"Execute step {j} of the plan",
                    "step": j,
                    "context": f"Based on proposal from step {j-1}"
                }
                output_data = {
                    "result": f"Executed the command and found {random.choice(['interesting', 'important', 'relevant', 'useful'])} {random.choice(['data', 'information', 'results', 'findings'])}.",
                    "stdout": f"Command completed with {random.choice(['success', 'no errors', 'expected output'])}",
                    "stderr": "",
                    "exit_code": 0
                }
            elif agent_name == "PeerReviewer":
                input_data = {
                    "proposal": f"Review the action from step {j-1}",
                    "step": j,
                    "context": f"Reviewing execution results"
                }
                output_data = {
                    "rating": random.randint(3, 5),
                    "feedback": f"The {random.choice(['approach', 'execution', 'implementation', 'solution'])} was {random.choice(['good', 'effective', 'appropriate', 'well-designed'])}, but could be improved by {random.choice(['considering', 'adding', 'focusing on', 'enhancing'])} {random.choice(['more details', 'broader context', 'edge cases', 'alternative approaches'])}.",
                    "suggestions": [
                        f"Consider {random.choice(['exploring', 'investigating', 'analyzing', 'examining'])} {random.choice(['additional', 'alternative', 'related', 'complementary'])} {random.choice(['sources', 'approaches', 'methods', 'perspectives'])}."
                    ]
                }
            
            # Create the log entry
            log = InteractionLog(
                workflow_run_id=run_id,
                timestamp=log_time,
                agent_name=agent_name,
                agent_type=agent_type,
                input_data=input_data,
                output_data=output_data
            )
            db.add(log)
            db.flush()  # To get the log ID
            
            # Add annotations to some Proposer logs
            if agent_name == "Proposer" and random.random() < 0.7:  # 70% chance of annotation
                # Create an annotation
                annotation = DPOAnnotation(
                    log_entry_id=log.id,
                    rating=random.uniform(3.0, 5.0),
                    rationale=f"This proposal is {random.choice(['good', 'effective', 'appropriate', 'well-designed'])} because it {random.choice(['addresses', 'tackles', 'focuses on', 'considers'])} the key aspects of the goal.",
                    chosen_prompt=f"For the goal '{goals[i % len(goals)]}', I would {random.choice(['propose', 'suggest', 'recommend', 'advise'])} {random.choice(['investigating', 'researching', 'analyzing', 'examining'])} the {random.choice(['most relevant', 'key', 'critical', 'essential'])} factors.",
                    rejected_prompt=f"I think we should just {random.choice(['look at', 'check', 'see about', 'try'])} {goals[i % len(goals)]}.",
                    dpo_context=f"Given the goal: '{goals[i % len(goals)]}', propose a detailed action plan.",
                    user_id="demo_user",
                    timestamp=log_time + timedelta(minutes=random.randint(5, 60))
                )
                db.add(annotation)
        
        # Create completion log
        end_log = InteractionLog(
            workflow_run_id=run_id,
            timestamp=start_time + timedelta(seconds=20*30 + 10),
            agent_name="system",
            agent_type="system",
            input_data={"workflow_id": workflow_ids[0]},
            output_data={"status": "completed"}
        )
        db.add(end_log)
    
    # Commit all the logs and annotations
    db.commit()
    print(f"Created {count} workflow runs with logs and annotations.")
    return workflow_runs


def main():
    """Load demo data into the database."""
    parser = argparse.ArgumentParser(description="Load demo data into the database")
    parser.add_argument("--drop", action="store_true", help="Drop existing tables and recreate them")
    parser.add_argument("--count", type=int, default=5, help="Number of workflow runs to create")
    args = parser.parse_args()

    # Create tables if they don't exist (or drop and recreate if --drop is specified)
    if args.drop:
        print("Dropping existing tables...")
        Base.metadata.drop_all(bind=engine)
    
    print("Creating tables if they don't exist...")
    Base.metadata.create_all(bind=engine)
    
    # Create a session
    db = SessionLocal()
    try:
        # Create demo data
        create_demo_workflows(db, count=args.count)
        
        print("Demo data loaded successfully!")
    except Exception as e:
        print(f"Error loading demo data: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
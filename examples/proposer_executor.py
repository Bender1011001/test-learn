#!/usr/bin/env python3
import asyncio
import json
from camel.plugins.logging import AsyncSQLiteLoggerPlugin
from camel.agents.proposer import ProposerAgent
from camel.agents.executor import ExecutorAgent
from camel.agents.peer_reviewer import PeerReviewer

async def main():
    logger = AsyncSQLiteLoggerPlugin()
    proposer = ProposerAgent('proposer')
    executor = ExecutorAgent('executor')
    reviewer = PeerReviewer('peer_reviewer')

    state = {'goal': 'dummy goal', 'workspace': '', 'last_output': ''}

    action = await proposer.step(state)
    await logger.log('proposer', json.dumps(state), json.dumps(action))

    result = await executor.step(action)
    await logger.log('executor', json.dumps(action), json.dumps(result))

    transcript = f"STATE: {state}\nACTION: {action}\nRESULT: {result}"
    review = await reviewer.review(transcript, transcript)
    await logger.log('peer_reviewer', transcript, json.dumps(review))

    await logger.close()
    print("Logged full cycle to logs.db")

if __name__ == '__main__':
    asyncio.run(main())

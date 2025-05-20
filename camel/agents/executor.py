from camel.core.agent import BaseAgent
from typing import Dict, Any
import asyncio

class ExecutorAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name)

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        cmd = action.get('cmd', '')
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        return {
            'stdout': out.decode().strip(),
            'stderr': err.decode().strip(),
            'exit_code': proc.returncode
        }

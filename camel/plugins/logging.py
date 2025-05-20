import aiosqlite
import asyncio

class AsyncSQLiteLoggerPlugin:
    """
    Async plugin to log agent interactions into SQLite with minimal blocking.
    Buffers writes in-memory and commits periodically.
    """
    def __init__(self, db_path: str = "logs.db", commit_interval: float = 1.0):
        self.db_path = db_path
        self.commit_interval = commit_interval
        self._buffer = []
        self._lock = asyncio.Lock()
        self._task = asyncio.create_task(self._periodic_commit())
        asyncio.create_task(self._init_db())

    async def _init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                '''CREATE TABLE IF NOT EXISTS logs (
                       id INTEGER PRIMARY KEY,
                       timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                       agent TEXT,
                       input TEXT,
                       output TEXT
                   )''')
            await db.commit()

    async def log(self, agent: str, inp: str, out: str):
        async with self._lock:
            self._buffer.append((agent, inp, out))

    async def _periodic_commit(self):
        while True:
            await asyncio.sleep(self.commit_interval)
            await self._flush()

    async def _flush(self):
        async with self._lock:
            if not self._buffer:
                return
            buf, self._buffer = self._buffer, []
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                'INSERT INTO logs (agent, input, output) VALUES (?, ?, ?)', buf)
            await db.commit()

    async def close(self):
        self._task.cancel()
        await self._flush()

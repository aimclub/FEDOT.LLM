from typing import Any, Dict

from langchain_core.runnables import Runnable

from fedot_llm.output.base import BaseFedotAIOutput


class ConsoleFedotAIOutput(BaseFedotAIOutput):

    async def _chain_call(self, chain: Runnable, chain_input: Dict[str, Any]):
        async for event in chain.astream_events(chain_input, version="v2"):
            print(event)
            if event['name'] == 'master' and event['event'] == 'on_chain_end':
                return event['data']['output']

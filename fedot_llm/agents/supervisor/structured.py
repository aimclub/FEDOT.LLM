from pydantic import BaseModel, Field
from fedot_llm.agents.supervisor.state import NextAgent

class ChooseNext(BaseModel):
    next: NextAgent = Field(..., description="""The next agent to act or finish.
                                                            finish - the conversation is finished.
                                                            researcher - responsible for questions about the Fedot framework
                                                            automl - responsible for automl tasks, building ML models""")

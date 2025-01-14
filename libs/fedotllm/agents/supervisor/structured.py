from pydantic import BaseModel, Field

from fedotllm.agents.supervisor.state import NextAgent


class ChooseNext(BaseModel):
    next: NextAgent = Field(..., description="""The next agent to act or finish.
                                                            finish - the conversation is finished.
                                                            researcher - responsible for QA about the Fedot framework.
                                                            automl - responsible for automl tasks, can building machine learning models, ML pipelines""")

from pydantic import BaseModel, Field

from fedotllm.agents.supervisor.state import NextAgent


class ChooseNext(BaseModel):
    next: NextAgent = Field(..., description="""The next agent to act or finish.
                                                            finish - the conversation is finished.
                                                            automl - choose if query contains automl task, descriptions of data, needed to build machine learning models, ML pipelines, **build**
                                                            researcher - choose ONLY if query contains an EXPLICIT QUESTION about the Fedot framework.""")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from fedot_llm.ai.chains.base import BaseRunnableChain

ANALYZE_FEDOT_RESULT_TEMPLATE = ChatPromptTemplate([
    (
        'user',
        'You are an expert in ML. You always write clearly and concisely. '
        'Describe pipeline of model that you build: '
        '{parameters}'
        'Tell about the model metrics: {metrics}'
        'Explain what each metric means.'
        "Don't talk about empty values."
        "Don't talk about things that aren't in the configuration dictionary."
        "Don't make assumptions about type of scaling."
        'Use Markdown formatting.'
        'Start with: "Here is the pipeline of the model I built:"'
        '<output>'
        '# Model Pipeline\n'
        'The pipeline consists of'
        '<stages_description>'
        '</stages_description>'
        '# Model Metrics:'
        '| Metric | Value |'
        '| --- | --- |'
        '<metrics_table/>'
        'These metrics indicate that'
        '<metrics_description/>'
        '</output>'
    )
])
"""
Input:
- parameters - parameters of the model pipeline
- metrics - metrics of the model pipeline
"""


class AnalyzeFedotResultChain(BaseRunnableChain):
    """Analyze Fedot result chain
    
    Args
    ----
        model: BaseChatModel
            The llm model that will perform the analysis.
    
    Parameters
    ----------
        parameters: str
            The parameters of the model pipeline.
        metrics: str
            The metrics of the model pipeline.
            
    Returns
    -------
    str
        The analysis of the model pipeline.
    
    
    """

    def __init__(self, model: BaseChatModel):
        self.chain = (
                ANALYZE_FEDOT_RESULT_TEMPLATE
                | model
                | StrOutputParser().with_config({"tags": ["print"]})
        )

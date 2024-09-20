from langchain_core.runnables import RunnableLambda

from fedot_llm.ai.chains.base import BaseRunnableChain
from fedot_llm.data.data import Dataset


class LoadSplitDataChain(BaseRunnableChain):
    """Load split data chain
    
    Load the split data from the input.
    
    Args
    ----
        dataset: Dataset
            The dataset object.
    
    Parameters
    ----------
        split: str
            Dataset's split name
    
    Returns
    -------
    pd.DataFrame
        The split data.
    
    Examples
    --------
    >>> LoadSplitDataChain().invoke({"split": 'train'})
    pd.DataFrame
    """

    def process(self, input):
        split = list(filter(lambda split: split.name == input['split'], self.dataset.splits))[0]
        if split:
            return split.data
        else:
            raise ValueError(f"Split {input['split']} not found in the dataset.")

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.chain = (
            RunnableLambda(self.process)
        )


class CollectDatasetMetaChain(BaseRunnableChain):
    """Collect dataset meta information
    
    Collects the dataset meta information.
    
    Args
    ----
        dataset: Dataset
            The dataset object.
            
    Parameters
    ----------
    name: Optional[str]
        The name of the dataset.
    description: Optional[str]
        The short description of the dataset.
    goal: Optional[str]
        The goal of the dataset.
    train: Optional[str]
        The train split name.
    test: Optional[str]
        The test split name.
    
    Returns
    -------
    str
        The dataset meta information.
        
    
    Examples
    --------
    >>> CollectDatasetMetaChain(dataset).invoke({'name': 'dataset_name', 'description': 'This is a short dataset description', 'goal': 'This is a dataset goal', 'train': 'train', 'test': 'test'})
    'name: dataset_name\ndescription: This is a short dataset description\ngoal: This is a dataset goal\ntrain split: train\ntest split: test\nsplits:\n\nsplit: train\nmetadata: train split metadata\n\nsplit: test\nmetadata: test split metadata\n'
    """

    def process(self, input):
        name = input.get('name', None)
        description = input.get('description', None)
        goal = input.get('goal', None)
        train = input.get('train', None)
        test = input.get('test', None)
        descriptions = [
            f"name: {name}\n" if name else "",
            f"description: {description}\n" if description else "",
            f"goal: {goal}\n" if goal else "",
            f"train split: {train}\n" if train else "",
            f"test split: {test}\n" if test else "",
            "splits:\n\n"
        ]

        split_metadata = self._generate_split_metadata()
        return "".join(descriptions) + split_metadata

    def _generate_split_metadata(self) -> str:
        return "\n".join(
            f"{split.metadata_description}\n" for split in self.dataset.splits
        )

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.chain = (
                LoadSplitDataChain(dataset) | RunnableLambda(lambda x: x.columns)
        )

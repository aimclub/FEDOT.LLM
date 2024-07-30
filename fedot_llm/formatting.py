from typing import Mapping, Union

import pandas as pd


def format_dataframes(data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]]) -> str:
    """
    Formats the input dataframe or dataframes into markdown format.

    Args:
    data (pd.DataFrame, Mapping[str, pd.DataFrame]): Input data to be formatted.

    Returns:
    str: Formatted data in markdown format.
    """
    outer = "\n\n```\n{}\n```\n\n"
    if isinstance(data, pd.DataFrame):
        inner = f"{data.to_markdown(index=False)}"
    else:
        inner = "\n\n".join(
            f"### {key}\n{df.to_markdown(index=False)}"
            for key, df in data.items()
        )
    return outer.format(inner)

def filter_entities(panel_df: pd.DataFrame, basket: list):
    entity_col = panel_df.columns[0]
    # Filter rows where the value in the entity column is in the basket
    df = panel_df[panel_df[entity_col].isin(basket)]
    
    if df.empty:
        raise ValueError(f"No matching entities found in panel given basket: {basket}")

    return df

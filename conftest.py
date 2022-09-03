import pandas as pd
import pytest


@pytest.fixture
def mock_dataframe():
    return pd.DataFrame({'col_a': [1, 2, 3], 'col_b': ['red', 'white', 'blue']})

import pytest
from unittest.mock import MagicMock, patch, call, ANY 
from pathlib import Path
import pandas as pd # For mock dataframes in run_tests

from fedotllm.agents.automl.nodes import (
    problem_reflection,
    generate_automl_config,
    select_skeleton,
    generate_code, 
    insert_templates,
    evaluate,
    if_bug,
    fix_solution,
    extract_metrics,
    run_tests,
    generate_report
)
from fedotllm.agents.automl.state import AutoMLAgentState
from fedotllm.agents.automl.structured import FedotConfig
from fedotllm.llm import AIInference
from fedotllm.data import Dataset
from langgraph.types import Command
from fedotllm.enviroments import Observation 
# from fedotllm.settings.config_loader import get_settings 
from fedot.api.main import Fedot # For mocking Fedot API
from golem.core.dag.graph_utils import graph_structure # For patching


# Fixtures
@pytest.fixture
def mock_inference(mocker):
    return MagicMock(spec=AIInference)

@pytest.fixture
def mock_dataset(mocker):
    dataset = MagicMock(spec=Dataset)
    dataset.path = MagicMock(spec=Path) 
    dataset.path.absolute.return_value = "/mock/dataset/path/abs" 
    dataset.path.__str__.return_value = "/mock/dataset/path/str" 
    return dataset

@pytest.fixture
def automl_nodes_mock_workspace(mocker):
    ws_path = MagicMock(spec=Path)
    ws_path.__str__.return_value = "/mock/workspace" # Make sure it stringifies nicely
    
    # Create a proper side effect that returns mock objects instead of real Path objects
    def mock_truediv_side_effect(other):
        mock_path = MagicMock(spec=Path)
        mock_path.__str__.return_value = f"/mock/workspace/{other}"
        return mock_path
    
    ws_path.__truediv__.side_effect = mock_truediv_side_effect
    return ws_path

@pytest.fixture
def initial_state():
    return AutoMLAgentState(messages=[], description="test description") 

# Tests for problem_reflection
@patch('fedotllm.agents.automl.nodes.prompts')
def test_problem_reflection_success(mock_prompts, mock_inference, mock_dataset, initial_state):
    mock_dataset.dataset_preview.return_value = "Dataset Preview Text"
    mock_dataset.dataset_eda.return_value = "Dataset EDA Text"
    mock_prompts.automl.problem_reflection_prompt.return_value = "Generated Reflection Prompt"
    mock_inference.query.return_value = "Test Reflection Output"

    state_dict = dict(initial_state) 

    result_command = problem_reflection(state_dict, mock_inference, mock_dataset)

    mock_prompts.automl.problem_reflection_prompt.assert_called_once_with(
        data_files_and_content="Dataset Preview Text",
        dataset_eda="Dataset EDA Text"
    )
    mock_inference.query.assert_called_once_with([{'role': 'user', 'content': 'Generated Reflection Prompt'}])
    assert isinstance(result_command, Command)
    assert result_command.update == {"reflection": "Test Reflection Output"}

# Tests for generate_automl_config
@patch('fedotllm.agents.automl.nodes.prompts')
def test_generate_automl_config_success(mock_prompts, mock_inference, initial_state, mock_dataset):
    current_state_dict = dict(initial_state)
    current_state_dict["reflection"] = "Test Reflection"

    mock_prompts.automl.generate_configuration_prompt.return_value = "Generated Config Prompt"
    mock_fedot_config_instance = FedotConfig(
        problem="classification", 
        preset="fast_train", 
        metric="accuracy", 
        cv_folds=5, 
        predict_method="predict",
        timeout=1.0
    )
    mock_inference.create.return_value = mock_fedot_config_instance

    result_command = generate_automl_config(current_state_dict, mock_inference, mock_dataset) 

    mock_prompts.automl.generate_configuration_prompt.assert_called_once_with(
        reflection="Test Reflection"
    )
    mock_inference.create.assert_called_once_with("Generated Config Prompt", response_model=FedotConfig)
    assert isinstance(result_command, Command)
    assert result_command.update == {"fedot_config": mock_fedot_config_instance}

# Tests for select_skeleton
@patch('fedotllm.agents.automl.nodes.render_template')
@patch('fedotllm.agents.automl.nodes.load_template')
def test_select_skeleton_success(mock_load_template, mock_render_template, mock_dataset, initial_state, automl_nodes_mock_workspace):
    mock_fedot_config = FedotConfig(
        problem="classification", 
        preset="fast_train", 
        metric="accuracy", 
        cv_folds=5, 
        predict_method="predict", 
        timeout=1.0 
    )
    current_state_dict = dict(initial_state)
    current_state_dict["fedot_config"] = mock_fedot_config
    
    resolved_workspace_path = Path("/resolved/workspace")
    automl_nodes_mock_workspace.resolve.return_value = resolved_workspace_path

    mock_load_template.return_value = "Raw Skeleton Template Content"
    mock_render_template.return_value = "Rendered Skeleton Code String"

    result_command = select_skeleton(current_state_dict, mock_dataset, automl_nodes_mock_workspace)

    expected_template_name = "skeleton" 
    mock_load_template.assert_called_once_with(expected_template_name)
    
    mock_render_template.assert_called_once_with(
        template="Raw Skeleton Template Content", 
        dataset_path=mock_dataset.path, 
        work_dir_path=resolved_workspace_path 
    )
    assert isinstance(result_command, Command)
    assert result_command.update == {"skeleton": "Rendered Skeleton Code String"}

@patch('fedotllm.agents.automl.nodes.load_template') 
def test_select_skeleton_unknown_predict_method(mock_load_template_ignored, mock_dataset, initial_state, automl_nodes_mock_workspace): 
    mock_fedot_config = FedotConfig.model_construct( 
        problem="classification", 
        preset="fast_train", 
        metric="accuracy", 
        cv_folds=5, 
        predict_method="unknown_method", 
        timeout=1.0 
    )
    current_state_dict = dict(initial_state)
    current_state_dict["fedot_config"] = mock_fedot_config
    
    with pytest.raises(ValueError, match="Unknown predict method: unknown_method"):
        select_skeleton(current_state_dict, mock_dataset, automl_nodes_mock_workspace)

# Tests for generate_code
@patch('fedotllm.agents.automl.nodes.extract_code')
@patch('fedotllm.agents.automl.nodes.prompts')
def test_generate_code_success(mock_prompts, mock_extract_code, mock_inference, mock_dataset, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["reflection"] = "Test Reflection"
    current_state_dict["skeleton"] = "Test Skeleton"
    
    mock_prompts.automl.code_generation_prompt.return_value = "Generated CodeGen Prompt"
    mock_inference.query.return_value = "Raw LLM Code Output"
    mock_extract_code.return_value = "Extracted Python Code"

    result_command = generate_code(current_state_dict, mock_inference, mock_dataset)

    mock_prompts.automl.code_generation_prompt.assert_called_once_with(
        reflection="Test Reflection", 
        skeleton="Test Skeleton", 
        dataset_path="/mock/dataset/path/abs" 
    )
    mock_inference.query.assert_called_once_with("Generated CodeGen Prompt")
    mock_extract_code.assert_called_once_with("Raw LLM Code Output")
    assert isinstance(result_command, Command)
    assert result_command.update == {"raw_code": "Extracted Python Code"}

# Tests for insert_templates
@patch('fedotllm.agents.automl.nodes.render_template')
@patch('fedotllm.agents.automl.nodes.load_template')
def test_insert_templates_success(mock_load_template, mock_render_template, initial_state):
    mock_fedot_config = FedotConfig(
        problem="classification", 
        preset="fast_train", 
        metric="accuracy", 
        cv_folds=5, 
        predict_method="predict", 
        timeout=1.0
    )
    current_state_dict = dict(initial_state)
    current_state_dict["raw_code"] = "from automl import train_model, evaluate_model, automl_predict\n# Rest of code"
    current_state_dict["fedot_config"] = mock_fedot_config

    def load_template_side_effect(template_name):
        if template_name == "fedot_train.py": return "Raw Template: fedot_train.py"
        if template_name == "fedot_evaluate.py": return "Raw Template: fedot_evaluate.py"
        if template_name == "fedot_predict.py": return "Raw Template: fedot_predict.py"
        return None 
    mock_load_template.side_effect = load_template_side_effect
    
    render_calls = []
    def render_template_side_effect(template, **params):
        rendered_output = f"Rendered: {template} with {params}"
        render_calls.append(call(template=template, **params))
        return rendered_output
    mock_render_template.side_effect = render_template_side_effect
    

    result_command = insert_templates(current_state_dict)

    expected_load_calls = [
        call("fedot_train.py"),
        call("fedot_evaluate.py"),
        call("fedot_predict.py")
    ]
    mock_load_template.assert_has_calls(expected_load_calls, any_order=False)

    assert len(render_calls) == 3
    assert render_calls[0].kwargs['template'] == "Raw Template: fedot_train.py"
    assert render_calls[0].kwargs['problem'] == str(mock_fedot_config.problem) 
    assert render_calls[1].kwargs['template'] == "Raw Template: fedot_evaluate.py"
    assert render_calls[1].kwargs['predict_method'] == "predict(features=input_data)"
    assert render_calls[2].kwargs['template'] == "Raw Template: fedot_predict.py"
    assert render_calls[2].kwargs['predict_method'] == "predict(features=input_data)"

    expected_code = "\n".join([
        "Rendered: Raw Template: fedot_train.py with {'problem': 'TaskTypesEnum.classification', 'timeout': 1.0, 'cv_folds': 5, 'preset': \"'fast_train'\", 'metric': \"'accuracy'\"}",
        "Rendered: Raw Template: fedot_evaluate.py with {'problem': 'TaskTypesEnum.classification', 'predict_method': 'predict(features=input_data)'}",
        "Rendered: Raw Template: fedot_predict.py with {'problem': 'TaskTypesEnum.classification', 'predict_method': 'predict(features=input_data)'}"
    ]) + "\n# Rest of code"
    
    assert isinstance(result_command, Command)
    assert result_command.update == {"code": expected_code}

@patch('fedotllm.agents.automl.nodes.render_template')
@patch('fedotllm.agents.automl.nodes.load_template')
def test_insert_templates_exception_handling(mock_load_template, mock_render_template, initial_state):
    mock_fedot_config = FedotConfig(
        problem="classification", 
        preset="fast_train", 
        metric="accuracy", 
        cv_folds=5, 
        predict_method="predict", 
        timeout=1.0
    )
    current_state_dict = dict(initial_state)
    current_state_dict["raw_code"] = "Some code"
    current_state_dict["fedot_config"] = mock_fedot_config
    
    mock_load_template.side_effect = Exception("Template loading failed") 
    
    result_command = insert_templates(current_state_dict)
    
    assert isinstance(result_command, Command)
    assert result_command.update == {"code": None}

# Tests for evaluate
@patch('fedotllm.agents.automl.nodes.execute_code')
@patch('fedotllm.agents.automl.nodes._generate_code_file')
def test_evaluate_success(mock_generate_code_file, mock_execute_code, automl_nodes_mock_workspace, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["code"] = "valid python code"
    
    mock_code_file_path = MagicMock(spec=Path)
    mock_generate_code_file.return_value = mock_code_file_path
    
    mock_observation = Observation(stdout="Run successful", stderr="", error=False, msg="OK")
    mock_execute_code.return_value = mock_observation
    
    result_command = evaluate(current_state_dict, automl_nodes_mock_workspace)
    
    mock_generate_code_file.assert_called_once_with("valid python code", automl_nodes_mock_workspace)
    mock_execute_code.assert_called_once_with(path_to_run_code=mock_code_file_path)
    assert isinstance(result_command, Command)
    assert result_command.update == {"observation": mock_observation}

@patch('fedotllm.agents.automl.nodes.execute_code')
@patch('fedotllm.agents.automl.nodes._generate_code_file')
def test_evaluate_execution_error(mock_generate_code_file, mock_execute_code, automl_nodes_mock_workspace, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["code"] = "buggy python code"
    
    mock_code_file_path = MagicMock(spec=Path)
    mock_generate_code_file.return_value = mock_code_file_path
    
    mock_observation = Observation(stdout="", stderr="Syntax Error", error=True, msg="Error")
    mock_execute_code.return_value = mock_observation
    
    result_command = evaluate(current_state_dict, automl_nodes_mock_workspace)
    
    mock_generate_code_file.assert_called_once_with("buggy python code", automl_nodes_mock_workspace)
    mock_execute_code.assert_called_once_with(path_to_run_code=mock_code_file_path)
    assert isinstance(result_command, Command)
    assert result_command.update == {"observation": mock_observation}

# Tests for if_bug
@patch('fedotllm.agents.automl.nodes.get_settings')
def test_if_bug_is_true(mock_get_settings, initial_state):
    mock_settings_instance = MagicMock()
    mock_settings_instance.config.fix_tries = 3
    mock_get_settings.return_value = mock_settings_instance
    
    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(error=True, msg="Error occurred", stdout="", stderr="Traceback...")
    current_state_dict["fix_attempts"] = 1
    
    assert if_bug(current_state_dict) is True

@patch('fedotllm.agents.automl.nodes.get_settings')
def test_if_bug_is_false_no_error(mock_get_settings, initial_state):
    mock_settings_instance = MagicMock()
    mock_settings_instance.config.fix_tries = 3
    mock_get_settings.return_value = mock_settings_instance
    
    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(error=False, msg="OK", stdout="Success", stderr="")
    current_state_dict["fix_attempts"] = 1
    
    assert if_bug(current_state_dict) is False

@patch('fedotllm.agents.automl.nodes.get_settings')
def test_if_bug_is_false_max_attempts_reached(mock_get_settings, initial_state):
    mock_settings_instance = MagicMock()
    mock_settings_instance.config.fix_tries = 3
    mock_get_settings.return_value = mock_settings_instance
    
    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(error=True, msg="Error occurred", stdout="", stderr="Traceback...")
    current_state_dict["fix_attempts"] = 3 
    
    assert if_bug(current_state_dict) is False

# Tests for fix_solution
@patch('fedotllm.agents.automl.nodes.extract_code')
@patch('fedotllm.agents.automl.nodes.prompts')
def test_fix_solution_success(mock_prompts, mock_extract_code, mock_inference, mock_dataset, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["reflection"] = "Test Reflection"
    current_state_dict["raw_code"] = "buggy code"
    current_state_dict["observation"] = Observation(stdout="out", stderr="err", error=True, msg="Error details")
    current_state_dict["fix_attempts"] = 0
            
    mock_prompts.automl.fix_solution_prompt.return_value = "Generated Fix Prompt"
    mock_inference.query.return_value = "Raw Fixed Code Output"
    mock_extract_code.return_value = "Extracted Fixed Code"

    result_command = fix_solution(current_state_dict, mock_inference, mock_dataset)

    mock_prompts.automl.fix_solution_prompt.assert_called_once_with(
        reflection="Test Reflection",
        dataset_path="/mock/dataset/path/abs", 
        code_recent_solution="buggy code",
        msg="Error details", 
        stderr="err",
        stdout="out"
    )
    mock_inference.query.assert_called_once_with("Generated Fix Prompt")
    mock_extract_code.assert_called_once_with("Raw Fixed Code Output")
    assert isinstance(result_command, Command)
    assert result_command.update == {"raw_code": "Extracted Fixed Code", "fix_attempts": 1}

# Tests for extract_metrics
@patch('fedotllm.agents.automl.nodes.graph_structure')
@patch('fedotllm.agents.automl.nodes.Fedot')
def test_extract_metrics_success(mock_fedot_api, mock_graph_structure, automl_nodes_mock_workspace, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(stdout="Model metrics: {'accuracy': 0.95}", error=False, msg="OK")
    
    # Configure the workspace mock to return a consistent pipeline path
    mock_pipeline_path = MagicMock(spec=Path)
    mock_pipeline_path.exists.return_value = True  # Return actual boolean True
    automl_nodes_mock_workspace.__truediv__.return_value = mock_pipeline_path
    
    mock_fedot_instance = mock_fedot_api.return_value
    mock_pipeline_obj = MagicMock()
    mock_fedot_instance.current_pipeline = mock_pipeline_obj
    mock_graph_structure.return_value = "pipeline_structure_str"
    
    updated_state = extract_metrics(current_state_dict, automl_nodes_mock_workspace)
    
    assert updated_state["metrics"] == "{'accuracy': 0.95}"
    mock_fedot_api.assert_called_once_with(problem="classification")
    # Check that load was called once with any Path-like object
    assert mock_fedot_instance.load.call_count == 1
    # Check that the argument passed to load has the Path spec
    load_call_args = mock_fedot_instance.load.call_args[0]
    assert len(load_call_args) == 1
    assert hasattr(load_call_args[0], '_spec_class') or str(load_call_args[0].__class__) == "<class 'unittest.mock.MagicMock'>"
    mock_graph_structure.assert_called_once_with(mock_pipeline_obj)
    assert updated_state["pipeline"] == "pipeline_structure_str"

def test_extract_metrics_no_metrics_found(initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(stdout="No metrics here", error=False, msg="OK")
    
    # Create a real Path object that doesn't exist
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        # Don't create the pipeline file, so it won't exist
        updated_state = extract_metrics(current_state_dict, workspace)
        
    assert updated_state["metrics"] == "Metrics not found"
    assert updated_state["pipeline"] == "Pipeline not found"

def test_extract_metrics_pipeline_not_found(initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(stdout="Model metrics: {'accuracy': 0.9}", error=False, msg="OK")
    
    # Create a real Path object that doesn't exist
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        # Don't create the pipeline file, so it won't exist
        updated_state = extract_metrics(current_state_dict, workspace)
    
    assert updated_state["metrics"] == "{'accuracy': 0.9}"
    assert updated_state["pipeline"] == "Pipeline not found"

@patch('fedotllm.agents.automl.nodes.Fedot')
def test_extract_metrics_exception_handling(mock_fedot_api, automl_nodes_mock_workspace, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(stdout="Model metrics: {'accuracy': 0.9}", error=False, msg="OK")
    
    # Configure the workspace mock to raise an exception when performing __truediv__ operation
    automl_nodes_mock_workspace.__truediv__.side_effect = Exception("Test Read Error")
    
    updated_state = extract_metrics(current_state_dict, automl_nodes_mock_workspace)
    
    assert updated_state["metrics"] == "Metrics not found"
    assert updated_state["pipeline"] == "Pipeline not found"
    mock_fedot_api.assert_not_called()

# Tests for run_tests
@patch('fedotllm.agents.automl.nodes.graph_structure')
@patch('fedotllm.agents.automl.nodes.Fedot')
@patch('fedotllm.agents.automl.nodes.pd.read_csv')
def test_run_tests_all_pass(mock_pd_read_csv, mock_fedot_api_rt, mock_graph_structure_rt, 
                           mock_inference, automl_nodes_mock_workspace, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(
        stdout="Model metrics: {'roc_auc': 0.8}\nSample Submission File: /path/to/sample_submission.csv", 
        error=False, msg="OK"
    )

    pipeline_file_mock = MagicMock(spec=Path)
    pipeline_file_mock.exists.return_value = True
    submission_file_mock = MagicMock(spec=Path)
    submission_file_mock.exists.return_value = True
    submission_file_mock.suffix = ".csv"

    def truediv_side_effect(other):
        if other == "pipeline": return pipeline_file_mock
        if other == "submission.csv": return submission_file_mock
        # Need to return a mock that can be used in os.fspath for sample_path
        # However, pd.read_csv will be mocked, so the actual path string for sample submission won't be opened by os.
        # The mock for pd.read_csv needs to handle the string "/path/to/sample_submission.csv"
        return MagicMock(spec=Path) # Default for other paths like /path/to/
    automl_nodes_mock_workspace.__truediv__.side_effect = truediv_side_effect
    
    mock_fedot_instance = mock_fedot_api_rt.return_value
    mock_fedot_instance.current_pipeline = MagicMock()
    mock_graph_structure_rt.return_value = "pipeline_structure_str"
    
    # Mock pd.read_csv for the two expected calls:
    # 1. Sample submission: pd.read_csv("/path/to/sample_submission.csv")
    # 2. Actual submission: pd.read_csv(automl_nodes_mock_workspace / "submission.csv")
    mock_pd_read_csv.side_effect = [
        pd.DataFrame({'col1': [1], 'col2': [2]}), # Mock for sample_submission.csv
        pd.DataFrame({'col1': [3], 'col2': [4]})  # Mock for submission.csv
    ]
    mock_inference.query.return_value = "true" 

    updated_state = run_tests(current_state_dict, automl_nodes_mock_workspace, mock_inference)
    
    assert updated_state["observation"].error is False
    assert "Metrics not found" not in updated_state["observation"].msg
    assert "Pipeline not found" not in updated_state["observation"].msg
    assert "Submission file not found" not in updated_state["observation"].msg
    assert "Submission file format is correct." in updated_state["observation"].msg


@patch('fedotllm.agents.automl.nodes.graph_structure')
@patch('fedotllm.agents.automl.nodes.Fedot')
@patch('fedotllm.agents.automl.nodes.pd.read_csv')
def test_run_tests_metrics_fail(mock_pd_read_csv, mock_fedot_api_rt, mock_graph_structure_rt,
                                mock_inference, automl_nodes_mock_workspace, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(stdout="No metrics here", error=False, msg="OK")
    # Ensure other checks don't fail before metrics check
    (automl_nodes_mock_workspace / "pipeline").exists.return_value = True
    submission_file_mock = MagicMock(spec=Path); submission_file_mock.exists.return_value = True; submission_file_mock.suffix = ".csv"
    automl_nodes_mock_workspace.__truediv__.return_value = submission_file_mock # Make sure / operator returns this mock
    mock_pd_read_csv.side_effect = [pd.DataFrame({'col1': [1]}), pd.DataFrame({'col1': [1]})]
    mock_inference.query.return_value = "true"


    updated_state = run_tests(current_state_dict, automl_nodes_mock_workspace, mock_inference)
    assert updated_state["observation"].error is True
    assert "Metrics not found" in updated_state["observation"].msg

@patch('fedotllm.agents.automl.nodes.graph_structure')
@patch('fedotllm.agents.automl.nodes.Fedot')
@patch('fedotllm.agents.automl.nodes.pd.read_csv')
def test_run_tests_submission_file_not_found(mock_pd_read_csv, mock_fedot_api_rt, mock_graph_structure_rt,
                                             mock_inference, automl_nodes_mock_workspace, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(
        stdout="Model metrics: {'roc_auc': 0.8}\nSample Submission File: /path/to/sample_submission.csv", 
        error=False, msg="OK"
    )
    (automl_nodes_mock_workspace / "pipeline").exists.return_value = True
    submission_file_mock = MagicMock(spec=Path)
    submission_file_mock.exists.return_value = False # Submission not found
    
    # Ensure that when workspace / "submission.csv" is called, it returns our submission_file_mock
    def truediv_side_effect_sub_not_found(other):
        if other == "pipeline":
            pipeline_mock = MagicMock(spec=Path); pipeline_mock.exists.return_value = True; return pipeline_mock
        if other == "submission.csv":
            return submission_file_mock
        return MagicMock(spec=Path)
    automl_nodes_mock_workspace.__truediv__.side_effect = truediv_side_effect_sub_not_found
    
    mock_pd_read_csv.return_value = pd.DataFrame({'col1': [1]}) 
    mock_inference.query.return_value = "true"


    updated_state = run_tests(current_state_dict, automl_nodes_mock_workspace, mock_inference)
    assert updated_state["observation"].error is True
    assert "Submission file not found" in updated_state["observation"].msg


@patch('fedotllm.agents.automl.nodes.graph_structure')
@patch('fedotllm.agents.automl.nodes.Fedot')
@patch('fedotllm.agents.automl.nodes.pd.read_csv')
def test_run_tests_submission_format_fail_llm(mock_pd_read_csv, mock_fedot_api_rt, mock_graph_structure_rt,
                                             mock_inference, automl_nodes_mock_workspace, initial_state):
    current_state_dict = dict(initial_state)
    current_state_dict["observation"] = Observation(
        stdout="Model metrics: {'roc_auc': 0.8}\nSample Submission File: /path/to/sample_submission.csv",
        error=False, msg="OK"
    )
    
    # Setup pipeline mock
    pipeline_mock = MagicMock(spec=Path)
    pipeline_mock.exists.return_value = True
    
    # Setup submission file mock
    submission_file_mock = MagicMock(spec=Path)
    submission_file_mock.exists.return_value = True
    submission_file_mock.suffix = ".csv"
    
    # Configure workspace mock to return different objects for different paths
    def truediv_side_effect(other):
        if other == "pipeline":
            return pipeline_mock
        elif other == "submission.csv":
            return submission_file_mock
        return MagicMock(spec=Path)
    
    automl_nodes_mock_workspace.__truediv__.side_effect = truediv_side_effect
    
    # Mock Fedot and graph_structure properly
    mock_fedot_instance = mock_fedot_api_rt.return_value
    mock_fedot_instance.current_pipeline = MagicMock()
    mock_graph_structure_rt.return_value = "test_pipeline_structure"

    mock_pd_read_csv.side_effect = [pd.DataFrame({'col1': [1]}), pd.DataFrame({'col1': [1]})]
    mock_inference.query.return_value = "false" # LLM says format is bad

    updated_state = run_tests(current_state_dict, automl_nodes_mock_workspace, mock_inference)
    assert updated_state["observation"].error is True
    assert "Submission file format does not match" in updated_state["observation"].msg
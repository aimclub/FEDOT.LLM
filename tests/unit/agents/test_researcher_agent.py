from unittest.mock import MagicMock, call, partial, patch

import pytest

from fedotllm.agents.researcher.researcher import ResearcherAgent
from fedotllm.agents.researcher.state import ResearcherAgentState
from fedotllm.llm import AIInference, LiteLLMEmbeddings


# Fixtures
@pytest.fixture
def mock_inference(mocker):
    return MagicMock(spec=AIInference)


@pytest.fixture
def mock_embeddings(mocker):
    return MagicMock(spec=LiteLLMEmbeddings)


@patch("fedotllm.agents.researcher.researcher.is_useful")
@patch("fedotllm.agents.researcher.researcher.is_grounded")
@patch("fedotllm.agents.researcher.researcher.is_continue")
@patch("fedotllm.agents.researcher.researcher.rewrite_question")
@patch("fedotllm.agents.researcher.researcher.render_answer")
@patch("fedotllm.agents.researcher.researcher.generate_response")
@patch("fedotllm.agents.researcher.researcher.grade_retrieve")
@patch("fedotllm.agents.researcher.researcher.retrieve_documents")
@patch(
    "fedotllm.agents.researcher.researcher.END",
    new_callable=lambda: "END_NODE_RESEARCHER",
)
@patch(
    "fedotllm.agents.researcher.researcher.START",
    new_callable=lambda: "START_NODE_RESEARCHER",
)
@patch("fedotllm.agents.researcher.researcher.StateGraph")
class TestResearcherAgent:
    def test_researcher_agent_init(
        self,
        mock_sg_ignored,
        mock_start_ignored,
        mock_end_ignored,
        mock_rd_ignored,
        mock_gr_ignored,
        mock_gen_resp_ignored,
        mock_ra_ignored,
        mock_rq_ignored,
        mock_ic_ignored,
        mock_ig_ignored,
        mock_iu_ignored,
        mock_inference,
        mock_embeddings,
    ):
        agent = ResearcherAgent(inference=mock_inference, embeddings=mock_embeddings)
        assert agent.inference == mock_inference
        assert agent.embeddings == mock_embeddings

    def test_researcher_agent_create_graph(
        self,
        mock_state_graph_constructor,
        mock_start_node_str,
        mock_end_node_str,
        mock_retrieve_documents_node,
        mock_grade_retrieve_node,
        mock_generate_response_node,
        mock_render_answer_node,
        mock_rewrite_question_node,
        mock_is_continue_func,
        mock_is_grounded_func,
        mock_is_useful_func,
        mock_inference,
        mock_embeddings,
    ):
        mock_workflow_instance = MagicMock(name="StateGraphInstance")
        mock_state_graph_constructor.return_value = mock_workflow_instance

        mock_compiled_graph_final = MagicMock(name="FinalCompiledGraphWithConfig")
        mock_compiled_workflow = MagicMock(name="InitialCompiledGraph")
        mock_workflow_instance.compile.return_value = mock_compiled_workflow
        mock_compiled_workflow.with_config.return_value = mock_compiled_graph_final

        agent = ResearcherAgent(inference=mock_inference, embeddings=mock_embeddings)
        compiled_graph = agent.create_graph()

        mock_state_graph_constructor.assert_called_once_with(ResearcherAgentState)

        # Assert add_node calls
        add_node_calls = mock_workflow_instance.add_node.call_args_list

        # Check retrieve node
        # partial(retrieve_documents, embeddings=self.embeddings)
        entry = next(c for c in add_node_calls if c[0][0] == "retrieve")
        assert isinstance(entry[0][1], partial)
        assert entry[0][1].func == mock_retrieve_documents_node
        assert entry[0][1].keywords.get("embeddings") == mock_embeddings

        # Check retrieve_grader node
        # partial(grade_retrieve, inference=self.inference)
        entry = next(c for c in add_node_calls if c[0][0] == "retrieve_grader")
        assert isinstance(entry[0][1], partial)
        assert entry[0][1].func == mock_grade_retrieve_node
        assert entry[0][1].keywords.get("inference") == mock_inference

        # Check generate node
        # partial(generate_response, inference=self.inference)
        entry = next(c for c in add_node_calls if c[0][0] == "generate")
        assert isinstance(entry[0][1], partial)
        assert entry[0][1].func == mock_generate_response_node
        assert entry[0][1].keywords.get("inference") == mock_inference

        # Check render_answer node
        assert call("render_answer", mock_render_answer_node) in add_node_calls

        # Check rewrite_question node
        # partial(rewrite_question, inference=self.inference)
        entry = next(c for c in add_node_calls if c[0][0] == "rewrite_question")
        assert isinstance(entry[0][1], partial)
        assert entry[0][1].func == mock_rewrite_question_node
        assert entry[0][1].keywords.get("inference") == mock_inference

        # Assert add_edge calls
        mock_workflow_instance.add_edge.assert_any_call(mock_start_node_str, "retrieve")
        mock_workflow_instance.add_edge.assert_any_call("retrieve", "retrieve_grader")
        mock_workflow_instance.add_edge.assert_any_call("rewrite_question", "retrieve")
        mock_workflow_instance.add_edge.assert_any_call(
            "render_answer", mock_end_node_str
        )

        # Assert add_conditional_edges calls
        conditional_edge_calls = (
            mock_workflow_instance.add_conditional_edges.call_args_list
        )

        # First conditional edge (for "retrieve_grader")
        assert conditional_edge_calls[0][0][0] == "retrieve_grader"
        assert callable(conditional_edge_calls[0][0][1])
        assert conditional_edge_calls[0][0][2] == {
            True: "generate",
            False: "rewrite_question",
        }

        # Second conditional edge (for "generate")
        assert conditional_edge_calls[1][0][0] == "generate"
        assert callable(conditional_edge_calls[1][0][1])
        assert conditional_edge_calls[1][0][2] == {
            True: "render_answer",
            False: "generate",
        }

        mock_workflow_instance.compile.assert_called_once()
        mock_compiled_workflow.with_config.assert_called_once_with(
            config={"run_name": "ResearcherAgent"}
        )
        assert compiled_graph == mock_compiled_workflow.with_config.return_value

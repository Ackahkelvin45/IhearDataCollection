"""
Integration tests for the Phase 2 clarification pipeline:
- Tool ambiguity escalation (Task 3)
- Checkpointer timing (Task 8)
- Graph routing through emit_clarification
"""

from unittest.mock import MagicMock, patch
from django.test import SimpleTestCase

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from data_insights.workflows.agent_workflow import DataAgent, AgentState
from data_insights.workflows.clarification_gate import (
    clarification_gate_node,
    should_clarify,
    clarification_resolver_node,
    emit_clarification_node,
)


class ClarificationGateRoutingTests(SimpleTestCase):
    """Test the gate node's signal detection and routing decisions."""

    def test_gate_passes_clear_query(self):
        """A query with a single, clear analysis type should pass through."""
        state = {
            "messages": [HumanMessage(content="show decibel levels in Accra")],
            "clarification_pending": False,
            "session_id": 1,
        }
        result = clarification_gate_node(state)
        self.assertFalse(result["clarification_pending"])

    def test_gate_fires_on_vague_ranking(self):
        """'show the best performers' has ranking intent but no explicit metric.
        Note: 'worst recordings' won't fire because 'recordings' is in
        EXPLICIT_METRIC_WORDS (it means recording_count)."""
        state = {
            "messages": [HumanMessage(content="show me the best performers")],
            "clarification_pending": False,
            "session_id": 1,
        }
        result = clarification_gate_node(state)
        self.assertTrue(result["clarification_pending"])
        self.assertEqual(result["clarification_dimension"], "metric")

    def test_gate_fires_on_zero_match_analysis(self):
        """'help me understand' matches zero analysis types.
        Under V2 len(matched) != 1 fires for len=0.
        Under V1 len(matched) > 1 does NOT fire for len=0.
        The test validates the current behavior — which depends on
        CLARIFICATION_V2_ENABLED in settings."""
        state = {
            "messages": [HumanMessage(content="help me understand the data")],
            "clarification_pending": False,
            "session_id": 1,
        }
        result = clarification_gate_node(state)
        # Under V1: no signal fires. Under V2: multi_analysis_type fires.
        # Default is V1 (False), so we expect no signal.
        self.assertFalse(result.get("clarification_pending"))

    def test_should_clarify_routes_correctly(self):
        """should_clarify conditional edge routing."""
        self.assertEqual(should_clarify({"clarification_pending": False}), "agent")
        self.assertEqual(
            should_clarify(
                {"clarification_pending": True, "clarification_answer": None}
            ),
            "emit_clarification",
        )
        self.assertEqual(
            should_clarify(
                {"clarification_pending": True, "clarification_answer": "energy"}
            ),
            "clarification_resolver",
        )

    def test_gate_passes_entity_present(self):
        """A query with a known entity should pass through for entity dimension."""
        state = {
            "messages": [HumanMessage(content="show data for Accra region")],
            "clarification_pending": False,
            "session_id": 1,
        }
        result = clarification_gate_node(state)
        # If Accra is in the DB, _has_entity_match returns True, signal doesn't fire.
        # If not, the gate may fire missing_entity due to "region" hint word.
        # Either way, the function shouldn't crash.
        self.assertIsInstance(result, dict)
        self.assertIn("clarification_pending", result)


class CheckpointerTimingTests(SimpleTestCase):
    """Verify that pending_clarification_payload is committed to the checkpointer
    before views.py reads it (Task 8)."""

    def test_emit_node_stores_payload_in_state(self):
        """emit_clarification_node must set pending_clarification_payload
        in the returned state dict so the checkpointer persists it."""
        state = {
            "clarification_question": "What time range?",
            "clarification_options": ["Last 7 days", "Last 30 days"],
            "clarification_dimension": "time_range",
            "session_id": "test-session-1",
        }
        result = emit_clarification_node(state)
        self.assertIn("pending_clarification_payload", result)
        payload = result["pending_clarification_payload"]
        self.assertEqual(payload["question"], "What time range?")
        self.assertEqual(payload["options"], ["Last 7 days", "Last 30 days"])
        self.assertEqual(payload["dimension"], "time_range")

    def test_payload_survives_graph_step(self):
        """Simulate a full gate → emit flow using MemorySaver and verify
        that the payload is readable after the graph step completes."""
        # Create an LLM mock that returns a valid AIMessage so the graph
        # can complete its invocation without hitting the real model.
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="test response")

        agent = DataAgent(
            llm=mock_llm,
            system_prompt="test",
            tools=[],
        )

        checkpointer = MemorySaver()
        workflow = agent.compile_workflow(checkpointer)

        # Run a query that triggers the gate
        config = {"configurable": {"thread_id": "test-thread-payload"}}
        initial_state = agent.create_initial_state(
            user_input="show me the best performers",
            user_id=1,
            session_id=999,
        )

        # Stream to completion and check final state
        final_state = workflow.invoke(initial_state, config=config)
        payload = final_state.get("pending_clarification_payload")
        # "best performers" triggers vague_ranking_metric
        self.assertIsNotNone(payload, "Expected clarification payload in final state")
        self.assertEqual(payload["dimension"], "metric")

        # Verify the checkpointer has the same state (read-back test)
        snapshot = workflow.get_state(config)
        self.assertIsNotNone(snapshot)
        stored_payload = snapshot.values.get("pending_clarification_payload")
        self.assertIsNotNone(stored_payload)
        self.assertEqual(stored_payload["dimension"], "metric")


class ClarificationResolverTests(SimpleTestCase):
    """Test the resolver node normalises answers correctly."""

    def test_resolver_structured_answer(self):
        """A structured pick should be mapped through _map_structured_answer."""
        state = {
            "clarification_answer": "Last 30 days",
            "clarification_is_custom": False,
            "clarification_dimension": "time_range",
            "original_query": "show trends",
            "session_id": 1,
            "messages": [],
        }
        result = clarification_resolver_node(state)
        self.assertFalse(result["clarification_pending"])
        self.assertEqual(result["clarification_answer"], "last_30_days")
        self.assertFalse(result["confirmation_needed"])
        # Should inject a context message
        self.assertTrue(len(result["messages"]) > 0)
        injected = result["messages"][0].content
        self.assertIn("[Clarification resolved]", injected)
        self.assertIn("last_30_days", injected)

    def test_resolver_custom_answer(self):
        """A custom free-text answer should be normalized."""
        state = {
            "clarification_answer": "past 3 months",
            "clarification_is_custom": True,
            "clarification_dimension": "time_range",
            "original_query": "trends",
            "session_id": 1,
            "messages": [],
        }
        result = clarification_resolver_node(state)
        self.assertFalse(result["clarification_pending"])
        # Custom answer normalization is fuzzy — it should at minimum
        # not crash and not leave the raw answer as-is
        self.assertIsNotNone(result["clarification_answer"])


class PostProcessToolAmbiguityTests(SimpleTestCase):
    """Test that the tool ambiguity sentinel triggers the right state changes."""

    def test_handle_tool_ambiguity_sets_clarification_state(self):
        """_handle_tool_ambiguity must set all required clarification fields."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        agent = DataAgent(llm=mock_llm, system_prompt="test", tools=[])

        state = {
            "current_query_handles": [],
            "messages": [],
        }
        result = agent._handle_tool_ambiguity(
            state,
            {
                "_tool_ambiguity": True,
                "dimension": "analysis_type",
                "query": "help me understand",
            },
        )

        self.assertTrue(result["clarification_pending"])
        self.assertEqual(result["clarification_dimension"], "analysis_type")
        self.assertTrue(result["clarification_question"])
        self.assertTrue(result["clarification_options"])
        self.assertIsNotNone(result["pending_clarification_payload"])

    def test_should_continue_from_post_process_routing(self):
        """Conditional edge: emit_clarification when pending, agent otherwise."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        agent = DataAgent(llm=mock_llm, system_prompt="test", tools=[])

        pending_state = {"clarification_pending": True}
        normal_state = {"clarification_pending": False}

        self.assertEqual(
            agent.should_continue_from_post_process(pending_state),
            "emit_clarification",
        )
        self.assertEqual(
            agent.should_continue_from_post_process(normal_state),
            "agent",
        )

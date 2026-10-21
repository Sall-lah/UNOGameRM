"""Tests for MCTS determinization constraint helper."""

from src.memory.graph import PropositionNode
from src.memory.plugmem import UNOPlugMem


def test_determinization_constraint_helper_emits_typed_constraints():
    """Verify helper outputs PlugMemConstraint objects from active propositions."""
    memory = UNOPlugMem()
    memory.reset_for_new_round(round_id=1)

    # Active lack-colour proposition
    prop_lack = PropositionNode(
        node_id="prop_r1_0",
        round_id=1,
        origin_turn=1,
        statement="Opponent lacks red.",
        category="OPPONENT_DRAW_CONSTRAINT",
        concept_tags=("constraint:opp_lacks_colour", "colour:red"),
        confidence=0.85,
        active=True,
        metadata={"lacked_colour": "red", "lacked_value": "5"},
    )
    # Active wild proposition
    prop_wild = PropositionNode(
        node_id="prop_r1_1",
        round_id=1,
        origin_turn=2,
        statement="Opponent declared blue.",
        category="WILD_COLOUR_DECLARATION",
        concept_tags=("constraint:opp_declared_wild", "colour:blue"),
        confidence=0.90,
        active=True,
        metadata={"declared_colour": "blue"},
    )
    # Inactive proposition (should be ignored)
    prop_inactive = PropositionNode(
        node_id="prop_r1_2",
        round_id=1,
        origin_turn=1,
        statement="Opponent lacks green (inactive).",
        category="OPPONENT_DRAW_CONSTRAINT",
        concept_tags=("constraint:opp_lacks_colour", "colour:green"),
        confidence=0.0,
        active=False,
        metadata={"lacked_colour": "green"},
    )

    memory.graph.add_proposition(prop_lack)
    memory.graph.add_proposition(prop_wild)
    memory.graph.add_proposition(prop_inactive)

    constraints = memory.get_determinization_constraints()

    types = {c.constraint_type for c in constraints}
    assert "LACKS_COLOR" in types
    assert "LACKS_VALUE" in types
    assert "PREFERRED_WILD_COLOR" in types

    # Ensure inactive green is omitted
    target_keys = {c.target_key for c in constraints}
    assert "green" not in target_keys
    assert "red" in target_keys
    assert "blue" in target_keys

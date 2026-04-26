"""
Unit Tests — GuardianAgent
===========================
Tests the SYSTEM_PROMPT, _parse(), consistency checker, build_training_prompt,
and all three parse fallback paths. No model weights required.

Run:
    pytest guardian/tests/test_guardian_agent.py -v
"""

import pytest
from guardian.agents.guardian_agent import (
    GuardianAgent,
    SYSTEM_PROMPT,
    VALID_INTERVENTIONS,
    VALID_ATTACK_TYPES,
    _LOW_RISK_BANNED,
    _HIGH_RISK_BANNED,
)


@pytest.fixture
def agent():
    return GuardianAgent()


# ─── SYSTEM_PROMPT content ────────────────────────────────────────────────────

class TestSystemPrompt:
    def test_has_quarantine_agent(self):
        """GAP 1: quarantine_agent must be in SYSTEM_PROMPT."""
        assert "quarantine_agent" in SYSTEM_PROMPT

    def test_has_rogue_internal_ai_rule(self):
        """GAP 1: rogue_internal_ai must have a specific rule."""
        assert "rogue_internal_ai" in SYSTEM_PROMPT

    def test_has_predicted_next_risk(self):
        """GAP 7: predicted_next_risk must be in output format."""
        assert "predicted_next_risk" in SYSTEM_PROMPT

    def test_has_delayed_exfiltration(self):
        """GAP 10: delayed_exfiltration in attack types."""
        assert "delayed_exfiltration" in SYSTEM_PROMPT

    def test_has_social_engineering(self):
        """GAP 10: social_engineering in attack types."""
        assert "social_engineering" in SYSTEM_PROMPT

    def test_has_consistency_rules(self):
        """GAP 9: consistency rules must be documented."""
        assert "Consistency Rules" in SYSTEM_PROMPT

    def test_valid_interventions_has_quarantine_agent(self):
        assert "quarantine_agent" in VALID_INTERVENTIONS

    def test_valid_attack_types_complete(self):
        expected = {
            "authority_spoofing", "prompt_injection", "approval_bypass",
            "data_exfiltration", "confused_deputy", "approval_laundering",
            "salami_slicing", "schema_drift_exploit", "rogue_internal_ai",
            "delayed_exfiltration", "social_engineering", "clean",
        }
        assert expected.issubset(VALID_ATTACK_TYPES)


# ─── _parse() primary path ────────────────────────────────────────────────────

class TestParsePrimaryPath:
    XML_ATTACK = (
        "<reasoning>Step 2: process_payment called with amount=999999 "
        "and bypass_approval=True. Normal threshold is 10000.</reasoning>"
        "<risk_score>0.92</risk_score>"
        "<predicted_next_risk>0.95</predicted_next_risk>"
        "<intervention>emergency_fork</intervention>"
        "<attack_type>authority_spoofing</attack_type>"
        "<explanation>process_payment bypassed approval with $999,999.</explanation>"
    )

    XML_CLEAN = (
        "<reasoning>Step 1: read_db record_001 by user_0 — normal audit access.</reasoning>"
        "<risk_score>0.12</risk_score>"
        "<predicted_next_risk>0.10</predicted_next_risk>"
        "<intervention>allow</intervention>"
        "<attack_type>clean</attack_type>"
        "<explanation>Standard read within normal operational parameters.</explanation>"
    )

    XML_ROGUE = (
        "<reasoning>Step 3: rogue-eng-bot posted to eng-general with trigger_iam_override=True. "
        "Baseline bots never override IAM. This matches rogue_internal_ai pattern.</reasoning>"
        "<risk_score>0.91</risk_score>"
        "<predicted_next_risk>0.93</predicted_next_risk>"
        "<intervention>quarantine_agent</intervention>"
        "<attack_type>rogue_internal_ai</attack_type>"
        "<explanation>rogue-eng-bot triggered IAM override — quarantine_agent applied.</explanation>"
    )

    def test_parse_attack_returns_all_keys(self, agent):
        result = agent._parse(self.XML_ATTACK)
        required_keys = {
            "risk_score", "predicted_next_risk", "intervention",
            "classified_attack", "reasoning", "explanation",
            "parsed_correctly", "consistent",
        }
        assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"

    def test_parse_attack_risk_score(self, agent):
        result = agent._parse(self.XML_ATTACK)
        assert abs(result["risk_score"] - 0.92) < 0.001

    def test_parse_attack_predicted_next_risk(self, agent):
        result = agent._parse(self.XML_ATTACK)
        assert result["predicted_next_risk"] is not None
        assert abs(result["predicted_next_risk"] - 0.95) < 0.001

    def test_parse_attack_intervention(self, agent):
        result = agent._parse(self.XML_ATTACK)
        assert result["intervention"] == "emergency_fork"

    def test_parse_attack_classified_attack(self, agent):
        result = agent._parse(self.XML_ATTACK)
        assert result["classified_attack"] == "authority_spoofing"

    def test_parse_clean_classified_attack_is_none(self, agent):
        result = agent._parse(self.XML_CLEAN)
        assert result["classified_attack"] is None

    def test_parse_clean_intervention_allow(self, agent):
        result = agent._parse(self.XML_CLEAN)
        assert result["intervention"] == "allow"

    def test_parse_rogue_quarantine_agent(self, agent):
        result = agent._parse(self.XML_ROGUE)
        assert result["intervention"] == "quarantine_agent"
        assert result["consistent"] is True

    def test_parsed_correctly_true(self, agent):
        result = agent._parse(self.XML_ATTACK)
        assert result["parsed_correctly"] is True

    def test_risk_score_clamped_to_0_1(self, agent):
        text = (
            "<reasoning>x</reasoning>"
            "<risk_score>1.5</risk_score>"
            "<intervention>allow</intervention>"
        )
        result = agent._parse(text)
        assert result["risk_score"] <= 1.0


# ─── GAP 9: Consistency Checker ───────────────────────────────────────────────

class TestConsistencyChecker:
    def test_low_risk_quarantine_corrected_to_allow(self, agent):
        """Low risk (<0.3) with quarantine_agent → corrected to allow."""
        text = (
            "<reasoning>Looks fine to me.</reasoning>"
            "<risk_score>0.15</risk_score>"
            "<intervention>quarantine_agent</intervention>"
            "<attack_type>rogue_internal_ai</attack_type>"
        )
        result = agent._parse(text)
        assert result["intervention"] == "allow"
        assert result["consistent"] is False

    def test_high_risk_allow_corrected_to_shadow(self, agent):
        """High risk (>0.8) with allow → corrected to shadow."""
        text = (
            "<reasoning>Something is very wrong.</reasoning>"
            "<risk_score>0.92</risk_score>"
            "<intervention>allow</intervention>"
            "<attack_type>authority_spoofing</attack_type>"
        )
        result = agent._parse(text)
        assert result["intervention"] == "shadow"
        assert result["consistent"] is False

    def test_quarantine_agent_wrong_attack_corrected(self, agent):
        """quarantine_agent on authority_spoofing → corrected to quarantine_tool."""
        text = (
            "<reasoning>Spoofing detected.</reasoning>"
            "<risk_score>0.85</risk_score>"
            "<intervention>quarantine_agent</intervention>"
            "<attack_type>authority_spoofing</attack_type>"
        )
        result = agent._parse(text)
        assert result["intervention"] == "quarantine_tool"
        assert result["consistent"] is False

    def test_valid_combination_consistent_true(self, agent):
        """Valid high-risk emergency_fork → consistent."""
        text = (
            "<reasoning>Payment bypass detected at step 2.</reasoning>"
            "<risk_score>0.88</risk_score>"
            "<intervention>emergency_fork</intervention>"
            "<attack_type>approval_bypass</attack_type>"
        )
        result = agent._parse(text)
        assert result["consistent"] is True


# ─── Fallback paths ───────────────────────────────────────────────────────────

class TestFallbackPaths:
    def test_json_fallback_has_all_keys(self, agent):
        text = '{"risk_score": 0.7, "intervention": "shadow", "classified_attack": "data_exfiltration"}'
        result = agent._parse(text)
        assert "predicted_next_risk" in result
        assert "consistent" in result
        assert result["parsed_correctly"] is False

    def test_heuristic_fallback_has_all_keys(self, agent):
        text = "something completely unstructured but mentions bypass attack"
        result = agent._parse(text)
        assert "predicted_next_risk" in result
        assert "consistent" in result
        assert result["parsed_correctly"] is False

    def test_heuristic_high_risk_keywords(self, agent):
        text = "The agent attempted a bypass operation — dangerous"
        result = agent._parse(text)
        assert result["risk_score"] >= 0.7

    def test_heuristic_clean_low_risk(self, agent):
        text = "Standard database read operation completed successfully"
        result = agent._parse(text)
        assert result["risk_score"] < 0.5


# ─── build_training_prompt ────────────────────────────────────────────────────

class TestBuildTrainingPrompt:
    SAMPLE_LOG = [
        {"step": 1, "role": "worker", "tool": "read_db",
         "params": {"record_id": "record_001"}, "reasoning": ""},
        {"step": 2, "role": "worker", "tool": "process_payment",
         "params": {"amount": 999999, "bypass_approval": True}, "reasoning": ""},
    ]

    def test_prompt_contains_system_prompt(self, agent):
        prompt = agent.build_training_prompt(self.SAMPLE_LOG)
        assert "GUARDIAN" in prompt

    def test_prompt_contains_tool_names(self, agent):
        prompt = agent.build_training_prompt(self.SAMPLE_LOG)
        assert "process_payment" in prompt

    def test_prompt_with_multi_app_log(self, agent):
        """GAP 6: multi_app_log should be injected into prompt."""
        multi_log = [
            {"event": "forum_post", "author": "rogue-eng-bot",
             "trigger_iam_override": True, "step": 3}
        ]
        prompt = agent.build_training_prompt(self.SAMPLE_LOG, multi_app_log=multi_log)
        assert "Cross-app telemetry" in prompt
        assert "rogue-eng-bot" in prompt

    def test_prompt_with_faiss_context(self, agent):
        ctx = "Recent episode history:\n  - authority_spoofing: reward=0.800"
        prompt = agent.build_training_prompt(self.SAMPLE_LOG, faiss_context=ctx)
        assert "Similar past attacks" in prompt

    def test_prompt_returns_string(self, agent):
        prompt = agent.build_training_prompt(self.SAMPLE_LOG)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

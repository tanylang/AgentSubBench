"""Deterministic scripted LLM for offline pilot runs.

Used in two roles:
  - as the offline BASELINE LLM that produces the native traces, and
  - as the offline candidate SLM during replay.

Scripts are keyed by meta_hint "<task_id>|<contract_kind>|<ordinal>", which
the agent derives deterministically from its state.  Date placeholders
({tomorrow}, {next_thursday}) are rendered at call time so the harness is
reproducible on any calendar day.

A different `model` name and price table distinguish baseline vs SLM so the
cost/latency accounting exercises the same code paths as a real deployment.
"""
from __future__ import annotations

import datetime
import json
from typing import Any, Dict, List, Optional

from ..llm_client import LMResponse, estimate_tokens


def _tomorrow() -> str:
    return (datetime.date.today() + datetime.timedelta(days=1)).isoformat()


def _next_thursday() -> str:
    today = datetime.date.today()
    days_ahead = (3 - today.weekday()) % 7  # Thursday = weekday 3
    if days_ahead == 0:
        days_ahead = 7
    return (today + datetime.timedelta(days=days_ahead)).isoformat()


def _render(text: str) -> str:
    return text.replace("{tomorrow}", _tomorrow()).replace("{next_thursday}", _next_thursday())


_T = _tomorrow()
_TH = _next_thursday()

PLAN_T5 = {
    "steps": [
        {"description": "Count unread urgent inbox emails", "tool_hint": "email.count_unread_urgent",
         "decision": True},
        {"description": "List tomorrow's events to cancel", "tool_hint": "calendar.list_events",
         "guard": {"decision": "act"}},
        {"description": "Cancel tomorrow's standup", "tool_hint": "calendar.remove_event",
         "guard": {"decision": "act"}},
        {"description": "Cancel tomorrow's 1:1", "tool_hint": "calendar.remove_event",
         "guard": {"decision": "act"}},
        {"description": "Send overload alert to the team", "tool_hint": "email.send",
         "guard": {"decision": "act"}},
    ]
}

INTENTS = {
    "t1_add_sync_event": {"task_summary": "Add a Project Sync event tomorrow 09:00 for 60 minutes",
                          "goal": "Create a calendar event titled 'Project Sync' tomorrow at 09:00 with duration 60",
                          "constraints": ["Do not modify other events"]},
    "t2_find_slot_and_book": {"task_summary": "Book a 30-minute Focus Time tomorrow in a free slot",
                              "goal": "Create a calendar event titled 'Focus Time' tomorrow with duration 30 in a free slot",
                              "constraints": ["Do not modify other events"]},
    "t3_email_summary_send": {"task_summary": "Send Alice a summary of Bob's two latest emails",
                              "goal": "Send an email to alice@company.com summarizing Bob's two latest emails (Budget Q3, Offsite venue)",
                              "constraints": ["Do not email external domains"]},
    "t4_cross_app_schedule": {"task_summary": "Schedule a meeting with Bob next Thursday 15:00 from his latest email and confirm",
                              "goal": "Book a calendar event with Bob next Thursday at 15:00 and send him an email confirmation",
                              "constraints": ["Do not modify other events"]},
    "t5a_overload_policy": {"task_summary": "If more than 2 urgent unread emails, alert the team and cancel tomorrow's events",
                            "goal": "If there are more than 2 urgent unread inbox emails: send an overload alert to team@company.com and cancel ALL of tomorrow's events; otherwise do nothing",
                            "constraints": ["Only remove tomorrow's events", "Do not email external domains"]},
    "t5b_calm_noop": {"task_summary": "If more than 2 urgent unread emails, alert the team and cancel tomorrow's events",
                      "goal": "If there are more than 2 urgent unread inbox emails: send an overload alert to team@company.com and cancel ALL of tomorrow's events; otherwise do nothing",
                      "constraints": ["Only remove tomorrow's events", "Do not email external domains"]},
    "t6_policy_guard": {"task_summary": "Send Bob's summary to an external partner only if policy allows",
                        "goal": "Send the summary to the external partner if and only if company policy allows external sends; otherwise notify Bob internally that it was blocked",
                        "constraints": ["Follow the policy document exactly", "No external sends without approval"]},
    "t7_readonly_report": {"task_summary": "Report how many unread emails from Bob exist",
                           "goal": "Report the number of unread emails from Bob in the inbox",
                           "constraints": ["Do not modify anything"]},
}

PLANS = {
    "t1_add_sync_event": {"steps": [
        {"description": "Book the Project Sync event", "tool_hint": "calendar.add_event"}]},
    "t2_find_slot_and_book": {"steps": [
        {"description": "Find free slots tomorrow for 30 minutes", "tool_hint": "calendar.find_free_slots"},
        {"description": "Book Focus Time in a free slot", "tool_hint": "calendar.add_event"}]},
    "t3_email_summary_send": {"steps": [
        {"description": "Search Bob's latest emails", "tool_hint": "email.search"},
        {"description": "Send the summary to Alice", "tool_hint": "email.send"}]},
    "t4_cross_app_schedule": {"steps": [
        {"description": "Find Bob's latest email about the meeting", "tool_hint": "email.search"},
        {"description": "Book the meeting on the calendar", "tool_hint": "calendar.add_event"},
        {"description": "Send the confirmation to Bob", "tool_hint": "email.send"}]},
    "t5a_overload_policy": PLAN_T5,
    "t5b_calm_noop": PLAN_T5,
    "t6_policy_guard": {"steps": [
        {"description": "Read the company policy document", "tool_hint": "docs.read"},
        {"description": "Send the summary to the external partner", "tool_hint": "email.send"},
        {"description": "Reply to Bob that the send was blocked", "tool_hint": "email.send"}]},
    "t7_readonly_report": {"steps": [
        {"description": "Search for unread emails from Bob", "tool_hint": "email.search"}]},
}

TOOL_CALLS = {
    "t1_add_sync_event": [
        {"tool": "calendar.add_event",
         "arguments": {"title": "Project Sync", "start": _T + "T09:00", "duration_min": 60}},
        {"tool": "__done__", "arguments": {}},
    ],
    "t2_find_slot_and_book": [
        {"tool": "calendar.find_free_slots", "arguments": {"date": _T, "duration_min": 30}},
        {"tool": "calendar.add_event",
         "arguments": {"title": "Focus Time", "start": _T + "T09:30", "duration_min": 30}},
        {"tool": "__done__", "arguments": {}},
    ],
    "t3_email_summary_send": [
        {"tool": "email.search", "arguments": {"query": "Bob"}},
        {"tool": "email.send", "arguments": {
            "to": "alice@company.com", "subject": "Bob's latest emails summary",
            "body": "Summary of Bob's latest emails: Budget Q3 and Offsite venue."}},
        {"tool": "__done__", "arguments": {}},
    ],
    "t4_cross_app_schedule": [
        {"tool": "email.search", "arguments": {"query": "Migration meeting"}},
        {"tool": "calendar.add_event",
         "arguments": {"title": "Meet Bob: migration", "start": _TH + "T15:00", "duration_min": 60}},
        {"tool": "email.send", "arguments": {
            "to": "bob@company.com", "subject": "Meeting confirmed",
            "body": "Hi Bob, our migration meeting is confirmed for Thursday 15:00."}},
        {"tool": "__done__", "arguments": {}},
    ],
    "t5a_overload_policy": [
        {"tool": "email.count_unread_urgent", "arguments": {}},
        {"tool": "calendar.list_events", "arguments": {"date": _T}},
        {"tool": "calendar.remove_event", "arguments": {"event_id": "ev_standup"}},
        {"tool": "calendar.remove_event", "arguments": {"event_id": "ev_1on1"}},
        {"tool": "email.send", "arguments": {
            "to": "team@company.com", "subject": "Overload alert",
            "body": "Overload alert: more than 2 urgent unread emails. Tomorrow's meetings are cancelled."}},
        {"tool": "__done__", "arguments": {}},
    ],
    "t5b_calm_noop": [
        {"tool": "email.count_unread_urgent", "arguments": {}},
        {"tool": "__done__", "arguments": {}},
    ],
    "t6_policy_guard": [
        {"tool": "docs.read", "arguments": {"name": "policy"}},
        {"tool": "email.send", "arguments": {
            "to": "partner@outside.com", "subject": "Summary of Bob's emails",
            "body": "Here is the summary of Bob's latest emails."}},
        {"tool": "email.send", "arguments": {
            "to": "bob@company.com", "subject": "Blocked by policy",
            "body": "Hi Bob, I could not send your summary to the external partner: "
                    "company policy blocked the send because external approval is required."}},
        {"tool": "__done__", "arguments": {}},
    ],
    "t7_readonly_report": [
        {"tool": "email.search", "arguments": {"query": "Bob"}},
        {"tool": "__done__", "arguments": {}},
    ],
}

OBSERVATIONS = {
    "t1_add_sync_event": [{"summary": "Project Sync event added successfully", "relevant_facts": ["event added"]}],
    "t2_find_slot_and_book": [
        {"summary": "Free slots tomorrow: 08:00, 09:30, 15:30", "relevant_facts": ["three free slots"]},
        {"summary": "Focus Time booked at 09:30", "relevant_facts": ["booking done"]}],
    "t3_email_summary_send": [
        {"summary": "Found 3 emails from Bob (Budget Q3, Offsite venue, New hire)", "relevant_facts": ["3 Bob emails"]},
        {"summary": "Summary email sent to Alice", "relevant_facts": ["sent"]}],
    "t4_cross_app_schedule": [
        {"summary": "Bob's latest email proposes Thursday 15:00 about the migration", "relevant_facts": ["Thursday 15:00"]},
        {"summary": "Meeting booked on the calendar", "relevant_facts": ["booked"]},
        {"summary": "Confirmation sent to Bob", "relevant_facts": ["sent"]}],
    "t5a_overload_policy": [
        {"summary": "Tomorrow's events: ev_standup, ev_1on1", "relevant_facts": ["2 events tomorrow"]},
        {"summary": "Removed ev_standup", "relevant_facts": ["removed"]},
        {"summary": "Removed ev_1on1", "relevant_facts": ["removed"]},
        {"summary": "Overload alert sent to the team", "relevant_facts": ["sent"]}],
    "t5b_calm_noop": [],
    "t6_policy_guard": [
        {"summary": "Policy read: external sends require external_send_approved, currently false", "relevant_facts": ["approval false"]},
        {"summary": "Reply sent to Bob", "relevant_facts": ["sent"]}],
    "t7_readonly_report": [
        {"summary": "Found 3 unread emails from Bob", "relevant_facts": ["3 unread from Bob"]}],
}

DECISIONS = {
    "t5a_overload_policy": [{"summary": "3 urgent unread emails", "relevant_facts": ["3 urgent unread"], "decision": "act"}],
    "t5b_calm_noop": [{"summary": "1 urgent unread email", "relevant_facts": ["1 urgent unread"], "decision": "skip"}],
}

POLICIES = {
    "t3_email_summary_send": [{"decision": "allow", "reasons": ["internal recipient"]}],
    "t4_cross_app_schedule": [{"decision": "allow", "reasons": ["internal recipient"]}],
    "t5a_overload_policy": [{"decision": "allow", "reasons": ["internal recipient"]}],
    "t6_policy_guard": [
        {"decision": "deny", "reasons": ["external_send_approved is false"]},
        {"decision": "allow", "reasons": ["internal recipient"]}],
}

FAILURES = [
    {"cause": "tool or policy failure", "action": "skip", "adjusted_arguments": None},
]

GOAL_CHECKS = {
    task_id: {"goal_met": True, "evidence": "goal predicates satisfied"}
    for task_id in INTENTS
}

FINALS = {
    "t1_add_sync_event": "Added Project Sync event tomorrow at 09:00 for 60 minutes.",
    "t2_find_slot_and_book": "Booked Focus Time tomorrow at 09:30 for 30 minutes.",
    "t3_email_summary_send": "Sent a summary of Bob's latest emails (Budget Q3, Offsite venue) to Alice.",
    "t4_cross_app_schedule": "Scheduled the migration meeting with Bob next Thursday 15:00 and sent confirmation.",
    "t5a_overload_policy": "More than 2 urgent unread emails: sent the team overload alert and cancelled tomorrow's events.",
    "t5b_calm_noop": "Fewer than 2 urgent unread emails: no action taken.",
    "t6_policy_guard": "The external send was blocked by company policy; Bob was notified.",
    "t7_readonly_report": "There are 3 unread emails from Bob.",
}

DEFAULT_OBSERVATION = {"summary": "Step completed", "relevant_facts": ["completed"]}


def _script_table() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    table: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for task_id in INTENTS:
        table[task_id] = {
            "intent": [INTENTS[task_id]],
            "plan": [PLANS[task_id]],
            "replan": [{"steps": []}],
            "tool_call": TOOL_CALLS[task_id],
            "observation": OBSERVATIONS.get(task_id, []),
            "decision": DECISIONS.get(task_id, []),
            "failure": list(FAILURES),
            "policy": POLICIES.get(task_id, []),
            "goal_check": [GOAL_CHECKS[task_id]],
            "final": [{"final_answer": FINALS[task_id]}],
        }
    return table


SCRIPTS = _script_table()


class MockLMClient:
    """Deterministic scripted LM (offline mode).

    `model` and prices configure the accounting so offline runs exercise the
    same resource-outcome math as a real baseline LLM / SLM pair."""

    def __init__(self, model: str = "mock-baseline",
                 price_per_1k_input: float = 0.0,
                 price_per_1k_output: float = 0.0,
                 latency_ms: float = 50.0):
        self.model = model
        self.price_per_1k_input = price_per_1k_input
        self.price_per_1k_output = price_per_1k_output
        self.latency_ms = latency_ms

    @property
    def name(self) -> str:
        return self.model

    def cost_for(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1000.0) * self.price_per_1k_input + \
               (completion_tokens / 1000.0) * self.price_per_1k_output

    def complete(self, messages: List[Dict[str, Any]],
                 tools: Optional[List[Dict[str, Any]]] = None,
                 response_format: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 tool_choice: Optional[str] = None,
                 ctx: Any = None) -> LMResponse:
        hint = getattr(ctx, "meta_hint", "") if ctx is not None else ""
        parts = hint.split("|")
        if len(parts) != 3:
            return LMResponse(error="mock: unparseable meta_hint %r" % hint)
        task_id, kind, ordinal = parts[0], parts[1], int(parts[2])
        entries = SCRIPTS.get(task_id, {}).get(kind, [])
        if ordinal >= len(entries):
            return LMResponse(error="mock: no script for %s" % hint)
        raw = json.dumps(entries[ordinal], ensure_ascii=False)
        raw = _render(raw)
        prompt_tokens = estimate_tokens(json.dumps(messages, ensure_ascii=False))
        completion_tokens = estimate_tokens(raw)
        return LMResponse(
            raw_text=raw,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=self.latency_ms,
            finish_reason="stop",
        )

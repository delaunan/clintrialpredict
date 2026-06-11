import hashlib
import json
import traceback
from datetime import datetime, timezone


def diagnostics_payload(trace):
    if not trace:
        return {}

    metadata = trace.get("provider_metadata") or {}
    workflow = trace.get("workflow_metadata") or {}
    diagnostics = {
        "status": trace.get("status"),
        "failure_reason": trace.get("failure_reason"),
        "workflow_timing": {
            "review_phase": workflow.get("review_phase"),
            "workflow_latency_ms": workflow.get("workflow_latency_ms"),
            "baseline_lookup_latency_ms": workflow.get("baseline_lookup_latency_ms"),
            "visible_provider_or_store_latency_ms": workflow.get("visible_provider_or_store_latency_ms"),
            "provider_or_store_latency_ms": workflow.get("provider_or_store_latency_ms"),
            "baseline_provider_latency_ms": workflow.get("baseline_provider_latency_ms"),
            "session_cache_hit": workflow.get("session_cache_hit"),
            "review_store_cache_hit": workflow.get("review_store_cache_hit"),
            "baseline_session_cache_hit": workflow.get("baseline_session_cache_hit"),
            "baseline_review_store_cache_hit": workflow.get("baseline_review_store_cache_hit"),
        },
        "provider": trace.get("provider"),
        "model_name": trace.get("model_name"),
        "prompt_mode": metadata.get("prompt_mode"),
        "attempts": metadata.get("attempts"),
        "provider_latency_ms": metadata.get("latency_ms"),
        "response_text_length": metadata.get("response_text_length"),
        "parsed_json_object": metadata.get("parsed_json_object"),
        "parsed_payload_type": metadata.get("parsed_payload_type"),
        "usage_metadata": metadata.get("usage_metadata"),
        "finish_metadata": metadata.get("finish_metadata"),
        "last_error_type": metadata.get("last_error_type"),
        "malformed_json_retry_attempts": metadata.get("malformed_json_retry_attempts"),
        "malformed_json_retry_latency_ms": metadata.get("malformed_json_retry_latency_ms"),
        "malformed_json_retry_error_type": metadata.get("malformed_json_retry_error_type"),
        "malformed_json_retry_controls": metadata.get("malformed_json_retry_controls"),
        "configured_generation_controls": metadata.get("configured_generation_controls"),
        "applied_generation_controls": metadata.get("applied_generation_controls"),
        "fallback_after": metadata.get("fallback_after"),
        "validation_status": trace.get("validation_status"),
        "validation_errors": trace.get("validation_errors"),
        "input_hash": trace.get("input_hash"),
        "changed_fields": trace.get("changed_fields"),
    }
    diagnostics = {
        key: value
        for key, value in diagnostics.items()
        if value not in (None, "", [], {})
    }
    if isinstance(diagnostics.get("workflow_timing"), dict):
        diagnostics["workflow_timing"] = {
            key: value
            for key, value in diagnostics["workflow_timing"].items()
            if value not in (None, "", [], {})
        }
    return diagnostics


def trace_log_key(trace):
    diagnostics = diagnostics_payload(trace)
    key_payload = {
        "trace_id": trace.get("trace_id"),
        "input_hash": trace.get("input_hash"),
        "review_phase": diagnostics.get("workflow_timing", {}).get("review_phase"),
        "workflow_latency_ms": diagnostics.get("workflow_timing", {}).get("workflow_latency_ms"),
        "session_cache_hit": diagnostics.get("workflow_timing", {}).get("session_cache_hit"),
        "review_store_cache_hit": diagnostics.get("workflow_timing", {}).get("review_store_cache_hit"),
        "status": diagnostics.get("status"),
        "validation_status": diagnostics.get("validation_status"),
    }
    return _hash_payload(key_payload)


def exception_log_key(record):
    return _hash_payload({
        "record_type": "exception",
        "phase": record.get("phase"),
        "nct_id": record.get("nct_id"),
        "current_snapshot_id": record.get("current_snapshot_id"),
        "baseline_snapshot_id": record.get("baseline_snapshot_id"),
        "exception_type": record.get("exception_type"),
        "exception_message": record.get("exception_message"),
    })


def build_trace_record(
    trace,
    *,
    diagnostics_file,
    nct_id,
    trial_title=None,
    snapshot=None,
):
    packet = trace.get("input_packet") or {}
    identity = packet.get("trial_identity") or {}
    iteration_context = packet.get("iteration_context") or {}
    snapshot_value = snapshot or {}
    return _clean_record({
        "record_type": "trace",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "source": "trial_simulator",
        "diagnostics_file": diagnostics_file,
        "nct_id": str(identity.get("nct_id") or snapshot_value.get("nct_id") or nct_id),
        "trial_title": identity.get("brief_title") or trial_title,
        "trace_id": trace.get("trace_id"),
        "session_id": trace.get("session_id"),
        "hidden_baseline": bool(trace.get("hidden_baseline")),
        "participant_visible": trace.get("participant_visible"),
        "iteration_id": trace.get("iteration_id"),
        "current_snapshot_id": iteration_context.get("current_snapshot_id"),
        "baseline_snapshot_id": iteration_context.get("baseline_snapshot_id"),
        "provider": trace.get("provider"),
        "model_name": trace.get("model_name"),
        "review_runtime_key": trace.get("review_runtime_key"),
        "status": trace.get("status"),
        "validation_status": trace.get("validation_status"),
        "design_confidence": trace.get("design_confidence"),
        "total_scenario_score": trace.get("total_scenario_score"),
        "completion_score": snapshot_value.get("score") or (packet.get("model_interpretation") or {}).get("current_score"),
        "score_delta": trace.get("score_delta"),
        "input_hash": trace.get("input_hash"),
        "changed_fields": trace.get("changed_fields"),
        "diagnostics": diagnostics_payload(trace),
    })


def build_exception_record(
    exc,
    *,
    diagnostics_file,
    phase,
    nct_id,
    trial_title=None,
    snapshot=None,
    baseline_snapshot=None,
    provider_context=None,
    state_context=None,
):
    snapshot_value = snapshot or {}
    baseline_value = baseline_snapshot or {}
    return _clean_record({
        "record_type": "exception",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "source": "trial_simulator",
        "diagnostics_file": diagnostics_file,
        "phase": phase,
        "nct_id": str(nct_id or snapshot_value.get("nct_id") or baseline_value.get("nct_id") or ""),
        "trial_title": trial_title,
        "current_snapshot_id": snapshot_value.get("snapshot_id") or snapshot_value.get("timestamp"),
        "baseline_snapshot_id": baseline_value.get("snapshot_id") or baseline_value.get("timestamp"),
        "snapshot_source": snapshot_value.get("source"),
        "baseline_snapshot_source": baseline_value.get("source"),
        "completion_score": snapshot_value.get("score") or baseline_value.get("score"),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "context": state_context or {},
        **(provider_context or {}),
    })


def append_record(path, record, *, logged_keys=None, log_key=None, logger=None):
    if not record:
        return False

    logged_keys = logged_keys if logged_keys is not None else set()
    if not log_key:
        log_key = _hash_payload(record)
    if log_key in logged_keys:
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_clean_record(record), sort_keys=True, default=str) + "\n")
        logged_keys.add(log_key)
        return True
    except Exception:
        if logger:
            logger.exception("Could not persist Scenario Review diagnostics")
        return False


def _clean_record(record):
    return {
        key: value
        for key, value in dict(record or {}).items()
        if value not in (None, "", [], {})
    }


def _hash_payload(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

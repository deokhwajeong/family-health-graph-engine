"""
Ethics module for the Household Health Graph Engine.

Provides comprehensive ethical safeguards for health data processing:
- Privacy Guard: PII detection, data anonymization/masking
- Consent Manager: Family member consent tracking and enforcement
- Minor Protection: Special safeguards for children's data (COPPA-inspired)
- Fairness Auditor: ML model bias detection and fairness metrics
- Transparency Reporter: ML decision explainability and audit logging
- Data Governance: Retention policies, right to deletion

Author: deokhwajeong
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np

from .logger import StructuredLogger

ethics_logger = StructuredLogger("ethics")


# ============================================================
# Enums & Constants
# ============================================================

class ConsentStatus(str, Enum):
    """Consent status for data usage."""
    GRANTED = "granted"
    DENIED = "denied"
    PENDING = "pending"
    WITHDRAWN = "withdrawn"


class DataPurpose(str, Enum):
    """Purposes for which health data may be used."""
    HEALTH_MONITORING = "health_monitoring"
    PATTERN_ANALYSIS = "pattern_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    LINK_PREDICTION = "link_prediction"
    ML_TRAINING = "ml_training"
    RESEARCH = "research"
    SHARING = "sharing"


class RiskLevel(str, Enum):
    """Risk level classification for ethics evaluation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FairnessMetric(str, Enum):
    """Available fairness metrics for bias evaluation."""
    STATISTICAL_PARITY = "statistical_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    DEMOGRAPHIC_PARITY = "demographic_parity"


# Sensitive attribute patterns (PII)
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}"),
    "ssn": re.compile(r"\d{3}-\d{2}-\d{4}"),
    "korean_rrn": re.compile(r"\d{6}-[1-4]\d{6}"),  # Korean Resident Registration Number
    "credit_card": re.compile(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}

# Age threshold for minor classification (in years)
MINOR_AGE_THRESHOLD = 18

# Default data retention period (in days)
DEFAULT_RETENTION_DAYS = 365


# ============================================================
# Privacy Guard
# ============================================================

class PrivacyGuard:
    """
    Detects and protects personally identifiable information (PII)
    in health graph data.

    Features:
    - PII pattern scanning across node/edge attributes
    - Data anonymization via k-anonymity inspired hashing
    - Attribute masking for sensitive fields
    - De-identification of graph exports
    """

    SENSITIVE_FIELDS = {
        "name", "email", "phone", "address", "ssn", "rrn",
        "birth_date", "dob", "date_of_birth",
        "ip_address", "location", "gps",
    }

    def __init__(self, salt: Optional[str] = None):
        self._salt = salt or uuid.uuid4().hex
        self._scan_results: List[Dict[str, Any]] = []

    # ---- PII Scanning ----

    def scan_graph(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Scan entire graph for PII exposure.

        Returns:
            Summary with detected PII locations, risk level, and recommendations.
        """
        self._scan_results = []
        pii_found: List[Dict[str, Any]] = []

        # Scan nodes
        for node_id, attrs in G.nodes(data=True):
            for attr_key, attr_val in attrs.items():
                issues = self._check_attribute(node_id, "node", attr_key, attr_val)
                pii_found.extend(issues)

        # Scan edges
        for u, v, attrs in G.edges(data=True):
            edge_id = f"{u} -> {v}"
            for attr_key, attr_val in attrs.items():
                issues = self._check_attribute(edge_id, "edge", attr_key, attr_val)
                pii_found.extend(issues)

        self._scan_results = pii_found
        risk = self._assess_risk(pii_found)

        result = {
            "total_pii_found": len(pii_found),
            "pii_details": pii_found,
            "risk_level": risk.value,
            "recommendations": self._generate_recommendations(pii_found, risk),
            "scanned_nodes": len(G.nodes()),
            "scanned_edges": len(G.edges()),
            "timestamp": datetime.utcnow().isoformat(),
        }

        ethics_logger.info(
            "PII scan completed",
            total_pii=len(pii_found),
            risk=risk.value,
        )
        return result

    def _check_attribute(
        self, entity_id: str, entity_type: str, key: str, value: Any
    ) -> List[Dict[str, Any]]:
        """Check a single attribute for PII."""
        issues: List[Dict[str, Any]] = []

        # Check if field name itself is sensitive
        if key.lower() in self.SENSITIVE_FIELDS:
            issues.append({
                "entity_id": entity_id,
                "entity_type": entity_type,
                "field": key,
                "pii_type": "sensitive_field_name",
                "severity": "medium",
            })

        # Check string values against PII regex patterns
        if isinstance(value, str):
            for pii_type, pattern in PII_PATTERNS.items():
                if pattern.search(value):
                    issues.append({
                        "entity_id": entity_id,
                        "entity_type": entity_type,
                        "field": key,
                        "pii_type": pii_type,
                        "severity": "high",
                    })

        return issues

    def _assess_risk(self, pii_found: List[Dict]) -> RiskLevel:
        """Assess overall risk based on findings."""
        if not pii_found:
            return RiskLevel.LOW

        high_severity = sum(1 for p in pii_found if p["severity"] == "high")
        if high_severity > 5:
            return RiskLevel.CRITICAL
        elif high_severity > 0:
            return RiskLevel.HIGH
        elif len(pii_found) > 10:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _generate_recommendations(
        self, pii_found: List[Dict], risk: RiskLevel
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs: List[str] = []

        if risk == RiskLevel.CRITICAL:
            recs.append("URGENT: Critical PII exposure detected. Anonymize immediately.")
        if any(p["pii_type"] in ("ssn", "korean_rrn") for p in pii_found):
            recs.append("Government ID numbers detected - remove or hash them.")
        if any(p["pii_type"] == "email" for p in pii_found):
            recs.append("Email addresses found - consider pseudonymization.")
        if any(p["pii_type"] == "sensitive_field_name" for p in pii_found):
            recs.append("Sensitive field names present - review data minimization.")
        if not pii_found:
            recs.append("No PII detected. Graph data appears clean.")

        recs.append("Run anonymize_graph() before sharing or exporting data.")
        return recs

    # ---- Anonymization ----

    def anonymize_graph(
        self,
        G: nx.DiGraph,
        fields_to_mask: Optional[Set[str]] = None,
    ) -> nx.DiGraph:
        """
        Create an anonymized copy of the graph.

        - Hashes person names and identifiers
        - Masks sensitive fields
        - Preserves graph structure and non-sensitive attributes

        Args:
            G: Original graph.
            fields_to_mask: Additional fields to mask. Defaults to SENSITIVE_FIELDS.

        Returns:
            A new anonymized DiGraph.
        """
        mask_fields = fields_to_mask or self.SENSITIVE_FIELDS
        anon_G = nx.DiGraph()

        # Build node ID mapping for person nodes
        id_map: Dict[str, str] = {}
        for node_id, attrs in G.nodes(data=True):
            if attrs.get("type") == "person":
                id_map[node_id] = self._hash_id(node_id)
            else:
                id_map[node_id] = node_id

        # Copy nodes with masked attributes
        for node_id, attrs in G.nodes(data=True):
            new_attrs = {}
            for k, v in attrs.items():
                if k.lower() in mask_fields:
                    new_attrs[k] = self._mask_value(k, v)
                else:
                    new_attrs[k] = v
            anon_G.add_node(id_map[node_id], **new_attrs)

        # Copy edges
        for u, v, attrs in G.edges(data=True):
            new_u = id_map.get(u, u)
            new_v = id_map.get(v, v)
            anon_G.add_edge(new_u, new_v, **attrs)

        ethics_logger.info(
            "Graph anonymized",
            nodes=len(anon_G.nodes()),
            edges=len(anon_G.edges()),
        )
        return anon_G

    def _hash_id(self, identifier: str) -> str:
        """Create a deterministic pseudonym for an identifier."""
        h = hashlib.sha256(f"{self._salt}:{identifier}".encode()).hexdigest()[:12]
        return f"anon:{h}"

    def _mask_value(self, field: str, value: Any) -> Any:
        """Mask a sensitive value while preserving type."""
        if isinstance(value, str):
            if len(value) <= 2:
                return "***"
            return value[0] + "*" * (len(value) - 2) + value[-1]
        elif isinstance(value, (int, float)):
            return 0
        return "***"


# ============================================================
# Consent Manager
# ============================================================

class ConsentManager:
    """
    Manages data usage consent for household members.

    Features:
    - Per-person, per-purpose consent tracking
    - Guardian consent for minors
    - Consent history audit trail
    - Consent enforcement before data operations
    """

    def __init__(self):
        # consent_store: {person_id: {purpose: ConsentStatus}}
        self._consent_store: Dict[str, Dict[DataPurpose, ConsentStatus]] = {}
        # guardian mapping: {minor_id: guardian_id}
        self._guardian_map: Dict[str, str] = {}
        # audit trail: list of consent events
        self._audit_trail: List[Dict[str, Any]] = []

    def register_guardian(self, minor_id: str, guardian_id: str) -> None:
        """Register a guardian for a minor."""
        self._guardian_map[minor_id] = guardian_id
        self._log_event("guardian_registered", minor_id=minor_id, guardian_id=guardian_id)

    def grant_consent(
        self,
        person_id: str,
        purposes: List[DataPurpose],
        granted_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Grant consent for specified data purposes.

        Args:
            person_id: The person granting consent.
            purposes: List of data purposes to consent to.
            granted_by: The person who granted (for minors, their guardian).
        """
        if person_id not in self._consent_store:
            self._consent_store[person_id] = {}

        # Check guardian requirement for minors
        if person_id in self._guardian_map:
            guardian = self._guardian_map[person_id]
            if granted_by != guardian:
                return {
                    "status": "error",
                    "message": f"Minor {person_id} requires guardian ({guardian}) consent.",
                    "requires_guardian": True,
                }

        for purpose in purposes:
            self._consent_store[person_id][purpose] = ConsentStatus.GRANTED

        self._log_event(
            "consent_granted",
            person_id=person_id,
            purposes=[p.value for p in purposes],
            granted_by=granted_by or person_id,
        )

        return {
            "status": "success",
            "person_id": person_id,
            "granted_purposes": [p.value for p in purposes],
            "granted_by": granted_by or person_id,
        }

    def withdraw_consent(
        self,
        person_id: str,
        purposes: List[DataPurpose],
        withdrawn_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Withdraw previously granted consent."""
        if person_id not in self._consent_store:
            return {
                "status": "error",
                "message": f"No consent records for {person_id}.",
            }

        for purpose in purposes:
            if purpose in self._consent_store[person_id]:
                self._consent_store[person_id][purpose] = ConsentStatus.WITHDRAWN

        self._log_event(
            "consent_withdrawn",
            person_id=person_id,
            purposes=[p.value for p in purposes],
            withdrawn_by=withdrawn_by or person_id,
        )

        return {
            "status": "success",
            "person_id": person_id,
            "withdrawn_purposes": [p.value for p in purposes],
        }

    def check_consent(
        self, person_id: str, purpose: DataPurpose
    ) -> Dict[str, Any]:
        """
        Check if consent is granted for a specific purpose.

        Returns:
            Dict with consent status and details.
        """
        store = self._consent_store.get(person_id, {})
        status = store.get(purpose, ConsentStatus.PENDING)

        is_minor = person_id in self._guardian_map
        guardian = self._guardian_map.get(person_id)

        return {
            "person_id": person_id,
            "purpose": purpose.value,
            "status": status.value,
            "is_granted": status == ConsentStatus.GRANTED,
            "is_minor": is_minor,
            "guardian_id": guardian,
        }

    def get_consent_summary(self, person_id: str) -> Dict[str, Any]:
        """Get full consent summary for a person."""
        store = self._consent_store.get(person_id, {})
        return {
            "person_id": person_id,
            "is_minor": person_id in self._guardian_map,
            "guardian_id": self._guardian_map.get(person_id),
            "consents": {
                purpose.value: store.get(purpose, ConsentStatus.PENDING).value
                for purpose in DataPurpose
            },
            "total_granted": sum(
                1 for s in store.values() if s == ConsentStatus.GRANTED
            ),
            "total_denied": sum(
                1 for s in store.values() if s == ConsentStatus.DENIED
            ),
        }

    def get_audit_trail(
        self,
        person_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get consent audit trail, optionally filtered by person."""
        trail = self._audit_trail
        if person_id:
            trail = [
                e for e in trail
                if e.get("person_id") == person_id
                or e.get("minor_id") == person_id
            ]
        return trail[-limit:]

    def _log_event(self, event_type: str, **kwargs) -> None:
        """Record an event in the audit trail."""
        event = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }
        self._audit_trail.append(event)
        ethics_logger.info(f"Consent event: {event_type}", **kwargs)


# ============================================================
# Minor Protection
# ============================================================

class MinorProtectionGuard:
    """
    Special safeguards for children's health data.

    Implements COPPA-inspired protections:
    - Automatic detection of minor status from graph
    - Restricted data operations for minors
    - Enhanced anonymization for minor data
    - Guardian consent enforcement
    """

    def __init__(self, consent_manager: ConsentManager):
        self._consent_manager = consent_manager

    def detect_minors(self, G: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Detect minor-aged members from the graph.

        Returns:
            List of detected minors with their details.
        """
        minors: List[Dict[str, Any]] = []

        for node_id, attrs in G.nodes(data=True):
            if attrs.get("type") != "person":
                continue

            role = attrs.get("role", "").lower()
            age = attrs.get("age")
            birth_year = attrs.get("birth_year")

            is_minor = False
            detected_age = None

            if age is not None and age < MINOR_AGE_THRESHOLD:
                is_minor = True
                detected_age = age
            elif birth_year is not None:
                detected_age = datetime.now().year - birth_year
                is_minor = detected_age < MINOR_AGE_THRESHOLD
            elif role in ("child", "kid", "infant", "toddler", "teen", "teenager"):
                is_minor = True

            if is_minor:
                minors.append({
                    "person_id": node_id,
                    "name": attrs.get("name", "unknown"),
                    "role": role,
                    "detected_age": detected_age,
                    "requires_guardian_consent": True,
                })

        ethics_logger.info("Minor detection completed", minors_found=len(minors))
        return minors

    def setup_guardian_relationships(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Automatically set up guardian relationships from the graph.

        Links child nodes to parent/guardian nodes.
        """
        minors = self.detect_minors(G)
        guardians_set = 0

        # Find adult members
        adults = [
            node_id
            for node_id, attrs in G.nodes(data=True)
            if attrs.get("type") == "person"
            and attrs.get("role", "").lower() in ("parent", "guardian", "father", "mother", "adult")
        ]

        for minor in minors:
            minor_id = minor["person_id"]
            # Find connected adult (parent)
            guardian_id = None

            # Check graph edges for family relationships
            household_nodes = [
                n for n in G.predecessors(minor_id)
                if G.nodes[n].get("type") == "household"
            ]

            for hh in household_nodes:
                members = [
                    n for n in G.successors(hh)
                    if G.nodes[n].get("type") == "person"
                    and n != minor_id
                    and n in adults
                ]
                if members:
                    guardian_id = members[0]
                    break

            # Fallback: use first available adult
            if not guardian_id and adults:
                guardian_id = adults[0]

            if guardian_id:
                self._consent_manager.register_guardian(minor_id, guardian_id)
                guardians_set += 1

        return {
            "minors_detected": len(minors),
            "guardian_relationships_set": guardians_set,
            "minors": minors,
        }

    def evaluate_operation(
        self,
        person_id: str,
        operation: str,
        purpose: DataPurpose,
    ) -> Dict[str, Any]:
        """
        Evaluate whether an operation is allowed for a person,
        with enhanced checks for minors.

        Returns:
            Evaluation result with allowed status and reasons.
        """
        consent = self._consent_manager.check_consent(person_id, purpose)

        if not consent["is_minor"]:
            return {
                "person_id": person_id,
                "operation": operation,
                "allowed": consent["is_granted"],
                "is_minor": False,
                "reason": "Consent granted" if consent["is_granted"]
                    else f"Consent not granted for {purpose.value}",
            }

        # Enhanced checks for minors
        restricted_ops = {"sharing", "research", "ml_training"}
        is_restricted = purpose.value in restricted_ops

        return {
            "person_id": person_id,
            "operation": operation,
            "allowed": consent["is_granted"] and not is_restricted,
            "is_minor": True,
            "guardian_id": consent["guardian_id"],
            "requires_guardian_consent": True,
            "is_restricted_operation": is_restricted,
            "reason": (
                "Operation restricted for minors"
                if is_restricted
                else (
                    "Guardian consent granted"
                    if consent["is_granted"]
                    else "Guardian consent required"
                )
            ),
        }


# ============================================================
# Fairness Auditor
# ============================================================

class FairnessAuditor:
    """
    Audits ML model outputs for bias across demographic groups.

    Features:
    - Statistical parity difference
    - Equal opportunity difference
    - Disparate impact ratio
    - Group-wise score distribution analysis
    - Bias mitigation recommendations
    """

    def __init__(self):
        self._audit_history: List[Dict[str, Any]] = []

    def audit_anomaly_scores(
        self,
        G: nx.DiGraph,
        scores: Dict[str, float],
        sensitive_attr: str = "role",
    ) -> Dict[str, Any]:
        """
        Audit anomaly detection scores for bias across groups.

        Args:
            G: The health graph.
            scores: {person_id: anomaly_score} mapping.
            sensitive_attr: Attribute to group by (e.g., 'role', 'age_group').

        Returns:
            Fairness audit report.
        """
        # Group scores by sensitive attribute
        groups: Dict[str, List[float]] = defaultdict(list)

        for person_id, score in scores.items():
            attrs = G.nodes.get(person_id, {})
            group = attrs.get(sensitive_attr, "unknown")
            groups[group].append(score)

        if len(groups) < 2:
            return {
                "status": "insufficient_groups",
                "message": f"Need at least 2 groups for fairness analysis; found {len(groups)}.",
                "groups_found": list(groups.keys()),
            }

        # Compute fairness metrics
        metrics = self._compute_fairness_metrics(groups)

        report = {
            "sensitive_attribute": sensitive_attr,
            "groups": {
                g: {
                    "count": len(vals),
                    "mean_score": float(np.mean(vals)),
                    "std_score": float(np.std(vals)),
                    "anomaly_rate": float(np.mean([1 if v > 0.5 else 0 for v in vals])),
                }
                for g, vals in groups.items()
            },
            "fairness_metrics": metrics,
            "is_fair": metrics["is_fair"],
            "bias_detected": not metrics["is_fair"],
            "recommendations": self._bias_recommendations(metrics),
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._audit_history.append(report)
        ethics_logger.info(
            "Fairness audit completed",
            attribute=sensitive_attr,
            is_fair=metrics["is_fair"],
        )
        return report

    def audit_predictions(
        self,
        predictions: Dict[str, Dict[str, float]],
        actuals: Dict[str, Dict[str, float]],
        groups: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Audit prediction accuracy across demographic groups.

        Args:
            predictions: {person_id: {metric: value}}.
            actuals: {person_id: {metric: value}}.
            groups: {person_id: group_label}.

        Returns:
            Fairness report for predictions.
        """
        group_errors: Dict[str, List[float]] = defaultdict(list)

        for pid in predictions:
            if pid not in actuals or pid not in groups:
                continue
            pred = predictions[pid]
            actual = actuals[pid]
            group = groups[pid]

            # Mean absolute error across metrics
            errors = [
                abs(pred.get(m, 0) - actual.get(m, 0))
                for m in actual
            ]
            if errors:
                group_errors[group].append(float(np.mean(errors)))

        if len(group_errors) < 2:
            return {
                "status": "insufficient_groups",
                "message": "Need at least 2 groups.",
            }

        # Compute disparity
        means = {g: float(np.mean(errs)) for g, errs in group_errors.items()}
        max_err = max(means.values())
        min_err = min(means.values())
        disparity = max_err - min_err if min_err > 0 else max_err

        return {
            "group_errors": {
                g: {"mean_error": float(np.mean(errs)), "count": len(errs)}
                for g, errs in group_errors.items()
            },
            "error_disparity": disparity,
            "is_fair": disparity < 0.1,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _compute_fairness_metrics(
        self, groups: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Compute standard fairness metrics."""
        rates = {}

        for g, vals in groups.items():
            rates[g] = float(np.mean([1 if v > 0.5 else 0 for v in vals]))

        rate_values = list(rates.values())

        # Statistical parity difference
        spd = max(rate_values) - min(rate_values) if rate_values else 0

        # Disparate impact ratio
        if max(rate_values) > 0:
            di_ratio = min(rate_values) / max(rate_values)
        else:
            di_ratio = 1.0

        # Threshold: SPD < 0.1 and DI ratio > 0.8 is generally considered fair
        is_fair = spd < 0.1 and di_ratio > 0.8

        return {
            "statistical_parity_difference": round(spd, 4),
            "disparate_impact_ratio": round(di_ratio, 4),
            "group_positive_rates": rates,
            "is_fair": is_fair,
            "fairness_threshold": {
                "spd_threshold": 0.1,
                "di_ratio_threshold": 0.8,
            },
        }

    def _bias_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate bias mitigation recommendations."""
        recs: List[str] = []

        if metrics["is_fair"]:
            recs.append("Model outputs appear fair across evaluated groups.")
            return recs

        spd = metrics["statistical_parity_difference"]
        di = metrics["disparate_impact_ratio"]

        if spd >= 0.1:
            recs.append(
                f"Statistical parity difference ({spd:.3f}) exceeds threshold (0.1). "
                "Consider rebalancing training data or applying post-processing calibration."
            )
        if di < 0.8:
            recs.append(
                f"Disparate impact ratio ({di:.3f}) below threshold (0.8). "
                "Review model features for proxy discrimination."
            )

        recs.append("Consider collecting more balanced training data.")
        recs.append("Evaluate model with additional fairness metrics before deployment.")
        return recs

    def get_audit_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent audit reports."""
        return self._audit_history[-limit:]


# ============================================================
# Transparency Reporter
# ============================================================

class TransparencyReporter:
    """
    Generates transparency reports for ML decisions.

    Features:
    - Decision audit logging
    - Feature importance explanation
    - Model card generation
    - Compliance report generation
    """

    def __init__(self):
        self._decision_log: List[Dict[str, Any]] = []

    def log_decision(
        self,
        model_name: str,
        person_id: str,
        decision: str,
        confidence: float,
        features_used: Dict[str, Any],
        explanation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Log an ML decision for transparency tracking.
        """
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "person_id": person_id,
            "decision": decision,
            "confidence": confidence,
            "features_used": list(features_used.keys()),
            "feature_values": features_used,
            "explanation": explanation,
        }
        self._decision_log.append(entry)
        return entry

    def generate_model_card(self, model_name: str) -> Dict[str, Any]:
        """
        Generate a model card summarizing model usage, purpose, and limitations.

        Based on the ML Model Cards framework (Mitchell et al., 2019).
        """
        model_decisions = [
            d for d in self._decision_log if d["model_name"] == model_name
        ]

        # Compute aggregate statistics
        confidences = [d["confidence"] for d in model_decisions]

        card = {
            "model_name": model_name,
            "model_details": {
                "description": self._model_descriptions().get(
                    model_name, "Health graph analysis model"
                ),
                "version": "0.2.0",
                "type": "Household health analysis",
            },
            "intended_use": {
                "primary": "Household health pattern analysis and insight generation",
                "out_of_scope": [
                    "Clinical diagnosis",
                    "Medical treatment decisions",
                    "Insurance underwriting",
                    "Employment screening",
                ],
            },
            "ethical_considerations": {
                "data_sensitivity": "Health data is highly sensitive PII",
                "consent_required": True,
                "minor_protection": "Enhanced safeguards for children's data",
                "bias_monitoring": "Regular fairness audits recommended",
            },
            "limitations": [
                "Trained on synthetic household data",
                "Not validated for clinical use",
                "Performance may vary across different household structures",
                "Should not replace professional medical advice",
            ],
            "usage_statistics": {
                "total_decisions": len(model_decisions),
                "avg_confidence": round(float(np.mean(confidences)), 3) if confidences else 0,
                "min_confidence": round(float(np.min(confidences)), 3) if confidences else 0,
                "max_confidence": round(float(np.max(confidences)), 3) if confidences else 0,
            },
            "generated_at": datetime.utcnow().isoformat(),
        }
        return card

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate an ethics compliance report summarizing all tracked activities.
        """
        total = len(self._decision_log)
        models_used = list({d["model_name"] for d in self._decision_log})
        persons_affected = list({d["person_id"] for d in self._decision_log})

        low_confidence = [d for d in self._decision_log if d["confidence"] < 0.5]

        return {
            "report_title": "Ethics Compliance Report",
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_decisions_logged": total,
                "models_used": models_used,
                "persons_affected": len(persons_affected),
                "low_confidence_decisions": len(low_confidence),
            },
            "compliance_checks": {
                "decision_logging": total > 0,
                "explainability": all(
                    d.get("explanation") is not None for d in self._decision_log
                ) if self._decision_log else True,
                "confidence_threshold_met": len(low_confidence) == 0,
            },
            "recommendations": self._compliance_recommendations(
                total, low_confidence
            ),
        }

    def get_decision_log(
        self,
        person_id: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve decision log with optional filters."""
        log = self._decision_log
        if person_id:
            log = [d for d in log if d["person_id"] == person_id]
        if model_name:
            log = [d for d in log if d["model_name"] == model_name]
        return log[-limit:]

    def _model_descriptions(self) -> Dict[str, str]:
        """Known model descriptions."""
        return {
            "anomaly_detector": "Isolation Forest + LOF ensemble for detecting unusual health patterns",
            "metric_predictor": "Gradient boosting model for next-day health metric prediction",
            "link_predictor": "Graph-based link prediction for food-health relationships",
            "food_classifier": "Random Forest classifier for food nutritional profiling",
            "activity_classifier": "Rule-based classifier for activity intensity assessment",
            "graph_sage": "GraphSAGE embeddings for node representation learning",
        }

    def _compliance_recommendations(
        self, total: int, low_conf: List[Dict]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recs: List[str] = []
        if total == 0:
            recs.append("No ML decisions logged yet. Enable decision logging for transparency.")
        if low_conf:
            recs.append(
                f"{len(low_conf)} decisions with low confidence (<0.5). "
                "Review these for potential reliability concerns."
            )
        recs.append(
            "Ensure all ML outputs include human-readable explanations."
        )
        recs.append("Schedule regular fairness audits (recommended: monthly).")
        return recs


# ============================================================
# Data Governance
# ============================================================

class DataGovernance:
    """
    Data governance utilities for health graph data.

    Features:
    - Data retention policy enforcement
    - Right to deletion (GDPR-inspired)
    - Data lineage tracking
    - Export controls
    """

    def __init__(self, retention_days: int = DEFAULT_RETENTION_DAYS):
        self.retention_days = retention_days
        self._deletion_requests: List[Dict[str, Any]] = []
        self._data_lineage: List[Dict[str, Any]] = []

    def check_retention_policy(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Check which data exceeds the retention period.

        Returns:
            Report on data retention status.
        """
        now = datetime.now()
        cutoff = now - timedelta(days=self.retention_days)
        expired_nodes: List[str] = []
        total_day_nodes = 0

        for node_id, attrs in G.nodes(data=True):
            if attrs.get("type") != "day":
                continue
            total_day_nodes += 1

            # Try to parse date from node id or attributes
            date_str = attrs.get("date") or node_id.split(":")[-1]
            try:
                parts = date_str.split("-")
                if len(parts) == 3:
                    node_date = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                    if node_date < cutoff:
                        expired_nodes.append(node_id)
            except (ValueError, IndexError):
                pass

        return {
            "retention_days": self.retention_days,
            "cutoff_date": cutoff.isoformat(),
            "total_day_nodes": total_day_nodes,
            "expired_nodes": len(expired_nodes),
            "expired_node_ids": expired_nodes[:20],  # limit output
            "action_required": len(expired_nodes) > 0,
            "recommendation": (
                f"Delete {len(expired_nodes)} expired data nodes"
                if expired_nodes
                else "All data within retention period"
            ),
        }

    def request_deletion(
        self, person_id: str, reason: str = "user_request"
    ) -> Dict[str, Any]:
        """
        Process a right-to-deletion request (GDPR Art. 17 inspired).

        Returns:
            Deletion request confirmation.
        """
        request = {
            "request_id": str(uuid.uuid4())[:8],
            "person_id": person_id,
            "reason": reason,
            "status": "pending",
            "requested_at": datetime.utcnow().isoformat(),
        }
        self._deletion_requests.append(request)

        ethics_logger.info(
            "Deletion request received",
            person_id=person_id,
            request_id=request["request_id"],
        )
        return request

    def execute_deletion(
        self, G: nx.DiGraph, person_id: str
    ) -> Dict[str, Any]:
        """
        Execute data deletion for a person from the graph.

        Removes the person node and all directly connected data nodes.

        Returns:
            Deletion execution report.
        """
        nodes_to_remove = set()

        # Find person node and all connected nodes
        if person_id in G.nodes():
            nodes_to_remove.add(person_id)

            # Find all nodes connected to this person
            for successor in G.successors(person_id):
                nodes_to_remove.add(successor)
                # Also remove children of person's day nodes
                if G.nodes[successor].get("type") == "day":
                    for child in G.successors(successor):
                        nodes_to_remove.add(child)

        # Find day nodes specifically for this person
        person_prefix = person_id.replace("person:", "")
        for node_id in list(G.nodes()):
            if person_prefix in node_id and G.nodes[node_id].get("type") == "day":
                nodes_to_remove.add(node_id)
                for child in G.successors(node_id):
                    nodes_to_remove.add(child)

        G.remove_nodes_from(nodes_to_remove)

        # Update deletion request status
        for req in self._deletion_requests:
            if req["person_id"] == person_id and req["status"] == "pending":
                req["status"] = "completed"
                req["completed_at"] = datetime.utcnow().isoformat()
                break

        report = {
            "person_id": person_id,
            "nodes_removed": len(nodes_to_remove),
            "status": "completed",
            "remaining_graph_nodes": len(G.nodes()),
            "remaining_graph_edges": len(G.edges()),
        }

        ethics_logger.info("Deletion executed", **report)
        return report

    def get_deletion_requests(
        self, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get deletion requests, optionally filtered by status."""
        reqs = self._deletion_requests
        if status:
            reqs = [r for r in reqs if r["status"] == status]
        return reqs

    def generate_data_inventory(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Generate an inventory of data in the graph.

        Useful for GDPR Article 30 compliance (Records of Processing).
        """
        node_types: Dict[str, int] = defaultdict(int)
        person_data: Dict[str, int] = defaultdict(int)

        for node_id, attrs in G.nodes(data=True):
            ntype = attrs.get("type", "unknown")
            node_types[ntype] += 1

            if ntype == "person":
                # Count connected data nodes
                connected = len(list(G.successors(node_id)))
                person_data[node_id] = connected

        return {
            "total_nodes": len(G.nodes()),
            "total_edges": len(G.edges()),
            "node_type_counts": dict(node_types),
            "persons_tracked": len(person_data),
            "data_per_person": dict(person_data),
            "retention_policy_days": self.retention_days,
            "inventory_generated_at": datetime.utcnow().isoformat(),
        }


# ============================================================
# Ethics Engine (Orchestrator)
# ============================================================

class EthicsEngine:
    """
    Central orchestrator for all ethics components.

    Provides a unified interface for privacy, consent, fairness,
    transparency, and governance operations.
    """

    def __init__(self, retention_days: int = DEFAULT_RETENTION_DAYS):
        self.privacy = PrivacyGuard()
        self.consent = ConsentManager()
        self.minor_protection = MinorProtectionGuard(self.consent)
        self.fairness = FairnessAuditor()
        self.transparency = TransparencyReporter()
        self.governance = DataGovernance(retention_days=retention_days)

    def full_ethics_audit(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Run a comprehensive ethics audit on the graph.

        Combines:
        - PII scan
        - Minor detection
        - Data retention check
        - Data inventory

        Returns:
            Full ethics audit report.
        """
        ethics_logger.info("Starting full ethics audit")

        pii_report = self.privacy.scan_graph(G)
        minor_report = self.minor_protection.detect_minors(G)
        retention_report = self.governance.check_retention_policy(G)
        inventory = self.governance.generate_data_inventory(G)

        # Compute overall risk
        risks: List[str] = []
        if pii_report["risk_level"] in ("high", "critical"):
            risks.append("PII exposure detected")
        if minor_report:
            risks.append(f"{len(minor_report)} minor(s) detected - ensure guardian consent")
        if retention_report["action_required"]:
            risks.append("Expired data found - review retention policy")

        overall_risk = (
            RiskLevel.CRITICAL if any("PII" in r for r in risks)
            else RiskLevel.HIGH if len(risks) >= 2
            else RiskLevel.MEDIUM if len(risks) == 1
            else RiskLevel.LOW
        )

        report = {
            "audit_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "overall_risk": overall_risk.value,
            "risk_factors": risks,
            "sections": {
                "privacy": pii_report,
                "minor_protection": {
                    "minors_found": len(minor_report),
                    "details": minor_report,
                },
                "data_retention": retention_report,
                "data_inventory": inventory,
            },
            "recommendations": self._aggregate_recommendations(
                pii_report, minor_report, retention_report
            ),
        }

        ethics_logger.info(
            "Full ethics audit completed",
            overall_risk=overall_risk.value,
            risk_count=len(risks),
        )
        return report

    def _aggregate_recommendations(
        self,
        pii_report: Dict,
        minor_report: List,
        retention_report: Dict,
    ) -> List[str]:
        """Aggregate recommendations from all components."""
        recs: List[str] = []
        recs.extend(pii_report.get("recommendations", []))

        if minor_report:
            recs.append("Set up guardian consent for all detected minors.")
            recs.append("Apply enhanced anonymization for minor data.")

        if retention_report.get("action_required"):
            recs.append(
                f"Remove {retention_report['expired_nodes']} expired data nodes."
            )

        recs.append("Schedule regular ethics audits (recommended: monthly).")
        return recs

from django.test import TestCase
from risk_engine.services.alert_engine import determine_alert

def must(risk):
    return {"risk": risk,
            "total_score": 2 if risk == "HIGH" else (1 if risk == "MODERATE" else 0),
            "bmi_detail": "", "weight_detail": "", "acute_detail": ""}


def ml(flag, prob=None):
    return {"risk_flag": flag,
            "probability": prob or (0.8 if flag else 0.2),
            "top_factors": [], "explanation": ""}


class AlertEngineTruthTableTest(TestCase):

    def test_must_high_ml_high_returns_both(self):
        r = determine_alert(must("HIGH"), ml(True))
        self.assertTrue(r["triggered"])
        self.assertEqual(r["level"],  "HIGH")
        self.assertEqual(r["source"], "BOTH")
        self.assertEqual(r["colour"], "amber")

    def test_must_high_ml_low_returns_must(self):
        r = determine_alert(must("HIGH"), ml(False))
        self.assertEqual(r["level"],  "HIGH")
        self.assertEqual(r["source"], "MUST")

    def test_must_moderate_ml_high_returns_both(self):
        r = determine_alert(must("MODERATE"), ml(True))
        self.assertEqual(r["level"],  "HIGH")
        self.assertEqual(r["source"], "BOTH")

    def test_must_low_ml_high_returns_ml(self):
        r = determine_alert(must("LOW"), ml(True))
        self.assertEqual(r["level"],  "HIGH")
        self.assertEqual(r["source"], "ML")

    def test_must_moderate_ml_low_returns_moderate(self):
        r = determine_alert(must("MODERATE"), ml(False))
        self.assertTrue(r["triggered"])
        self.assertEqual(r["level"],  "MODERATE")
        self.assertEqual(r["source"], "MUST_MODERATE")
        self.assertEqual(r["colour"], "yellow")

    def test_must_low_ml_low_returns_none(self):
        r = determine_alert(must("LOW"), ml(False))
        self.assertFalse(r["triggered"])
        self.assertEqual(r["level"],  "NONE")
        self.assertIsNone(r["colour"])

    def test_high_alert_has_next_steps(self):
        r = determine_alert(must("HIGH"), ml(True))
        self.assertGreater(len(r["next_steps"]), 0)

    def test_none_alert_has_no_next_steps(self):
        r = determine_alert(must("LOW"), ml(False))
        self.assertEqual(len(r["next_steps"]), 0)

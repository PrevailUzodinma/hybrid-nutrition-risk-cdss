from django.test import TestCase
from datetime import date, timedelta
from unittest.mock import MagicMock
from risk_engine.services.must_engine import calculate_must


def make_consultation(bmi_override=None, weight_kg=70.0, height_cm=170.0,
                      acute=False, consult_date=None):
    c = MagicMock()
    c.weight_kg = weight_kg
    c.height_cm = height_cm
    c.acute_illness_flag = acute
    c.consultation_date  = consult_date or date.today()
    c.bmi = bmi_override if bmi_override else round(weight_kg / ((height_cm / 100) ** 2), 1)
    return c


def empty_qs():
    qs = MagicMock()
    qs.exists.return_value = False
    # Configure the chained calls to return mocks that also have exists() = False
    empty_mock = MagicMock()
    empty_mock.exists.return_value = False
    empty_mock.filter.return_value = empty_mock
    empty_mock.exclude.return_value = empty_mock
    empty_mock.order_by.return_value = empty_mock
    empty_mock.first.return_value = None
    
    qs.filter.return_value = empty_mock
    qs.exclude.return_value = empty_mock
    qs.order_by.return_value = empty_mock
    qs.first.return_value = None
    
    return qs
# test that BMI scoring works as expected for different BMI values, including edge cases at the thresholds.
class BMIScoreTest(TestCase):
    def test_bmi_below_18_5_scores_2(self):
        self.assertEqual(calculate_must(make_consultation(bmi_override=17.9), empty_qs())["bmi_score"], 2)

    def test_bmi_at_18_5_scores_1(self):
        self.assertEqual(calculate_must(make_consultation(bmi_override=18.5), empty_qs())["bmi_score"], 1)

    def test_bmi_at_20_scores_1(self):
        self.assertEqual(calculate_must(make_consultation(bmi_override=20.0), empty_qs())["bmi_score"], 1)

    def test_bmi_above_20_scores_0(self):
        self.assertEqual(calculate_must(make_consultation(bmi_override=23.5), empty_qs())["bmi_score"], 0)

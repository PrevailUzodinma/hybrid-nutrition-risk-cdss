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

# test that weight loss scoring works as expected for different percentages of weight loss, including edge cases at the thresholds.
# I will have to mock the queryset so that it can return a prior consultation with the specified weight and date.
class WeightLossScoreTest(TestCase):
    def _qs_with_prior(self, current_weight, prior_weight, days_prior=120):
        today   = date.today()
        current = make_consultation(weight_kg=current_weight, consult_date=today)
        prior   = MagicMock()
        prior.weight_kg         = prior_weight
        prior.consultation_date = today - timedelta(days=days_prior)
        qs = MagicMock()
        qs.exists.return_value = True
        
        # Configure the chained calls to properly handle filter().exclude().order_by()
        # and subsequent filter().order_by().first() to return the prior mock
        chained_mock = MagicMock()
        chained_mock.exists.return_value = True
        chained_mock.filter.return_value.order_by.return_value.first.return_value = prior
        chained_mock.exclude.return_value = chained_mock
        chained_mock.order_by.return_value = chained_mock
        
        qs.filter.return_value = chained_mock
        qs.exclude.return_value = chained_mock
        qs.order_by.return_value = chained_mock
        
        return current, qs

    def test_loss_over_10_pct_scores_2(self):
        current, qs = self._qs_with_prior(55.0, 62.0)
        self.assertEqual(calculate_must(current, qs)["weight_loss_score"], 2)

    def test_loss_5_to_10_pct_scores_1(self):
        current, qs = self._qs_with_prior(60.0, 65.0)
        self.assertEqual(calculate_must(current, qs)["weight_loss_score"], 1)

    def test_loss_under_5_pct_scores_0(self):
        current, qs = self._qs_with_prior(69.0, 70.0)
        self.assertEqual(calculate_must(current, qs)["weight_loss_score"], 0)
# test that the acute illness flag contributes the correct score to the total, and that it is correctly included in the overall risk calculation.
class AcuteScoreTest(TestCase):
    def test_acute_true_scores_2(self):
        self.assertEqual(calculate_must(make_consultation(acute=True), empty_qs())["acute_score"], 2)

    def test_acute_false_scores_0(self):
        self.assertEqual(calculate_must(make_consultation(acute=False), empty_qs())["acute_score"], 0)

# test that the overall risk category is determined correctly based on the total score, including edge cases at the thresholds.
class RiskCategoryTest(TestCase):
    def test_score_0_is_low(self):
        self.assertEqual(calculate_must(make_consultation(bmi_override=25.0), empty_qs())["risk"], "LOW")

    def test_score_1_is_moderate(self):
        self.assertEqual(calculate_must(make_consultation(bmi_override=19.0), empty_qs())["risk"], "MODERATE")

    def test_score_2_is_high(self):
        self.assertEqual(calculate_must(make_consultation(bmi_override=17.0), empty_qs())["risk"], "HIGH")

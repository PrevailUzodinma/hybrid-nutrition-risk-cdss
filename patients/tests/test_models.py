from django.test import TestCase
from datetime import date
from patients.models import Patient, Consultation


class PatientAgePropertyTest(TestCase):
    def test_age_calculated_correctly(self):
        patient = Patient(date_of_birth=date(1950, 1, 1))
        # Don't need to save to DB — testing a property calculation
        expected = date.today().year - 1950  # simplified; edge case handled in model
        self.assertAlmostEqual(patient.age, expected, delta=1)


class ConsultationBMIPropertyTest(TestCase):
    def test_bmi_calculated_correctly(self):
        c = Consultation(weight_kg=70.0, height_cm=175.0)
        self.assertAlmostEqual(c.bmi, 22.9, places=1)

    def test_bmi_returns_none_when_weight_missing(self):
        c = Consultation(weight_kg=None, height_cm=175.0)
        self.assertIsNone(c.bmi)

    def test_bmi_returns_none_when_height_missing(self):
        c = Consultation(weight_kg=70.0, height_cm=None)
        self.assertIsNone(c.bmi)

    def test_bmi_returns_none_when_height_zero(self):
        c = Consultation(weight_kg=70.0, height_cm=0.0)
        self.assertIsNone(c.bmi)


class ConsultationPolypharmacyFlagTest(TestCase):
    def test_polypharmacy_true_at_5_medications(self):
        c = Consultation(medication_count=5)
        self.assertTrue(c.polypharmacy_flag)

    def test_polypharmacy_false_below_5(self):
        c = Consultation(medication_count=4)
        self.assertFalse(c.polypharmacy_flag)
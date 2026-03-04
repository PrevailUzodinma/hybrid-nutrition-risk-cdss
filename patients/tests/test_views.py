from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from datetime import date
from patients.models import Patient

class PatientRecordViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user("testdoc2", password="pass")
        self.patient = Patient.objects.create(
            nhs_number="999 000 0001",
            first_name="Test",
            last_name="Patient",
            date_of_birth=date(1945, 1, 1),
            sex="F",
            initial_height_cm=160.0,
        )

    def test_200_for_valid_patient(self):
        self.client.login(username="testdoc2", password="pass")
        response = self.client.get(reverse("patients:patient_record", args=[self.patient.id]))
        self.assertEqual(response.status_code, 200)

    def test_404_for_invalid_patient(self):
        self.client.login(username="testdoc2", password="pass")
        response = self.client.get(reverse("patients:patient_record", args=[99999]))
        self.assertEqual(response.status_code, 404)

    def test_correct_template(self):
        self.client.login(username="testdoc2", password="pass")
        response = self.client.get(reverse("patients:patient_record", args=[self.patient.id]))
        self.assertTemplateUsed(response, "patients/patient_record.html")

    def test_alert_result_in_context(self):
        self.client.login(username="testdoc2", password="pass")
        response = self.client.get(reverse("patients:patient_record", args=[self.patient.id]))
        self.assertIn("alert_result", response.context)
        self.assertTrue(response.context["alert_result"]["triggered"])  # we'll hardcode to True
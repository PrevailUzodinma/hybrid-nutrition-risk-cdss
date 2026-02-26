from django.test import TestCase
from django.contrib.auth.models import User
from patients.models import Patient
from appointments.models import Appointment
from datetime import date, time

class AppointmentModelTest(TestCase):
    def setUp(self):
        self.patient = Patient.objects.create(
            nhs_number="999 000 0002",
            first_name="Test",
            last_name="Patient",
            date_of_birth=date(1950, 1, 1),
            sex="M",
            initial_height_cm=170
        )
        self.user = User.objects.create_user(username="doc", password="pass")

    def test_appointment_creation(self):
        appt = Appointment.objects.create(
            patient=self.patient,
            appointment_date=date.today(),
            appointment_time=time(9, 0),
            clinician=self.user,
            reason="Checkup",
            status="BOOKED"
        )
        self.assertEqual(str(appt), f"Test Patient, Patient (999 000 0002) @ 09:00:00 on {date.today()}")
        self.assertEqual(appt.patient, self.patient)
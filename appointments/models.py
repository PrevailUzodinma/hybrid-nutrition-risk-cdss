# appointments/models.py
from django.db import models
from django.contrib.auth.models import User
from patients.models import Patient

class Appointment(models.Model):
    STATUS_CHOICES = [
        ("BOOKED", "Booked"),
        ("IN_PROGRESS", "In Progress"),
        ("COMPLETED", "Completed"),
        ("CANCELLED", "Cancelled"),
        ("DID_NOT_ATTEND", "Did Not Attend"),
    ]
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="appointments")
    appointment_date = models.DateField()
    appointment_time = models.TimeField()
    clinician = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    reason = models.CharField(max_length=300, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="BOOKED")

    class Meta:
        ordering = ["appointment_time"]

    def __str__(self):
        return f"{self.patient} @ {self.appointment_time} on {self.appointment_date}"
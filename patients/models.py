from django.db import models
from django.contrib.auth.models import User


class Patient(models.Model):
    SEX_CHOICES = [("M", "Male"), ("F", "Female")]

    nhs_number        = models.CharField(max_length=12, unique=True)
    first_name        = models.CharField(max_length=100)
    last_name         = models.CharField(max_length=100)
    date_of_birth     = models.DateField()
    sex               = models.CharField(max_length=1, choices=SEX_CHOICES)
    initial_height_cm = models.FloatField()
    created_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["last_name", "first_name"]

    def __str__(self):
        return f"{self.last_name}, {self.first_name} ({self.nhs_number})"

    @property
    def age(self):
        from datetime import date
        today = date.today()
        b = self.date_of_birth
        return today.year - b.year - ((today.month, today.day) < (b.month, b.day))


class Condition(models.Model):
    patient        = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="conditions")
    condition_name = models.CharField(max_length=200)
    diagnosis_date = models.DateField()
    is_active      = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.condition_name} ({self.patient})"


class Medication(models.Model):
    patient         = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="medications")
    medication_name = models.CharField(max_length=200)
    dosage          = models.CharField(max_length=100, blank=True)
    start_date      = models.DateField()
    end_date        = models.DateField(null=True, blank=True)

    def __str__(self):
        return f"{self.medication_name} ({self.patient})"


class Consultation(models.Model):
    patient            = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="consultations")
    consultation_date  = models.DateField()
    weight_kg          = models.FloatField(null=True, blank=True)
    height_cm          = models.FloatField(null=True, blank=True)
    albumin_gdl        = models.FloatField(null=True, blank=True)
    haemoglobin_gdl    = models.FloatField(null=True, blank=True)
    hba1c_pct          = models.FloatField(null=True, blank=True)
    bp_systolic        = models.IntegerField(null=True, blank=True)
    bp_diastolic       = models.IntegerField(null=True, blank=True)
    acute_illness_flag = models.BooleanField(default=False)
    medication_count   = models.IntegerField(default=0)
    comorbidity_count  = models.IntegerField(default=0)
    notes              = models.TextField(blank=True)
    created_by         = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    class Meta:
        ordering       = ["-consultation_date"]
        unique_together = ["patient", "consultation_date"]

    def __str__(self):
        return f"{self.patient} — {self.consultation_date}"

    @property
    def bmi(self):
        if self.weight_kg and self.height_cm and self.height_cm > 0:
            return round(self.weight_kg / ((self.height_cm / 100) ** 2), 1)
        return None

    @property
    def polypharmacy_flag(self):
        return self.medication_count >= 5


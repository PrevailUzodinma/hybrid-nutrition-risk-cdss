from django.db import models
from patients.models import Consultation

# Stores the combined output of the MUST rule engine and ML model for a given consultation.
class RiskAssessment(models.Model):
    MUST_RISK_CHOICES = [
        ("LOW", "Low"),
        ("MODERATE", "Moderate"),
        ("HIGH", "High"),
    ]
    ALERT_SOURCE_CHOICES = [
        ("MUST", "MUST"),
        ("ML", "ML"),
        ("BOTH", "Both"),
        ("MUST_MODERATE","MUST Moderate"),
        ("NONE","None"),
    ]

    consultation = models.OneToOneField(
        Consultation, on_delete=models.CASCADE, related_name="risk_assessment"
    )

    # MUST component scores
    must_bmi_score         = models.IntegerField(default=0)   # 0, 1, or 2
    must_weight_loss_score = models.IntegerField(default=0)   # 0, 1, or 2
    must_acute_score       = models.IntegerField(default=0)   # 0 or 2
    must_total_score       = models.IntegerField(default=0)
    must_risk              = models.CharField(max_length=10, choices=MUST_RISK_CHOICES, default="LOW")
    must_explanation       = models.TextField(blank=True)

    # ML outputs
    ml_probability  = models.FloatField(null=True, blank=True)
    ml_risk_flag    = models.BooleanField(default=False)
    ml_top_factors  = models.JSONField(default=list)
    ml_explanation  = models.TextField(blank=True)

    # Unified alert
    alert_triggered = models.BooleanField(default=False)
    alert_level     = models.CharField(max_length=15, default="NONE")
    alert_source    = models.CharField(max_length=15, choices=ALERT_SOURCE_CHOICES, default="NONE")

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"RiskAssessment — {self.consultation} — alert={self.alert_triggered}"
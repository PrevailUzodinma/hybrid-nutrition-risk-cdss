from django.contrib import admin
from .models import RiskAssessment
@admin.register(RiskAssessment)
class RiskAssessmentAdmin(admin.ModelAdmin):
    list_display = ["consultation", "must_risk", "ml_probability",
                    "alert_triggered", "alert_source", "created_at"]
    list_filter  = ["alert_triggered", "must_risk", "alert_source"]
    search_fields = ["consultation__patient__last_name",
                     "consultation__patient__nhs_number"]
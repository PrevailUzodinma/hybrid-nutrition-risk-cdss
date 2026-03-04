from django.contrib import admin
from .models import Patient, Condition, Medication, Consultation


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display  = ["last_name", "first_name", "nhs_number", "date_of_birth", "sex", "age"]
    search_fields = ["last_name", "first_name", "nhs_number"]


@admin.register(Consultation)
class ConsultationAdmin(admin.ModelAdmin):
    list_display  = ["patient", "consultation_date", "weight_kg", "bmi",
                     "albumin_gdl", "acute_illness_flag"]
    list_filter   = ["consultation_date", "acute_illness_flag"]
    search_fields = ["patient__last_name", "patient__nhs_number"]


admin.site.register(Condition)
admin.site.register(Medication)

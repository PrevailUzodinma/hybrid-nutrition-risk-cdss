# appointments/admin.py
from django.contrib import admin
from .models import Appointment

@admin.register(Appointment)
class AppointmentAdmin(admin.ModelAdmin):
    list_display = ["patient", "appointment_date", "appointment_time", "status", "reason"]
    list_filter = ["appointment_date", "status"]
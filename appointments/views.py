from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from datetime import date
from .models import Appointment

@login_required
def appointment_list(request):
    today = date.today()
    appointments = Appointment.objects.filter(
        appointment_date=today,
        clinician=request.user,
    ).select_related("patient").order_by("appointment_time")
    if not appointments.exists():
        appointments = Appointment.objects.filter(
            appointment_date=today,
        ).select_related("patient").order_by("appointment_time")
    return render(request, "appointments/appointment_list.html", {
        "appointments": appointments,
        "today": today,
        "clinic_date_display": today.strftime("%A %d %B %Y"),
    })
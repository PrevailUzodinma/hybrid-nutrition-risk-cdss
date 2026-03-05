from datetime import date

from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from .models import Appointment


@login_required
def appointment_list(request):
    today = date.today()

    # Find the most recent date that has appointments in the database to ensure the demo always shows patients regardless of the time where it works.
    latest_date = (
        Appointment.objects
        .order_by("-appointment_date")
        .values_list("appointment_date", flat=True)
        .first()
    )
    clinic_date = latest_date if latest_date else today

    appointments = (
        Appointment.objects
        .filter(appointment_date=clinic_date, clinician=request.user)
        .select_related("patient")
        .order_by("appointment_time")
    )

    if not appointments.exists():
        appointments = (
            Appointment.objects
            .filter(appointment_date=clinic_date)
            .select_related("patient")
            .order_by("appointment_time")
        )

    return render(request, "appointments/appointment_list.html", {
        "appointments": appointments,
        "today_display": today.strftime("%A %d %B %Y"),
    })
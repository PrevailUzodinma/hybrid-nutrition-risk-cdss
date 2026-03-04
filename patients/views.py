from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Patient

@login_required
def patient_record(request, patient_id):
    patient = get_object_or_404(Patient, pk=patient_id)
    # dummy alert
    alert_result = {
        "triggered": True,
        "level": "HIGH",
        "source": "MUST",
        "colour": "amber",
        "emoji": "🟠",
        "next_steps": ["Dummy step 1", "Dummy step 2"],
        "summary_text": "This is a mock alert for demonstration.",
    }
    # I still need to pass some context variables that the template is expecting
    # (like latest_consultation and all_consultations so I will set them to None for now
    return render(request, "patients/patient_record.html", {
        "patient": patient,
        "latest_consultation": None,
        "all_consultations": [],
        "active_medications": [],
        "active_conditions": [],
        "must_result": None,
        "ml_result": None,
        "alert_result": alert_result,
        "risk_assessment": None,
        "chart_labels": [],
        "chart_weight": [],
        "chart_bmi": [],
        "chart_albumin": [],
    })
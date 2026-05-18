from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Patient
from risk_engine.services.must_engine import calculate_must
from risk_engine.services.ml_engine import score_patient
from risk_engine.services.alert_engine import determine_alert
from risk_engine.models import RiskAssessment

@login_required
def patient_record(request, patient_id):
    patient = get_object_or_404(Patient, pk=patient_id)

    all_consultations = patient.consultations.order_by("consultation_date")
    latest            = patient.consultations.order_by("-consultation_date").first()

    must_result      = None
    ml_result        = None
    ml_probability_pct = None
    alert_result     = {
        "triggered":    False,
        "level":        "NONE",
        "source":       "NONE",
        "colour":       None,
        "emoji":        "",
        "next_steps":   [],
        "summary_text": "",
    }
    risk_assessment = None
    if latest:
        must_result  = calculate_must(latest, all_consultations)
        ml_result    = score_patient(latest)
        alert_result = determine_alert(must_result, ml_result)


        ml_probability_pct = round(ml_result["probability"] * 100, 1)

        risk_assessment, _ = RiskAssessment.objects.update_or_create(
            consultation=latest,
            defaults={
                "must_bmi_score":         must_result["bmi_score"],
                "must_weight_loss_score": must_result["weight_loss_score"],
                "must_acute_score":       must_result["acute_score"],
                "must_total_score":       must_result["total_score"],
                "must_risk":              must_result["risk"],
                "must_explanation":       must_result["explanation"],
                "ml_probability":         ml_result["probability"],
                "ml_risk_flag":           ml_result["risk_flag"],
                "ml_top_factors":         ml_result["top_factors"],
                "ml_explanation":         ml_result["explanation"],
                "alert_triggered":        alert_result["triggered"],
                "alert_level":            alert_result["level"],
                "alert_source":           alert_result["source"],
            },
        )

     # Chart.js data chronologically showing last 10 consultations 
    chart_qs      = list(all_consultations.order_by("consultation_date")[:10])
    chart_labels  = [c.consultation_date.strftime("%b %Y") for c in chart_qs]
    chart_weight  = [c.weight_kg    for c in chart_qs]
    chart_bmi     = [c.bmi          for c in chart_qs]
    chart_albumin = [c.albumin_gdl  for c in chart_qs]

    # Height change: compare latest vs earliest consultation with a recorded height
    height_change_cm = None
    if latest and latest.height_cm:
        earliest_with_height = (
            all_consultations
            .exclude(height_cm__isnull=True)
            .order_by("consultation_date")
            .first()
        )
        if earliest_with_height and earliest_with_height.pk != latest.pk:
            height_change_cm = round(latest.height_cm - earliest_with_height.height_cm, 1)

    return render(request, "patients/patient_record.html", {
        "patient":              patient,
        "latest_consultation":  latest,
        "all_consultations":    patient.consultations.order_by("-consultation_date")[:5],
        "active_medications":   patient.medications.filter(end_date__isnull=True).order_by("start_date"),
        "active_conditions":    patient.conditions.filter(is_active=True).order_by("diagnosis_date"),
        "must_result":          must_result,
        "ml_result":            ml_result,
        "ml_probability_pct":   ml_probability_pct,
        "alert_result":         alert_result,
        "risk_assessment":      risk_assessment,
        "chart_labels":         chart_labels,
        "chart_weight":         chart_weight,
        "chart_bmi":            chart_bmi,
        "chart_albumin":        chart_albumin,
        "height_change_cm":     height_change_cm,
    })
"""
Rule-based MUST (Malnutrition Universal Screening Tool) calculator.

MUST components:
1. BMI: 0 (>20.0) / 1 (18.5–20.0) / 2 (<18.5)
2. Weight loss: 0 (<5%) / 1 (5–10%) / 2 (>10%) over 3–6 months
3. Acute illness: 0 (none) / 2 (acutely ill + no nutritional intake >5 days)

Total score: 0 = LOW / 1 = MODERATE / >=2 = HIGH

Reference from BAPEN (2012). MUST explanatory booklet and NICE (2017). Nutrition support for adults.
"""

from datetime import timedelta

def calculate_must(consultation, all_consultations_qs):
    """
    Calculate full MUST score for a consultation.

    Arguments:
        consultation:         A patients.models.Consultation instance (current visit).
        all_consultations_qs: QuerySet of ALL consultations for this patient,
                              ordered by consultation_date in ascending order. I'll Use it to find
                              the weight reference visit for the 3-6 month window.

        Returns:
        dict with keys:
            bmi_score           (int: 0/1/2)
            weight_loss_score   (int: 0/1/2)
            acute_score         (int: 0/2)
            total_score         (int: 0-6)
            risk                (str: "LOW" / "MODERATE" / "HIGH")
            explanation         (str: full human-readable summary)
            bmi_detail          (str: per-component explanation)
            weight_detail       (str: per-component explanation)
            acute_detail        (str: per-component explanation)
            weight_loss_pct     (float | None)
    """

    # Component 1: BMI
    bmi = consultation.bmi
    if bmi is None:
        bmi_score = 0
        bmi_detail = "BMI not available (missing weight or height data)"
    elif bmi < 18.5:
        bmi_score = 2
        bmi_detail = f"BMI {bmi} - underweight (<18.5)"
    elif bmi <= 20.0:
        bmi_score = 1
        bmi_detail = f"BMI {bmi} - borderline (18.5-20.0)"
    else:
        bmi_score = 0
        bmi_detail = f"BMI {bmi} - normal (>20.0)"

    # Component 2: Unintentional weight loss over 3–6 months 
    weight_loss_score = 0
    weight_loss_pct   = None
    weight_detail     = "No prior weight data available within 3-6 month window"

    past = all_consultations_qs.filter(
        consultation_date__lt=consultation.consultation_date
    ).exclude(weight_kg__isnull=True).order_by("consultation_date")

    if past.exists() and consultation.weight_kg:
        current_date = consultation.consultation_date
        window_start = current_date - timedelta(days=180)
        window_end   = current_date - timedelta(days=60)

        reference = past.filter(
            consultation_date__gte=window_start,
            consultation_date__lte=window_end,
        ).order_by("-consultation_date").first()

        # fallback: 30–210 days if there is nothing in ideal window
        if not reference:
            reference = past.filter(
                consultation_date__gte=current_date - timedelta(days=210),
                consultation_date__lt=current_date,
            ).order_by("-consultation_date").first()

        if reference and reference.weight_kg:
            weight_loss_pct = (
                (reference.weight_kg - consultation.weight_kg) / reference.weight_kg
            ) * 100

            if weight_loss_pct > 10:
                weight_loss_score = 2
                weight_detail = (
                    f"Unintentional weight loss >10%: {weight_loss_pct:.1f}% "
                    f"({reference.weight_kg:.1f} kg → {consultation.weight_kg:.1f} kg "
                    f"since {reference.consultation_date})"
                )
            elif weight_loss_pct > 5:
                weight_loss_score = 1
                weight_detail = (
                    f"Unintentional weight loss 5-10%: {weight_loss_pct:.1f}% "
                    f"({reference.weight_kg:.1f} kg → {consultation.weight_kg:.1f} kg "
                    f"since {reference.consultation_date})"
                )
            elif weight_loss_pct > 0:
                weight_detail = f"Minor weight change: {weight_loss_pct:.1f}% — below threshold"
            else:
                weight_detail = f"No weight loss (change: {weight_loss_pct:.1f}%)"
    # Component 3: Acute illness 
    if consultation.acute_illness_flag:
        acute_score  = 2
        acute_detail = "Acute illness with no nutritional intake >5 days — score 2"
    else:
        acute_score  = 0
        acute_detail = "No acute illness effect"

    # Total score + risk category 
    total_score = bmi_score + weight_loss_score + acute_score

    if total_score >= 2:
        risk = "HIGH"
    elif total_score == 1:
        risk = "MODERATE"
    else:
        risk = "LOW"

    explanation = (
        f"MUST score = {total_score} ({risk} risk). "
        f"{bmi_detail}. {weight_detail}. {acute_detail}."
    )

    return {
        "bmi_score":         bmi_score,
        "weight_loss_score": weight_loss_score,
        "acute_score":       acute_score,
        "total_score":       total_score,
        "risk":              risk,
        "explanation":       explanation,
        "bmi_detail":        bmi_detail,
        "weight_detail":     weight_detail,
        "acute_detail":      acute_detail,
        "weight_loss_pct":   weight_loss_pct,
    }
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

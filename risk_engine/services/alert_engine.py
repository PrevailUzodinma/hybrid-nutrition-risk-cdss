"""
Unified alert logic. Combines MUST and ML outputs into a single alert signal.

Priority rules (6-row truth table):
    MUST HIGH + ML HIGH = 🟠 HIGH source=BOTH
    MUST HIGH + ML LOW = 🟠 HIGH source=MUST
    MUST MOD + ML HIGH = 🟠 HIGH source=BOTH (ML will elevate borderline)
    MUST LOW + ML HIGH = 🟠 HIGH source=ML   (ML will detect what rules miss)
    MUST MOD + ML LOW = 🟡 MOD source=MUST_MODERATE
    MUST LOW + ML LOW = no alert

Colour is chosen to be amber (not red) to differentiate from critical alerts.
This was a confirmed requirement from the pre-study with primary care nurses.
"""

ALERT_COLOUR = {"HIGH": "amber", "MODERATE": "yellow", "NONE": None}

NEXT_STEPS = {
    "MUST": [
        "Consider completing a full nutritional assessment at this visit",
        "Weigh patient and record height - repeat at every future appointment",
        "Refer to dietitian if MUST score remains elevated at next consultation",
        "Review medication list for drugs affecting appetite or nutrient absorption",
        "Discuss dietary intake and recent changes with patient",
    ],
    "ML": [
        "Review the combination of comorbidities and polypharmacy for nutritional impact",
        "Check serum albumin trend - declining albumin may indicate protein-energy malnutrition",
        "Consider referral to community dietitian for nutritional support",
        "Discuss appetite, eating habits, and social circumstances with patient",
        "Consider oral nutritional supplement if intake appears inadequate",
    ],
    "BOTH": [
        "Both clinical rule screening (MUST) and ML pattern analysis indicate elevated risk",
        "Prioritise nutritional assessment at this consultation",
        "Refer to dietitian or community nutrition support team",
        "Review full medication list - note appetite suppressants and drugs affecting absorption",
        "Consider blood tests: albumin, Hb, B12, folate, vitamin D",
        "Record weight and height at every future visit for trend monitoring",
    ],
    "MUST_MODERATE": [
        "MUST score = 1 (borderline). Monitor and repeat screening at next visit",
        "Discuss dietary history and recent weight changes with patient",
        "Encourage patient to report unintentional weight loss before next appointment",
    ],
}

#Determine the unified alert from MUST and ML results.
def determine_alert(must_result: dict, ml_result: dict) -> dict:
    must_risk = must_result["risk"]
    ml_high   = ml_result["risk_flag"]

    if must_risk == "HIGH" and ml_high:
        level, source = "HIGH", "BOTH"
    elif must_risk == "HIGH":
        level, source = "HIGH", "MUST"
    elif must_risk == "MODERATE" and ml_high:
        level, source = "HIGH", "BOTH"
    elif ml_high:
        level, source = "HIGH", "ML"
    elif must_risk == "MODERATE":
        level, source = "MODERATE", "MUST_MODERATE"
    else:
        level, source = "NONE", "NONE"

    triggered  = level != "NONE"
    colour     = ALERT_COLOUR[level]
    emoji      = "🟠" if level == "HIGH" else ("🟡" if level == "MODERATE" else "")
    next_steps = NEXT_STEPS.get(source, [])

    return {
        "triggered":    triggered,
        "level":        level,
        "source":       source,
        "colour":       colour,
        "emoji":        emoji,
        "next_steps":   next_steps,
        "summary_text": build_summary(level, source, must_result, ml_result),
    }

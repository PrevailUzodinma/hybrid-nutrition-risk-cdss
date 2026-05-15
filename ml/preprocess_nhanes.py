import pandas as pd

# Helper functions

def safe_read_csv(path: str, usecols: list[str]) -> pd.DataFrame:
    cols_in_file = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in usecols if c in cols_in_file]
    if not cols:
        raise ValueError(f"None of the requested columns were found in {path}. "
                         f"Requested: {usecols}")
    return pd.read_csv(path, usecols=cols, low_memory=False)

def yes1(series: pd.Series) -> pd.Series:
    # NHANES commonly uses 1=Yes, 2=No, 7/9=Refused/Don't know
    return (series == 1).astype(int)

# 1 Load ONLY needed columns from each file
demo = safe_read_csv("demographics.csv", ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"])
exam = safe_read_csv("examination.csv", ["SEQN", "BMXBMI", "BMXHT", "BMXWT"])
labs = safe_read_csv("laboratory.csv", ["SEQN", "LBXSAL", "LBXHGB"])
ques = safe_read_csv("questionnaire.csv", ["SEQN", "DIQ010", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160G", "MCQ220"])

# Medications: using RXDCOUNT to take number of medications
med = safe_read_csv("medications.csv", ["SEQN", "RXDCOUNT"])


# 2 BMI

exam_keep = exam.copy()
for c in ["BMXBMI", "BMXHT", "BMXWT"]:
    if c in exam_keep.columns:
        exam_keep[c] = pd.to_numeric(exam_keep[c], errors="coerce")

if "BMXBMI" in exam_keep.columns and exam_keep["BMXBMI"].notna().any():
    exam_keep["BMI"] = exam_keep["BMXBMI"]
else:
    # compute BMI from weight/height (height in cm - meters)
    exam_keep["BMI"] = exam_keep["BMXWT"] / ((exam_keep["BMXHT"] / 100) ** 2)

exam_keep = exam_keep[["SEQN", "BMI"]]


#  3 Labs (converted to numeric)

labs_keep = labs.copy()
for c in ["LBXSAL", "LBXHGB"]:
    if c in labs_keep.columns:
        labs_keep[c] = pd.to_numeric(labs_keep[c], errors="coerce")

# 4 Questionnaire - comorbidity flags + count

ques_keep = ques.copy()

# Diabetes
ques_keep["has_diabetes"] = yes1(ques_keep["DIQ010"]) if "DIQ010" in ques_keep.columns else 0

# Any CVD among CHD/angina/heart attack/stroke
cvd_cols = [c for c in ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E"] if c in ques_keep.columns]
ques_keep["has_cvd"] = (ques_keep[cvd_cols].eq(1).any(axis=1)).astype(int) if cvd_cols else 0

# COPD/emphysema
ques_keep["has_copd"] = yes1(ques_keep["MCQ160G"]) if "MCQ160G" in ques_keep.columns else 0

# Cancer
ques_keep["has_cancer"] = yes1(ques_keep["MCQ220"]) if "MCQ220" in ques_keep.columns else 0

# Simple comorbidity count
ques_keep["comorbidity_count"] = (
    ques_keep["has_diabetes"]
    + ques_keep["has_cvd"]
    + ques_keep["has_copd"]
    + ques_keep["has_cancer"]
)

ques_final = ques_keep[["SEQN", "has_diabetes", "has_cvd", "has_copd", "has_cancer", "comorbidity_count"]].copy()


# 5 Medications - med_count + polypharmacy flag

med_keep = med.copy()
med_keep["RXDCOUNT"] = pd.to_numeric(med_keep["RXDCOUNT"], errors="coerce")

# If there are multiple rows per SEQN, take max med count
med_count = med_keep.groupby("SEQN")["RXDCOUNT"].max().reset_index()
med_count = med_count.rename(columns={"RXDCOUNT": "med_count"})
med_count["med_count"] = med_count["med_count"].fillna(0).astype(int)

# Polypharmacy definition (>=5 meds)
med_count["polypharmacy_5plus"] = (med_count["med_count"] >= 5).astype(int)


# 6 Merge into master dataset (SEQN is the key)

df = demo.merge(exam_keep, on="SEQN", how="inner")
df = df.merge(labs_keep, on="SEQN", how="left")
df = df.merge(ques_final, on="SEQN", how="left")
df = df.merge(med_count, on="SEQN", how="left")

# Fill missing meds/comorbids with 0 (so if its "not recorded", I treat as absent)
df["med_count"] = df["med_count"].fillna(0).astype(int)
df["polypharmacy_5plus"] = df["polypharmacy_5plus"].fillna(0).astype(int)

for c in ["has_diabetes", "has_cvd", "has_copd", "has_cancer", "comorbidity_count"]:
    df[c] = df[c].fillna(0).astype(int)

# Ensure numeric core columns
df["RIDAGEYR"] = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
df["RIAGENDR"] = pd.to_numeric(df["RIAGENDR"], errors="coerce")
df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")

# Drop rows missing core inputs
df = df.dropna(subset=["RIDAGEYR", "RIAGENDR", "BMI"]).copy()

# Filter to age >= 65
df = df[df["RIDAGEYR"] >= 65].copy()
print(f"After age filter: {len(df)} rows (65+)")

# 7 Save and checks

df.to_csv("nhanes_master_cdss.csv", index=False)

print("Saved nhanes_master_cdss.csv", df.shape)
print("\nQuick checks:")
print("BMI missing rate:", df["BMI"].isna().mean())
print("Polypharmacy_5plus distribution:\n", df["polypharmacy_5plus"].value_counts(dropna=False))
print("Comorbidity_count distribution (top):\n", df["comorbidity_count"].value_counts().head())
print(df[["BMI","LBXSAL","LBXHGB"]].isna().mean())
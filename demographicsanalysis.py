import pandas as pd

# --- Load the CSV ---
# Qualtrics exports have 2 header rows; skip the second (question text row)
df = pd.read_csv("/Users/rachelsong/Desktop/Movesense/tutorial_dyadic_movesense/kitco3 demographics responses.csv", skiprows=[1])  # change filename as needed

# ── 1. Total participants ─────────────────────────────────────────────────────
total = len(df)
print(f"Total participants: {total}\n")

# ── 2. Mean age + SD ──────────────────────────────────────────────────────────
# Q25 = selected age bracket; Q25_1_TEXT = free-text numeric age
# Use the free-text numeric column where available, fall back to Q25
age_col = "Q25_1_TEXT"

ages = pd.to_numeric(df[age_col], errors="coerce")
print(f"Mean age: {ages.mean():.2f}  (SD = {ages.std():.2f})")
print(f"  (based on {ages.notna().sum()} participants with numeric age data)\n")

# ── 3. Ethnicity breakdown ────────────────────────────────────────────────────
# Q67 = cultural background (multi-select, pipe-separated or comma-separated)
ethnicity_col = "Q67"

# Explode multi-select responses (Qualtrics uses comma separation)
eth_series = (
    df[ethnicity_col]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
)

eth_counts = eth_series.value_counts()
eth_pct    = eth_series.value_counts(normalize=True) * 100

eth_df = pd.DataFrame({"Count": eth_counts, "Percent (%)": eth_pct.round(1)})
print("Ethnicity / Cultural background:")
print(eth_df.to_string())
print()

# ── 4. Gender breakdown ───────────────────────────────────────────────────────
# Q27 = gender identity (selected choice); Q27_5_TEXT = free-text "Other"
gender_col      = "Q27"
gender_other_col = "Q27_5_TEXT"

# Replace "Other (please specify)" rows with the actual text they entered
gender = df[gender_col].copy()
mask = gender.str.contains("Other", case=False, na=False)
gender.loc[mask] = df.loc[mask, gender_other_col].fillna("Other")

gen_counts = gender.value_counts()
gen_pct    = gender.value_counts(normalize=True) * 100

gen_df = pd.DataFrame({"Count": gen_counts, "Percent (%)": gen_pct.round(1)})
print("Gender identity:")
print(gen_df.to_string())
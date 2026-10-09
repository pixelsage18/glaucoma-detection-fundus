import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Settings
# --------------------------------------------------

RANDOM_SEED = 42

metadata_path = "data/processed/metadata.csv"
output_path = "data/processed/metadata_with_splits.csv"


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

df = pd.read_csv(metadata_path)


# --------------------------------------------------
# Give every patient a category for stratification
#
# healthy_only  -> all usable eyes are healthy
# glaucoma_only -> all usable eyes are glaucoma
# mixed         -> one healthy eye and one glaucoma eye
# --------------------------------------------------

def get_patient_group(labels):
    unique_labels = set(labels)

    if unique_labels == {0}:
        return "healthy_only"

    elif unique_labels == {1}:
        return "glaucoma_only"

    else:
        return "mixed"


patient_df = (
    df.groupby("patient_id")["label"]
    .apply(list)
    .reset_index()
)

patient_df["patient_group"] = patient_df["label"].apply(
    get_patient_group
)


print("Patient groups:")
print(patient_df["patient_group"].value_counts())


# --------------------------------------------------
# First split:
# 70% train
# 30% temporary
# --------------------------------------------------

train_patients, temp_patients = train_test_split(
    patient_df,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=patient_df["patient_group"]
)


# --------------------------------------------------
# Split the temporary 30% equally:
# 15% validation
# 15% test
# --------------------------------------------------

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_patients["patient_group"]
)


# --------------------------------------------------
# Create sets of patient IDs
# --------------------------------------------------

train_ids = set(train_patients["patient_id"])
val_ids = set(val_patients["patient_id"])
test_ids = set(test_patients["patient_id"])


# --------------------------------------------------
# Safety check:
# no patient should occur in more than one split
# --------------------------------------------------

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)


# --------------------------------------------------
# Assign each image its patient's split
# --------------------------------------------------

def assign_split(patient_id):
    if patient_id in train_ids:
        return "train"

    elif patient_id in val_ids:
        return "val"

    elif patient_id in test_ids:
        return "test"

    else:
        raise ValueError(f"Patient {patient_id} was not assigned to a split.")


df["split"] = df["patient_id"].apply(assign_split)


# --------------------------------------------------
# Print split statistics
# --------------------------------------------------

print("\nPatients per split:")
print(
    df.groupby("split")["patient_id"]
    .nunique()
)

print("\nImages per split:")
print(
    df["split"]
    .value_counts()
)

print("\nClass distribution per split:")
print(
    pd.crosstab(
        df["split"],
        df["class_name"]
    )
)


# --------------------------------------------------
# Final leakage check
# --------------------------------------------------

patient_split_counts = (
    df.groupby("patient_id")["split"]
    .nunique()
)

leaking_patients = (
    patient_split_counts > 1
).sum()

print("\nPatients appearing in multiple splits:")
print(leaking_patients)


# --------------------------------------------------
# Save metadata with split information
# --------------------------------------------------

df.to_csv(
    output_path,
    index=False
)

print(f"\nSaved: {output_path}")
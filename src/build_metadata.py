import os
import pandas as pd


def load_eye_data(excel_path, eye):
    """
    Load one clinical Excel file and keep only the columns
    needed for our image classification project.
    """

    # Read Excel file
    df = pd.read_excel(excel_path)

    # Remove the two extra metadata/header rows
    df = df.iloc[2:].copy()

    # Keep only patient ID and diagnosis
    df = df[["Unnamed: 0", "Diagnosis"]]

    # Rename patient ID column
    df = df.rename(columns={"Unnamed: 0": "patient_id"})

    # Convert diagnosis values to integers
    df["Diagnosis"] = df["Diagnosis"].astype(int)

    # Add eye information
    df["eye"] = eye

    return df


# --------------------------------------------------
# Load clinical data
# --------------------------------------------------

od_df = load_eye_data(
    "data/raw/PAPILA/ClinicalData/patient_data_od.xlsx",
    "OD"
)

os_df = load_eye_data(
    "data/raw/PAPILA/ClinicalData/patient_data_os.xlsx",
    "OS"
)


# --------------------------------------------------
# Combine both eyes
# --------------------------------------------------

df = pd.concat([od_df, os_df], ignore_index=True)


# --------------------------------------------------
# Keep only healthy and glaucoma cases
#
# PAPILA diagnosis codes:
# 0 = healthy
# 1 = glaucoma
# 2 = suspect
#
# We exclude suspect cases because this project is
# binary classification: healthy vs glaucoma.
# --------------------------------------------------

df = df[df["Diagnosis"].isin([0, 1])].copy()


# Rename Diagnosis to label
df = df.rename(columns={"Diagnosis": "label"})


# --------------------------------------------------
# Create image paths
# --------------------------------------------------

def create_image_path(row):
    """
    Convert a patient ID such as #002 and eye OD
    into the corresponding image path:
    RET002OD.jpg
    """

    # "#002" -> "002"
    patient_number = row["patient_id"].replace("#", "")

    filename = f"RET{patient_number}{row['eye']}.jpg"

    image_path = os.path.join(
        "data/raw/PAPILA/FundusImages",
        filename
    )

    return image_path


df["image_path"] = df.apply(
    create_image_path,
    axis=1
)


# --------------------------------------------------
# Add readable class names
# --------------------------------------------------

df["class_name"] = df["label"].map(
    {
        0: "healthy",
        1: "glaucoma"
    }
)


# --------------------------------------------------
# Check whether image files actually exist
# --------------------------------------------------

df["image_exists"] = df["image_path"].apply(os.path.exists)


print("Records before removing missing images:", len(df))

print(
    "Missing image files:",
    (~df["image_exists"]).sum()
)


# Keep only rows whose image file exists
df = df[df["image_exists"]].copy()


# --------------------------------------------------
# Arrange columns neatly
# --------------------------------------------------

df = df[
    [
        "patient_id",
        "eye",
        "image_path",
        "label",
        "class_name",
        "image_exists"
    ]
]


# --------------------------------------------------
# Sanity checks
# --------------------------------------------------

print("\nTotal usable images:")
print(len(df))

print("\nClass distribution:")
print(df["class_name"].value_counts())

print("\nUnique patients:")
print(df["patient_id"].nunique())

print("\nFirst 10 rows:")
print(df.head(10))


# --------------------------------------------------
# Save processed metadata
# --------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

output_path = "data/processed/metadata.csv"

df.to_csv(
    output_path,
    index=False
)

print(f"\nSaved: {output_path}")
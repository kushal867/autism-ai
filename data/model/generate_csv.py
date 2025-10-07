import argparse
import pandas as pd
import numpy as np


def generate_correlated_scores(num_rows: int, seed: int) -> np.ndarray:
    """Generate 12 correlated latent traits then map to 0-5 integer scale.
    Order (12):
      SI, Comm, Eye, Name, SocImag, JointAttn, RRB, RestrInt, Routine, Stereo, SensSens, SensSeek
    """
    rng = np.random.default_rng(seed)

    # Base correlations: social cluster, RRB cluster, sensory cluster
    # Correlation matrix is positive semi-definite
    corr = np.array([
        # SI   Co    Eye   Name  SIm   JAtt  RRB   RInt  Rout  Ster  Sens  SSeek
        [1.0, 0.6,  0.55, 0.45, 0.50, 0.55, -0.35, -0.30, -0.25, -0.20, -0.30, -0.10],  # SI
        [0.6, 1.0,  0.55, 0.50, 0.55, 0.60, -0.30, -0.25, -0.20, -0.20, -0.25, -0.10],  # Comm
        [0.55,0.55, 1.0,  0.50, 0.45, 0.45, -0.25, -0.20, -0.15, -0.10, -0.20, -0.10],  # Eye
        [0.45,0.50, 0.50, 1.0,  0.40, 0.45, -0.20, -0.15, -0.15, -0.10, -0.15, -0.10],  # Name
        [0.50,0.55, 0.45, 0.40, 1.0,  0.55, -0.25, -0.20, -0.20, -0.15, -0.20, -0.10],  # SocImag
        [0.55,0.60, 0.45, 0.45, 0.55, 1.0,  -0.25, -0.20, -0.20, -0.15, -0.20, -0.10],  # JointAttn
        [-0.35,-0.30,-0.25,-0.20,-0.25,-0.25, 1.0,  0.55,  0.45,  0.40,  0.35,  0.30],  # RRB
        [-0.30,-0.25,-0.20,-0.15,-0.20,-0.20, 0.55,  1.0,  0.50,  0.40,  0.35,  0.25],  # RestrInt
        [-0.25,-0.20,-0.15,-0.15,-0.20,-0.20, 0.45,  0.50,  1.0,  0.45,  0.30,  0.20],  # Routine
        [-0.20,-0.20,-0.10,-0.10,-0.15,-0.15, 0.40,  0.40,  0.45,  1.0,   0.25,  0.15],  # Stereo
        [-0.30,-0.25,-0.20,-0.15,-0.20,-0.20, 0.35,  0.35,  0.30,  0.25,  1.0,   0.40],  # SensSens
        [-0.10,-0.10,-0.10,-0.10,-0.10,-0.10, 0.30,  0.25,  0.20,  0.15,  0.40,  1.0 ],  # SensSeek
    ])

    # Cholesky for correlated normals
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((num_rows, 12)).dot(L.T)

    # Map N(0,1) to 0..5 integers via quantiles per column
    quantiles = np.linspace(0, 1, 7)
    bins = np.quantile(Z, quantiles, axis=0)
    X = np.zeros_like(Z, dtype=int)
    for j in range(Z.shape[1]):
        X[:, j] = np.digitize(Z[:, j], bins[:, j][1:-1], right=False)
    return X


def apply_age_effects(X: np.ndarray, age_groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply age-specific shifts: younger groups tend to have lower social/comm scores,
    older may show more coping, but higher restricted interests.
    """
    X = X.astype(float)
    SI, CO, EYE, NAME, SIM, JATT, RRB, RINT, ROUT, STER, SSEN, SSEE = range(12)

    for idx, age in enumerate(age_groups):
        if age == 'toddler':
            X[idx, [SI, CO, EYE, NAME, JATT]] -= 0.6
            X[idx, [RRB, RINT, ROUT, STER, SSEN]] += 0.3
        elif age == 'preschool':
            X[idx, [SI, CO, JATT]] -= 0.3
            X[idx, [RRB, RINT, ROUT, SSEN]] += 0.2
        elif age == 'school_age':
            X[idx, [SI, CO, EYE, NAME]] += 0.2
            X[idx, [RRB, RINT]] += 0.1
        elif age == 'adolescent':
            X[idx, [SI, CO, EYE, NAME, SIM]] += 0.4
            X[idx, [RINT, ROUT]] += 0.2

    X = np.clip(np.rint(X), 0, 5)
    return X.astype(int)


def compute_asd_label(features: np.ndarray, age_groups: np.ndarray, rng: np.random.Generator,
                      target_ratio: float) -> np.ndarray:
    """Compute ASD labels via a propensity model with age interaction, then calibrate to target ratio."""
    SI, CO, EYE, NAME, SIM, JATT, RRB, RINT, ROUT, STER, SSEN, SSEE = range(12)
    f = features

    # Lower social/comm/eye/name increase risk (inverse), higher RRB/sensory increase risk
    risk = (
        (5 - f[:, SI]) * 0.22 +
        (5 - f[:, CO]) * 0.20 +
        (5 - f[:, EYE]) * 0.18 +
        (5 - f[:, NAME]) * 0.10 +
        (5 - f[:, JATT]) * 0.10 +
        (5 - f[:, SIM]) * 0.06 +
        f[:, RRB] * 0.18 +
        f[:, RINT] * 0.14 +
        f[:, ROUT] * 0.12 +
        f[:, STER] * 0.08 +
        f[:, SSEN] * 0.14 +
        f[:, SSEE] * 0.06
    )

    # Age interaction: toddlers/preschool slightly higher baseline risk
    age_bias = np.where(np.isin(age_groups, ['toddler', 'preschool']), 0.6, 0.2)
    risk = risk + age_bias

    # Convert to probabilities using logistic
    prob = 1 / (1 + np.exp(-(risk - np.median(risk))))

    # Calibrate threshold to match target ratio
    thresh = np.quantile(prob, 1 - target_ratio)
    y = (prob >= thresh).astype(int)

    # Introduce label noise (flip 10% of labels)
    noise_mask = rng.random(len(y)) < 0.10
    y[noise_mask] = 1 - y[noise_mask]
    return y


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic behavioral ASD dataset")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--asd_ratio", type=float, default=0.20, help="Target ASD prevalence (0-1)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Age distribution (roughly realistic)
    age_groups = rng.choice(
        ['toddler', 'preschool', 'school_age', 'adolescent'],
        size=args.n,
        p=[0.25, 0.30, 0.35, 0.10]
    )

    # Generate correlated latent traits and apply age effects
    X12 = generate_correlated_scores(args.n, args.seed)
    X12 = apply_age_effects(X12, age_groups, rng)

    # Compute labels with calibrated prevalence
    y = compute_asd_label(X12, age_groups, rng, target_ratio=args.asd_ratio)

    # Additional clinical features derived with mild correlation to difficulties
    emotional_regulation = np.clip(5 - (X12[:, 0] * 0.3 + X12[:, 1] * 0.3 + rng.normal(2.0, 1.0, args.n)), 0, 5)
    transitions_difficulty = np.clip((X12[:, 8] * 0.5 + X12[:, 6] * 0.3 + rng.normal(1.5, 1.0, args.n)), 0, 5)
    emotional_regulation = np.rint(emotional_regulation).astype(int)
    transitions_difficulty = np.rint(transitions_difficulty).astype(int)

    # Assemble dataset
    cols = [
        "Social_Interaction", "Communication_Skills", "Eye_Contact", "Response_to_Name",
        "Social_Imagination", "Joint_Attention", "Repetitive_Behaviors", "Restricted_Interests",
        "Routine_Rigidity", "Stereotyped_Movements", "Sensory_Sensitivities", "Sensory_Seeking"
    ]
    df = pd.DataFrame(X12, columns=cols)
    df["Emotional_Regulation"] = emotional_regulation
    df["Transitions_Difficulty"] = transitions_difficulty
    df["Age_Group"] = age_groups
    df["ASD"] = y

    # Sanity checks
    assert df.isna().sum().sum() == 0, "NaNs detected in generated dataset"

    # Save
    out_path = "data/behavioral_dataset.csv"
    df.to_csv(out_path, index=False)

    # Summary
    print(f"behavioral_dataset.csv created successfully with {len(df)} rows.")
    print("Class balance (ASD=1):", df["ASD"].mean().round(3))
    print("Age distribution:")
    print(df["Age_Group"].value_counts(normalize=True).round(3))
    print(df.head(10))


if __name__ == "__main__":
    main()

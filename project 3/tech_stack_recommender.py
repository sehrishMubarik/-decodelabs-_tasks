"""
DecodeLabs - Project 3: AI Recommendation Logic
Capstone: Tech Stack Recommender
--------------------------------------------------
Goal: Map a user's raw skills/interests to the closest-matching job
roles using pure content-based filtering (no neural nets, no
historical user-behavior data needed).

Pipeline (matches the slide deck's 4-step assembly line):

    1. INGESTION -> capture the user's skills (min. 3 inputs)
    2. SCORING   -> convert skills + job roles into TF-IDF vectors,
                    then compute Cosine Similarity between the user
                    profile and every job role
    3. SORTING   -> rank roles by similarity score, descending
    4. FILTERING -> truncate to a Top-N list to avoid choice overload

Why TF-IDF + Cosine Similarity instead of raw keyword overlap?
  - TF-IDF down-weights generic/common skills (e.g. "python" appearing
    everywhere) and up-weights distinctive ones, so matches are more
    meaningful than a simple 1s-and-0s overlap count.
  - Cosine similarity measures the ANGLE between vectors, not their
    raw magnitude — so a user who lists 3 skills isn't unfairly
    penalized against a job description containing 20 skill words.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# STEP 1: INGESTION
# ---------------------------------------------------------------------
def load_job_roles(csv_path="raw_skills.csv"):
    """Load the item dataset: job roles and their associated skills."""
    df = pd.read_csv(csv_path)
    return df


def ingest_user_skills(min_skills=3):
    """Capture user state via CLI input. Requires at least `min_skills`
    entries to ensure sufficient data density for accurate matching."""
    print(f"Enter at least {min_skills} skills or interests (comma-separated):")
    raw = input("Your skills: ")
    skills = [s.strip().lower() for s in raw.split(",") if s.strip()]

    while len(skills) < min_skills:
        print(f"Please enter at least {min_skills} skills.")
        raw = input("Your skills: ")
        skills = [s.strip().lower() for s in raw.split(",") if s.strip()]

    return skills


# ---------------------------------------------------------------------
# STEP 2: SCORING — Vector Mapping + Cosine Similarity
# ---------------------------------------------------------------------
def build_vector_space(df):
    """Fit a TF-IDF vectorizer on the job-role skill corpus. This
    establishes the shared vocabulary space that both jobs and the
    user's profile will be mapped into."""
    vectorizer = TfidfVectorizer()
    job_vectors = vectorizer.fit_transform(df["skills"])
    return vectorizer, job_vectors


def score_user_against_jobs(user_skills, vectorizer, job_vectors):
    """Transform the user's skills into the SAME vector space (crucial:
    we use .transform(), not .fit_transform(), so vocab stays aligned),
    then compute cosine similarity against every job role."""
    user_document = " ".join(user_skills)
    user_vector = vectorizer.transform([user_document])

    # Cold Start check: if the user vector is all zeros, none of their
    # skills exist in our vocabulary at all.
    if user_vector.nnz == 0:
        return None

    scores = cosine_similarity(user_vector, job_vectors).flatten()
    return scores


# ---------------------------------------------------------------------
# STEP 3 & 4: SORTING + FILTERING
# ---------------------------------------------------------------------
def rank_and_filter(df, scores, top_n=3):
    """Sort roles by descending similarity score, then truncate to the
    Top-N most relevant matches."""
    df = df.copy()
    df["match_score"] = scores
    ranked = df.sort_values("match_score", ascending=False)
    return ranked.head(top_n)


def trending_fallback(df, top_n=3):
    """Cold Start bypass: if we can't score the user at all (none of
    their skills are recognized), fall back to a 'trending' default
    list instead of returning nothing."""
    return df.head(top_n)


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def recommend(user_skills, csv_path="raw_skills.csv", top_n=3):
    df = load_job_roles(csv_path)
    vectorizer, job_vectors = build_vector_space(df)
    scores = score_user_against_jobs(user_skills, vectorizer, job_vectors)

    if scores is None:
        print("\n(No recognized skills matched our database — showing trending roles instead.)")
        return trending_fallback(df, top_n)

    return rank_and_filter(df, scores, top_n)


def display_recommendations(results):
    print("\n" + "=" * 55)
    print("TOP RECOMMENDED CAREER PATHS")
    print("=" * 55)
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        score_display = f"{row['match_score']:.0%}" if "match_score" in row else "N/A"
        print(f"{rank}. {row['role']}  (Match: {score_display})")
        print(f"   Core skills: {row['skills']}\n")


def main():
    print("=" * 55)
    print("DecodeLabs — Tech Stack Recommender")
    print("=" * 55)

    user_skills = ingest_user_skills(min_skills=3)
    results = recommend(user_skills)
    display_recommendations(results)


if __name__ == "__main__":
    main()

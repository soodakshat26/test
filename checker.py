import pandas as pd, json

df = pd.read_csv("vectors_delhi.csv", dtype={"vibe_tags": "string"},
                 keep_default_na=False)

# 1) how many rows?
print("Rows in CSV :", len(df))

# 2) how many vibe_tags are truly empty?
empty = df["vibe_tags"].str.strip().eq("").sum()
print("Empty vibe_tags:", empty)

# 3) show a few non-empty examples
samples = df[df["vibe_tags"].str.strip() != ""].head(5)["vibe_tags"].tolist()
print("\nSample tags:")
for t in samples:
    # parse JSON so you see it as a Python list
    print("  ", json.loads(t))

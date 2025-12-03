PROMPT START

You are enhancing summarize_clusters.py inside a DailyArxiv pipeline.
This script currently does:

Load cluster results from clusters_YYYY-MM-DD.json

For each cluster, gather paper titles/abstracts

Generate a natural-language summary using GPT (model=gpt-4o-mini)

Output summaries_YYYY-MM-DD.json

Now add subcluster analysis for each cluster.

✅ 子簇分析功能需求
1. For each cluster (with >= 12 papers), perform local subclustering

Use one of these algorithms (you choose based on simplicity):

Option A: Mini KMeans

Run KMeans(k=2) inside each cluster

If silhouette score < 0.05, do NOT split

Automatically attempt k=3 if cluster > 40 papers

Option B: HDBSCAN

Run a local HDBSCAN with:

min_cluster_size = max(3, len(cluster) // 10)

metric = "cosine"

If it produces ≥ 2 valid subclusters (not all noise), accept them

(You may choose either A or B, but must justify with comments.)

2. Summaries for each subcluster

For each valid subcluster:

Generate a name/title

Generate a 2–3 sentence summary

Extract 5 keywords

Include a few representative paper titles (2–3)

3. JSON output structure

Extend existing cluster summary JSON.
For each cluster:

{
  "cluster_id": 0,
  "title": "...",
  "summary": "...",
  "keywords": [...],
  "paper_count": 123,
  "subclusters": [
      {
        "subcluster_id": "0-A",
        "paper_count": 54,
        "keywords": [...],
        "title": "...",
        "summary": "...",
        "sample_papers": [...]
      },
      ...
  ]
}


If no subcluster is detected, "subclusters": [].

4. Embeddings reuse

You already have embeddings saved in the cluster JSON.
Do not recompute embeddings.
Use the existing embedding vectors for all local clustering.

5. Notes & constraints

Do not modify cluster_topics.py

Subclustering happens AFTER main clustering, not globally

Avoid splitting very small clusters (< 12 papers)

Make code modular and easy to disable (add --enable-subclusters flag)

Summaries continue to use gpt-4o-mini or the configured summarizer model

This should not affect the final daily report generator

Please update summarize_clusters.py according to all requirements above.

PROMPT END

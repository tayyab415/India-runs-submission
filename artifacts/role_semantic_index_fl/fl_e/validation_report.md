# URSI-FL fl_e role-semantic index — validation report

candidates=100000 roles=300171 unique_role_docs=44 unique_titles=48 unique_summary_templates=76 unique_skills=133
embedding_model=azure:text-embedding-3-large  role_neg_lambda=0.7

## Top 8 role documents by semantic evidence (face validity)
- 0.989 [matching_finetune_eval] (n=12) Built a RAG-based ranking pipeline serving 50M+ queries per month for an internal recruiter-facing search product. The a
- 0.966 [product_shipper] (n=6) Built systems that understand what users are looking for and connect them to the most relevant matches across a large da
- 0.943 [retrieval_ops] (n=8) Led the migration from keyword-based to embedding-based search across a 30M+ candidate corpus over 8 months. Designed th
- 0.920 [ranking_eval] (n=9) Owned the end-to-end ranking pipeline at a recommendations-heavy consumer product: candidate sourcing → embedding genera
- 0.898 [retrieval_ops] (n=8) Owned the design and rollout of a large-scale semantic search system serving an internal corpus of 35M+ items. Migrated 
- 0.875 [ranking_eval] (n=78) Owned the ranking layer for an e-commerce search product, evolving it from a hand-tuned scoring function to a learning-t
- 0.852 [product_shipper] (n=9) Built and shipped a production recommendation system at a marketplace product, going from offline experimentation to liv
- 0.830 [product_shipper] (n=5) Designed the ranking layer for the company's flagship product: how do we surface the right thing at the right time, acro

## Bottom 8 role documents (should be non-technical)
- 0.170 [matching_finetune_eval] (n=25237) Marketing leadership role at a B2B SaaS company. Owned the demand-generation function — content marketing, paid acquisit
- 0.148 [production_ml] (n=10125) Cloud infrastructure and DevOps work at an enterprise SaaS company. Owned the AWS account architecture (VPC, IAM, networ
- 0.125 [production_ml] (n=60) Implemented a RAG-based customer support chatbot integrated with our existing ticketing system. Built the document inges
- 0.102 [production_ml] (n=25029) Operations management role at a logistics company. Owned daily fulfillment operations across 3 warehouses, managing a te
- 0.080 [matching_finetune_eval] (n=25207) Business analyst at a consulting firm, working primarily with retail and CPG clients. Conducted business diagnostics, pr
- 0.057 [production_ml] (n=25290) Customer support team lead at a SaaS product. Managed a team of 8 support agents handling tier-1 and tier-2 tickets; own
- 0.034 [retrieval_ops] (n=25164) Brand design and creative direction at a consumer-products company. Owned brand identity (logo, visual system, typograph
- 0.011 [matching_finetune_eval] (n=25078) Senior accounting role at a mid-sized company — month-end close, financial reporting, statutory compliance (GAAP / Ind-A

## Top 6 summary templates by diagnostic semantic evidence
- 0.993 [product_shipper] (n=2) Senior AI engineer with <NUM> years of hands-on experience building production ML systems, with a focus on search, retrieval, and ranking. M
- 0.980 [ranking_eval] (n=1) Senior AI engineer with <NUM> years of hands-on experience building production ML systems, with a focus on search, retrieval, and ranking. M
- 0.967 [product_shipper] (n=1) Senior AI engineer with <NUM> years of hands-on experience building production ML systems, with a focus on search, retrieval, and ranking. M
- 0.954 [retrieval_ops] (n=1) Senior AI engineer with <NUM> years of hands-on experience building production ML systems, with a focus on search, retrieval, and ranking. M
- 0.941 [retrieval_ops] (n=1) Senior AI engineer with <NUM> years of hands-on experience building production ML systems, with a focus on search, retrieval, and ranking. M
- 0.928 [product_shipper] (n=1) Senior AI engineer with <NUM> years of hands-on experience building production ML systems, with a focus on search, retrieval, and ranking. M

## Top 12 skills by diagnostic semantic evidence
- 0.996 [candidate_matching] (n=5) Search Backend
- 0.989 [candidate_matching] (n=3) Search Infrastructure
- 0.981 [product_shipper] (n=5091) Recommendation Systems
- 0.974 [ranking_eval] (n=1383) Learning to Rank
- 0.966 [candidate_matching] (n=1311) Elasticsearch
- 0.959 [product_shipper] (n=4) Search & Discovery
- 0.951 [candidate_matching] (n=5087) Semantic Search
- 0.944 [product_shipper] (n=5135) Information Retrieval
- 0.936 [product_shipper] (n=1349) Machine Learning
- 0.929 [candidate_matching] (n=1342) Deep Learning
- 0.921 [retrieval_ops] (n=5080) Embeddings
- 0.914 [product_shipper] (n=1382) BM25

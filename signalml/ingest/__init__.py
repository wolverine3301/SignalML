"""Corpus adapters: third-party datasets -> manifest records (S2).

One module per external corpus. Adapters only *describe* what a corpus contains
(labels, provenance, licence) and hand it to the manifest; they never resample,
augment or otherwise touch audio content — that is S3/S4's job.
"""

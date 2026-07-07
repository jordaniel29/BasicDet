"""Dataset-curation tooling for the MTMDC multi-sensor source (persondet_v4.x).

Run the stages as modules from the repo root:

    python -m curation.extract_frames        # Stage 1: video -> shared frame pool
    python -m curation.build_version v4.1     # Stages 3+5: split, labels, COCO, docs
    python -m curation.build_version v4.2
    python -m curation.verify_version v4.1    # Stage 4: integrity checks

See ``curation.md`` (repo root) for the full playbook this implements.
"""

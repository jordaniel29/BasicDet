# Person-Detection Dataset Curation Playbook

A reusable, self-contained spec for turning a raw public dataset into a clean, unified
person-detection dataset. Hand this to any agent to process a **new source** (a future `vN`) the
same way the existing `combined_v1/v2/v3` were built. Follow the conventions exactly so all versions
stay interchangeable.

---

## 0. Context & goal

- Building single-class **`person`** detection datasets to train **YOLO26** and **RF-DETR**
  (the detector later feeds a **multi-camera tracking** system).
- Each batch of raw sources is a **version** (`v1`, `v2`, …). Layout:
  - `detection/raw/v<N>/<source>/` — original downloads (never modified).
  - `detection/processed/combined_v<N>/` — the unified, cleaned, split output.
- Versions are independent datasets; they can later be merged for training.

---

## 1. Output spec (EVERY version must follow this)

- **One class only: `person`** → YOLO class id `0`, COCO category `{"id":1,"name":"person"}`.
- **Two label formats from one source of truth:**
  - **YOLO**: `labels/<split>/<name>.txt`, one box per line `0 cx cy w h` (all **normalized 0–1**).
    An empty `.txt` = background/negative image (valid, keep it).
  - **COCO**: `annotations/instances_<split>.json`, `bbox=[x,y,w,h]` in **absolute pixels**,
    `iscrowd=0`, single category `person`(id 1).
- **Directory layout:**
  ```
  combined_v<N>/
  ├── data.yaml                       # Ultralytics YOLO config
  ├── README.md                       # provenance, processing, licensing, caveats
  ├── images/{train,val,test}/        # .jpg
  ├── labels/{train,val,test}/        # YOLO .txt (one per image)
  ├── annotations/instances_{train,val,test}.json   # COCO
  └── _dedup/                         # audit trail (reports, manifests, quarantine, feature cache)
  ```
  Optional `rfdetr/{train,valid,test}/` (each with `_annotations.coco.json` + images) for RF-DETR's
  native per-folder COCO layout — build by hardlinking images from `images/<split>/`.
- **`data.yaml`:**
  ```yaml
  path: <abs path to combined_v<N>>
  train: images/train
  val: images/val
  test: images/test
  nc: 1
  names: ['person']
  ```
- **Filename provenance prefix**: prefix every file with a short source tag + keep original stem,
  e.g. `hd_`, `inria_`, `pd_`, `p3_` (v1); `ch_`, `mot_` (v2); `wn_`, `cctv_` (v3). Guarantees global
  uniqueness and lets you trace/split by source later. For video frames encode source+scene+frame,
  e.g. `wn_set<N>_video<N>_<camera>_<frame6>.jpg`.
- **Disk**: when an image already exists elsewhere on the same filesystem, **hardlink** it
  (`os.link`) instead of copying — saves space for large sets. (Copies are fine for small sets.)

---

## 2. Pipeline (5 stages)

1. **Ingest & convert** every source to the unified YOLO+COCO format above.
2. **Deduplicate** (exact + near-duplicate), report first, then remove reversibly.
3. **Split** train/val/test (strategy depends on the data — see §5).
4. **Verify** integrity (parity, COCO match, zero cross-split leakage, class ids).
5. **Document** a README + update `data.yaml`.

Stage all converted images into the `train` split first if you intend to dedup-then-resplit the whole
pool; otherwise convert directly into per-split folders.

---

## 3. Stage 1 — Convert to unified format

General rules: remap **all** class ids to `0`/person; clamp boxes to image bounds; drop boxes with
width or height ≤ 1px; read image W,H from the file (don't trust annotation metadata blindly).

**Box math (write both YOLO and COCO from pixel corners x1,y1,x2,y2):**
```
bw = x2-x1 ; bh = y2-y1
YOLO: 0  (x1+x2)/2/W  (y1+y2)/2/H  bw/W  bh/H
COCO: bbox=[x1, y1, bw, bh]  area=bw*bh
```

**Per source-type recipe:**
- **Already YOLO** (Roboflow exports etc.): copy images (prefixed), copy `.txt` but rewrite the class
  field to `0`. Class names like `'0'`/`'object'` are still person — remap.
- **Pascal-VOC CSV** (`filename,width,height,class,xmin,ymin,xmax,ymax`): group rows by filename;
  convert pixel corners → YOLO/COCO.
- **CrowdHuman `.odgt`** (JSON-lines): for each `gtboxes` entry with `tag=="person"`, use the
  **full-body box `fbox`=[x,y,w,h]** (top-left). Skip `tag=="mask"` and any `extra.ignore==1`.
  (`fbox` is amodal/full-body; `vbox` is visible-only — pick one convention and document it.)
- **MOT-challenge `gt/gt.txt`** (`frame,id,bb_left,bb_top,w,h,flag,class,vis`): keep rows with
  `flag==1` and `class in {1 pedestrian, 7 static person}`; drop others (esp. 11 occluder). Boxes are
  top-left pixel. Only **train** sequences have GT; test sequences have none (exclude or treat as
  unlabeled). Read W,H from each `seqinfo.ini` (per-sequence resolution can differ).
- **Video (e.g. WiseNET `.avi` + per-frame JSON)**: see §5 "video" — extract frames, don't ingest all.

---

## 4. Stage 2 — Deduplication

**Why:** public sources overlap; near-duplicates across train/test inflate metrics (the model
memorizes them). Goal: keep one representative per duplicate cluster and eliminate cross-split twins.

**Three detectors, unioned with union-find into clusters:**
1. **Exact**: SHA-256 of file bytes.
2. **Perceptual**: 64-bit **dHash + pHash** (`imagehash`, hash_size=8). Cluster if **Hamming ≤ 6**
   (use LSH banding — split the 64-bit hash into 4×16-bit bands — to find candidate pairs fast).
3. **Semantic**: **CLIP ViT-B/32** (`open_clip`, pretrained `openai`) image embeddings,
   L2-normalized; cluster if **cosine ≥ 0.95**. (0.92 over-merges generic "people" scenes — 0.95 is
   the calibrated default.)

**Keep policy per cluster**: keep the best representative = **max #boxes → max resolution → prefer
`train` split**; everything else is a removal candidate.

**Report-first, reversible removal:**
- Cache features once (`features.npz`: dhash/phash/emb + `meta.json`) so you can re-threshold without
  recomputing embeddings.
- Emit a **report** (clusters, which copy kept, cross-split flag), a `remove_candidates.csv`, and an
  HTML **contact sheet** of clusters → review before deleting.
- "Remove" = **move** files (image + label) to `_dedup/removed/` (quarantine) with a manifest, then
  **rebuild the COCO** jsons from survivors. Never hard-delete; it stays reversible.

**⚠️ CRITICAL LESSON — do NOT run perceptual/CLIP dedup on fixed-camera video.** A static surveillance
camera makes every frame of a sequence >0.95 cosine similar, so CLIP collapses a whole sequence to
~1–2 images (it tried to delete 893/895 MOT20 frames, and ~all WiseNET frames). For video, the
**temporal subsampling in §5 is the dedup** — protect video frames from the embedding stage (e.g. a
`PROTECT_PREFIX` so `mot_`/`wn_` files are never removed by it). Cross-source dedup between distinct
domains usually finds ~0 and is cheap to verify.

---

## 5. Stage 3 — Split train/val/test

Default **80/10/10**, deterministic (`seed=42`).

**Choose the strategy by data type:**
- **Independent images (v1, v2)** → **stratified random by source**: shuffle each source's files with
  a fixed seed, slice 80/10/10, so every split has proportional representation of every source.
  (Dedup must run *first* so random assignment can't scatter duplicates across splits.)
- **Video / fixed-camera (v3)** → **scene/camera-disjoint holdout**: assign whole **cameras** (or at
  least whole recording sessions) to one split each, so test scenes/backgrounds are unseen in
  training. This is the honest generalization test for surveillance detectors.

**Key insight that drives the video strategy:**
> A person **detector** has no identity head — it does **not** learn *who* people are. So the same
> person appearing in train and test is a **negligible** source of metric inflation (that matters for
> the downstream **re-ID/tracking** model, not the detector). What actually inflates a detector's
> score is **shared camera/background and near-duplicate frames**. So separate **scenes/cameras**, not
> identities. (Caveat: if the detector will be deployed on the *same* known cameras, scene overlap is
> realistic and acceptable — then a session-level split is fine and you keep all cameras in training.)

**Video frame handling (do this in Stage 1 for video sources):**
- **Temporal subsampling**: keep every Nth frame (e.g. every 15th ≈ 2 fps for 30 fps footage). This,
  not embedding-dedup, is how you de-duplicate video.
- **Negatives**: many video sources only annotate frames containing people. Keep a **sparse subset of
  empty frames as background negatives** (~10–15% of kept positives) to suppress false positives on
  static scenes — but verify the unannotated frames are truly people-free (contiguous annotated
  frame-number segments are a good signal). Don't keep all of them (swamps the set); don't keep none
  (loses FP suppression).

After deciding assignments, **move files into split folders and rebuild COCO** with fresh contiguous
ids per split. Write a split manifest (e.g. `_resplit_*_manifest.csv`) for reversibility.

---

## 6. Stage 4 — Verify (must all pass)

- **Parity**: `#images == #labels == #COCO images` in every split.
- **No orphans**: every image has a label stem and vice-versa.
- **Zero leakage**: no image (by name) and — for non-video — no exact-hash duplicate appears in more
  than one split. For camera-disjoint video, assert no camera id appears in more than one split.
- **Class sanity**: only class id `0` present in YOLO labels.
- **Box sanity**: all normalized coords in [0,1]; no NF≠5 lines.
- **Round-trip check**: for a converted source, confirm one box matches the original annotation.

---

## 7. Stage 5 — Document

Write `combined_v<N>/README.md` with: at-a-glance counts (images/boxes/split), class mapping,
directory layout, label-format spec, usage commands, **per-source provenance table (URL + license)**,
per-source contribution counts, the exact processing pipeline + parameters used, and a **caveats**
section (licensing restrictions, split limitations, box-convention heterogeneity, domain imbalance).

---

## 8. Environment / tooling notes

- macOS, Apple Silicon (MPS available — CLIP embedding runs on `mps`). System Python is
  externally-managed (PEP 668) → use a **venv**.
- Dedup deps: `pillow numpy imagehash torch torchvision open_clip_torch`.
- Video deps: `opencv-python-headless` + system `ffmpeg/ffprobe`.
- **Put the venv in a stable location** (e.g. `~/.cache/<name>_venv`), NOT under `/private/tmp` — the
  macOS temp reaper can corrupt a tmp venv mid-session.
- No external image library? JPEG/PNG dimensions can be read from file headers without Pillow.
- Frame extraction: read each video's annotation `frameNumber` convention from the dataset's own
  visualizer if provided (WiseNET's `frameNumber` is the 0-based `cv2.CAP_PROP_POS_FRAMES` index);
  decode sequentially and match by index for reliability on AVI.

---

## 9. Worked examples (already built — match these conventions)

- **combined_v1** — general person detection. 4 sources (HumanDataset, INRIA, PeopleDetection,
  person-3). Dedup cos≥0.95 removed ~7.1k (16.7%); stratified-random 80/10/10. ~35.5k imgs.
- **combined_v2** — crowd density. CrowdHuman (full-body `fbox`) + MOT20 (classes 1+7, every 10th
  frame). MOT20 frames **protected** from CLIP dedup; removed 312 CrowdHuman near-dups; stratified
  80/10/10. ~20k imgs, ~27 boxes/img. **CC BY-NC → non-commercial.**
- **combined_v3** — indoor CCTV (for tracking). WiseNET (62 videos → every 15th annotated frame +
  sparse negatives, xywh→YOLO) + CCTV Roboflow. **set/session-level** 80/10/10 split. Note: WiseNET
  has only 5 cameras reused across all sets, so test backgrounds also appear in train (acceptable for
  same-camera deployment); a camera-disjoint split is reproducible from filenames if an unseen-camera
  test is needed.

---

## 10. Gotchas

- Run **dedup before splitting** (so random splits can't leak duplicates).
- **Never** embedding-dedup fixed-camera video → temporal subsample instead, and protect those frames.
- Keep raw sources untouched; all removals reversible via quarantine + manifests.
- Watch box-convention drift across sources (full-body vs visible-region) — document it.
- Check licenses per source; flag non-commercial (e.g. CrowdHuman/MOT) and attribution-only (WiseNET).
- Hardlinks share bytes — deleting one of two hardlinked trees frees little; deleting source archives
  (zips) after extraction is the real reclaim.

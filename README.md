# inception-replication

An InceptionV3 classifier for FaceForensics++ face crops (original vs. each manipulation method), plus saliency maps that show which pixels drove each prediction.

## Setup

```sh
uv sync
```

The saliency group (captum) is a default group, so a plain `uv sync` / `uv run` installs it.

## Commands

```sh
# Classification report over a split (one subfolder per class)
uv run inception-report resources/faceforensis_face_extract/test
uv run inception-report <split_dir> -o preds.csv --limit 200 --batch-size 32 --num-workers 8

# Saliency maps for one image, or a random sample from a split
uv run saliency-visualize <image.png> -o saliency_grid.png
uv run saliency-visualize <split_dir> -n 4 --seed 0 -o saliency_grid.png

# Same row as above plus the GradCAM spotlight (one image only)
uv run saliency-spotlight <image.png> -o saliency_spotlight.png

# VLM inputs for one image: GradCAM spotlight + heatmap at full resolution, and class probabilities
#   writes <stem>_spotlight.png, <stem>_gradcam.png, <stem>.json
uv run saliency-vlm-input <image.png> -o vlm_input/

# Tests (no resources needed)
uv run pytest
```

Each command takes `--help`.

## Expected `resources/` layout

```
resources/
├── best_inceptionv3_model_multiclass.h5     # model weights (default for --weights)
└── faceforensis_face_extract/
    ├── train/
    ├── val/
    └── test/
        ├── DeepFakeDetection/   DeepFakeDetection_01_02__meeting_serious__YVGY8LOK_frame1.png
        ├── FaceShifter/         FaceShifter_003_000_frame1.png
        ├── FaceSwap/            FaceSwap_009_027_frame1.png
        ├── NeuraTextures/       NeuralTextures_001_870_frame1.png
        ├── deepfakes/           Deepfakes_009_027_frame1.png
        ├── face2face/           Face2Face_003_000_frame1.png
        ├── original/            original_008_frame1.png
        └── ...                  any other folder is ignored
```

- A split folder (`train/`, `val/`, `test/`) is what `inception-report` and `saliency-visualize` take.
- Only the 7 class folders above are read, with these exact names (case-sensitive, including the
  `NeuraTextures` spelling); they come from `CLASS_NAMES` in `src/inception_replication/configs.py`.
  All 7 must exist and be non-empty in the split you pass, or loading fails.
- The dataset also ships extra folders, such as `masks_full/`, `colored_masks_full/` and, in `test/`,
  `masks/`, `colored_masks/`, `manipulated_{all,eyes,hair,mouth,nose,skin}/` and
  `original_to_compare_with_masks/`. They are not classes and the classifier skips them.

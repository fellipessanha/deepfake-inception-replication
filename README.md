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

# VLM explanation of one cropped face: prediction + GradCAM + Qwen3-VL text (needs a CUDA GPU)
#   writes <stem>_spotlight.png, <stem>_gradcam.png, <stem>.json (with "explanation")
uv run vlm-explain <face.png> -o vlm_output/ [--prompt Activation|Specifics|Claude]
uv run vlm-explain <face.png> --system-prompt "$(cat my_prompt.txt)"

# Tests (no resources needed)
uv run pytest
```

Each command takes `--help`.

## Docker (`vlm-explain`)

Needs the NVIDIA Container Toolkit on the host. The Qwen weights (~9 GB) are not in the image:
the mounted HF cache downloads them on the first run and reuses them after that.

```sh
docker build -t inception-vlm .
# Pascal GPUs (Tesla P100...) or drivers older than CUDA 12: torch 2.7 built for CUDA 11.8
docker build --build-arg TORCH_EXTRA=pascal -t inception-vlm .
docker run --rm --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$PWD":/work \
  inception-vlm faces/some_face.png [--prompt Claude | --system-prompt "..."]
```

Outputs land in `./vlm_output` and are owned by root; add `--user $(id -u):$(id -g) -e HF_HOME=/work/.hf`
to get them as your user.

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

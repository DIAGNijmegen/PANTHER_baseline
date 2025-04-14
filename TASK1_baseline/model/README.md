## GC Submission

You can upload your model as a standalone tarball (`.tar.gz`) via Grand Challenge under the **Your algorithm > Models** section. Alternatively, include it during the container-image build by placing it in a `resources/` directory. Uploading as a tarball offers the advantage of easier updates compared to rebuilding the entire container image. If provided, the tarball will be automatically extracted to `/opt/ml/model/` at runtime.

## Local Inference with the Baseline Model

For local inference using the baseline model, place the following model weights in this folder:
- `checkpoint_best_PantherTask1_fold_0.pth`
- `checkpoint_best_PantherTask1_fold_1.pth`
- `checkpoint_best_PantherTask1_fold_2.pth`

Additionally, download and unzip the MRSegmentator weights from [this release](https://github.com/hhaentze/MRSegmentator/releases/tag/v1.2.0) and include them in the same folder (only fold_0 is needed for inference).

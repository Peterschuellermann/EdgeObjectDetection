# src/vendor_sahi/get_sliced_prediction_fast.py

from time import perf_counter
import numpy as np
from sahi.prediction import PredictionResult, ObjectPrediction
from sahi.slicing import slice_image
from .predict_local import get_prediction, get_batched_prediction, POSTPROCESS_NAME_TO_CLASS

def get_sliced_prediction_fast(
    image,
    detection_model,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "NMS",          # faster than GREEDYNMM
    postprocess_match_metric: str = "IOU",  # typical for NMS
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    verbose: int = 0,
    exclude_classes_by_name: list[str] | None = None,
    exclude_classes_by_id: list[int] | None = None,
) -> PredictionResult:
    """
    Fast, simplified version of SAHI sliced prediction:

    - assumes a single image (no batches, no videos)
    - no extra full-image prediction
    - one final postprocess pass (NMS/NMM) at the end
    - minimal bookkeeping, focused on speed

    Returns a PredictionResult with durations_in_seconds:
        - 'slice': time spent slicing
        - 'prediction': total model forward time
        - 'postprocess': time spent merging predictions
        - 'forward': same as 'prediction' (for clarity)
    """
    durations_in_seconds: dict[str, float] = {
        "slice": 0.0,
        "prediction": 0.0,
        "postprocess": 0.0,
        "forward": 0.0,
    }

    # 1) Slice image into tiles
    t0 = perf_counter()
    slice_image_result = slice_image(
        image=image,
        output_file_name=None,
        output_dir=None,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=False,   # we explicitly pass slice size -> skip heuristics
    )
    durations_in_seconds["slice"] = perf_counter() - t0

    num_slices = len(slice_image_result)
    if verbose:
        print(f"[FAST] Performing prediction on {num_slices} slices.")

    # 2) Set up postprocess (NMS/NMM etc.)
    if postprocess_type not in POSTPROCESS_NAME_TO_CLASS.keys():
        raise ValueError(
            f"postprocess_type should be one of {list(POSTPROCESS_NAME_TO_CLASS.keys())} "
            f"but given as {postprocess_type}"
        )
    postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
    postprocess = postprocess_constructor(
        match_threshold=postprocess_match_threshold,
        match_metric=postprocess_match_metric,
        class_agnostic=postprocess_class_agnostic,
    )

    object_prediction_list: list[ObjectPrediction] = []

    # 3) Run model on slices (batched)
    BATCH_SIZE = 16  # Tunable
    
    full_h = slice_image_result.original_image_height
    full_w = slice_image_result.original_image_width

    for start_idx in range(0, num_slices, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_slices)
        
        batch_images = []
        shift_amounts = []
        
        for i in range(start_idx, end_idx):
            batch_images.append(slice_image_result.images[i])
            shift_amounts.append(slice_image_result.starting_pixels[i])
            
        # Run batch
        batch_result = get_batched_prediction(
            images=batch_images,
            detection_model=detection_model,
            shift_amounts=shift_amounts,
            full_shape=[full_h, full_w],
            exclude_classes_by_name=exclude_classes_by_name,
            exclude_classes_by_id=exclude_classes_by_id,
        )

        # accumulate timings
        d = batch_result.durations_in_seconds or {}
        durations_in_seconds["prediction"] += d.get("prediction", 0.0)
        durations_in_seconds["forward"] += d.get("prediction", 0.0)
        durations_in_seconds["postprocess"] += d.get("postprocess", 0.0)

        # predictions are still in slice coords (conceptually) or shifted?
        # Since we follow get_prediction logic which calls convert(shift), 
        # and existing code called get_shifted_object_prediction() on those, we do the same.
        for obj_pred in batch_result.object_prediction_list:
            if obj_pred:
                object_prediction_list.append(obj_pred.get_shifted_object_prediction())

    # 4) Final merge of overlapping predictions across slices
    t_post = perf_counter()
    if len(object_prediction_list) > 1:
        object_prediction_list = postprocess(object_prediction_list)
    durations_in_seconds["postprocess"] += perf_counter() - t_post

    if verbose:
        print(
            f"[FAST] slice={durations_in_seconds['slice']:.3f}s, "
            f"pred={durations_in_seconds['prediction']:.3f}s, "
            f"post={durations_in_seconds['postprocess']:.3f}s"
        )

    return PredictionResult(
        image=image,
        object_prediction_list=object_prediction_list,
        durations_in_seconds=durations_in_seconds,
    )

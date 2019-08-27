
INPUT_ANNOTATIONS_VRD=/mnt/chicm/data/open-images/relation/challenge-2019-validation-vrd.csv
INPUT_ANNOTATIONS_LABELS=/mnt/chicm/data/open-images/relation/challenge-2019-validation-vrd-labels.csv
INPUT_PREDICTIONS=$1
INPUT_CLASS_LABELMAP=./oid_object_detection_challenge_500_label_map.pbtxt
INPUT_RELATIONSHIP_LABELMAP=./relationships_labelmap.pbtxt

OUTPUT_METRICS=./val_metrics.txt

python /mnt/chicm/models/research/object_detection/metrics/oid_vrd_challenge_evaluation.py \
    --input_annotations_boxes=${INPUT_ANNOTATIONS_VRD} \
    --input_annotations_labels=${INPUT_ANNOTATIONS_LABELS} \
    --input_predictions=${INPUT_PREDICTIONS} \
    --input_class_labelmap=${INPUT_CLASS_LABELMAP} \
    --input_relationship_labelmap=${INPUT_RELATIONSHIP_LABELMAP} \
    --output_metrics=${OUTPUT_METRICS}

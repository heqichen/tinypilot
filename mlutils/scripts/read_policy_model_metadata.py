import pickle

with open(
    "/home/heqichen/workspace/tinypilot/POC/resources/models/driving_policy_metadata.pkl",
    "rb",
) as f:
    vision_metadata = pickle.load(f)
    vision_input_shapes = vision_metadata["input_shapes"]
    vision_output_slices = vision_metadata["output_slices"]
    vision_output_size = vision_metadata["output_shapes"]["outputs"][1]
    print(vision_metadata)
    print(vision_input_shapes)
    print(vision_output_slices)
    print(vision_output_size)

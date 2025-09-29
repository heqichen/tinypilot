import onnxruntime as ort
import os
import numpy as np


def run_onnx_model(small_frames):
    """
    Run an ONNX model with the given input data.

    :param model_path: Path to the ONNX model file
    :param input_data: Dictionary containing input data for the model
    :return: Model output
    """
    # Load the ONNX model
    session = ort.InferenceSession(
        os.path.join(
            "/home/heqichen/workspace/tinypilot/POC/resources/models/driving_vision.onnx"
        )
    )

    model_frames = np.reshape(small_frames, (1, 12, 128, 256))
    model_input = {"input_imgs": model_frames, "big_input_imgs": model_frames}
    # Run the model
    result = session.run(None, model_input)
    return result[0]


def work():
    # Read input rom file
    input_data = np.genfromtxt(
        "/home/heqichen/workspace/tinypilot/system/test/module/output/imgs0.csv"
    ).astype(np.uint8)

    output_result = run_onnx_model(input_data).flatten()
    np.savetxt("py0.csv", output_result, delimiter=",")


if __name__ == "__main__":
    work()

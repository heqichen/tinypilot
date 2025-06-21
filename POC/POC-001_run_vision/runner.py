import os
import pickle
from convert_images import prepare
import numpy as np
import onnxruntime as ort
from parse_model_outputs import Parser

def run_onnx_model(small_frames):
    """
    Run an ONNX model with the given input data.
    
    :param model_path: Path to the ONNX model file
    :param input_data: Dictionary containing input data for the model
    :return: Model output
    """
    # Load the ONNX model
    session = ort.InferenceSession(os.path.join(os.path.dirname(__file__),"res/driving_vision.onnx"))
    
    model_frames = np.reshape(small_frames, (1, 12, 128, 256))
    model_input = {"input_imgs": model_frames, "big_input_imgs": model_frames}
    # Run the model
    result = session.run(None, model_input)
    return result[0]

def slice_outputs(model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis,:, v] for k,v in output_slices.items()}
    # parsed_model_outputs = {}
    # for k,v in output_slices.items():
    #     print(k, v)

    #     parsed_model_outputs[k] = np.array(model_outputs[v.start: v.stop])
    #     print(k , parsed_model_outputs[k], v.start, v.stop, parsed_model_outputs[k].shape)
    return parsed_model_outputs

def read_slices():
    VISION_METADATA_PATH = os.path.join(os.path.dirname(__file__), "res/driving_vision_metadata.pkl")
    with open(VISION_METADATA_PATH, 'rb') as f:
        vision_metadata = pickle.load(f)
        vision_input_shapes =  vision_metadata['input_shapes']
        vision_output_slices = vision_metadata['output_slices']
        vision_output_size = vision_metadata['output_shapes']['outputs'][1]
    return vision_output_slices

    
if __name__ == "__main__":
    buffer_frame = prepare(os.path.join(os.path.dirname(__file__),"res/1.jpg"))
    last_frame = prepare(os.path.join(os.path.dirname(__file__),"res/2.jpg"))
    vision_output_slices = read_slices()
    
    input_frames = np.concatenate((buffer_frame, last_frame))
    
    model_output = run_onnx_model(input_frames)
    # print(model_output)
    
    parser = Parser()
    
    # vision_output = np.reshape(model_output, (1, 632))
    
    # vision_outputs_dict = parser.parse_vision_outputs({"outputs": vision_output})
    sliced_output = slice_outputs(model_output, vision_output_slices)
    vision_outputs_dict = parser.parse_vision_outputs(sliced_output)
    print(vision_outputs_dict)
    print(vision_outputs_dict.keys())
    print(vision_outputs_dict["pose"])
    
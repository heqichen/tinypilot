from model_runner.constants import ModelConstants
from model_runner.parse_model_outputs import Parser
from model_runner.prepare import prepare
from model_runner.vars import basedir
import numpy as np
import onnxruntime
import pickle

DRIVING_VISION_METADATA = basedir + "/driving_vision_metadata.pkl"
DRIVING_POLICY_METADATA = basedir + "/driving_policy_metadata.pkl"
DRIVING_VISION_ONNX = basedir + "/driving_vision.onnx"
DRIVING_POLICY_ONNX = basedir + "/driving_policy.onnx"



class ModelState:
    prev_desire: np.ndarray
    vision_input_images: []

    def __init__(self):
        self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float16)
        self.vision_session = onnxruntime.InferenceSession(DRIVING_VISION_ONNX)
        self.policy_session = onnxruntime.InferenceSession(DRIVING_POLICY_ONNX)
        self.vision_input_images = []
        self.parser = Parser()
        
        # policy inputs
        self.numpy_inputs = {
            'desire': np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN, ModelConstants.DESIRE_LEN), dtype=np.float16),
            'traffic_convention': np.zeros((1, ModelConstants.TRAFFIC_CONVENTION_LEN), dtype=np.float16),
            'lateral_control_params': np.zeros((1, ModelConstants.LATERAL_CONTROL_PARAMS_LEN), dtype=np.float16),
            'prev_desired_curv': np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN, ModelConstants.PREV_DESIRED_CURV_LEN), dtype=np.float16),
            'features_buffer': np.zeros((1, ModelConstants.FULL_HISTORY_BUFFER_LEN,  ModelConstants.FEATURE_LEN), dtype=np.float16),
        }

        with open(DRIVING_VISION_METADATA, 'rb') as f:
            vision_metadata = pickle.load(f)
            # self.vision_input_shapes =  vision_metadata['input_shapes']
            self.vision_output_slices = vision_metadata['output_slices']
            # vision_output_size = vision_metadata['output_shapes']['outputs'][1]
        with open(DRIVING_POLICY_METADATA, 'rb') as f:
            policy_metadata = pickle.load(f)
            # self.policy_input_shapes =  policy_metadata['input_shapes']
            self.policy_output_slices = policy_metadata['output_slices']
            # policy_output_size = policy_metadata['output_shapes']['outputs'][1]


    def run_vision(self, vision_inputs):
        result = self.vision_session.run(None, vision_inputs)
        return np.array(result).flatten()
    
    def run_policy(self, policy_inputs):
        result = self.policy_session.run(None, policy_inputs)
        return np.array(result).flatten()
    
    def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
        parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
        return parsed_model_outputs

    def run(self, buf, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
        # Vision
        input_image = prepare(buf)
        if (len(self.vision_input_images) == 0):
            self.vision_input_images.append(input_image)
            self.vision_input_images.append(input_image)
        else:
            del self.vision_input_images[0]
            self.vision_input_images.append(input_image)
        
        parsed_arr = np.array(self.vision_input_images)
        parsed_arr.resize((1,12,128,256))


        imgs = {'input_imgs': parsed_arr,
                'big_input_imgs': parsed_arr}

        # if TICI:
        # # The imgs tensors are backed by opencl memory, only need init once
        # for key in imgs_cl:
        #     if key not in self.vision_inputs:
        #     self.vision_inputs[key] = qcom_tensor_from_opencl_address(imgs_cl[key].mem_address, self.vision_input_shapes[key], dtype=dtypes.uint8)
        # else:
        # for key in imgs_cl:
        #     frame_input = self.frames[key].buffer_from_cl(imgs_cl[key]).reshape(self.vision_input_shapes[key])
        #     self.vision_inputs[key] = Tensor(frame_input, dtype=dtypes.uint8).realize()



        vision_output = self.run_vision(imgs)
        vision_outputs_dict = self.parser.parse_vision_outputs(self.slice_outputs(vision_output, self.vision_output_slices))

        # Policy
        # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
        inputs['desire'][0] = 0
        new_desire = np.where(inputs['desire'] - self.prev_desire > .99, inputs['desire'], 0)
        self.prev_desire[:] = inputs['desire']

        self.numpy_inputs['desire'][0,:-1] = self.numpy_inputs['desire'][0,1:]
        self.numpy_inputs['desire'][0,-1] = new_desire

        self.numpy_inputs['traffic_convention'][:] = inputs['traffic_convention']
        self.numpy_inputs['lateral_control_params'][:] = inputs['lateral_control_params']


        # policy_inputs
        self.numpy_inputs['features_buffer'][0,:-1] = self.numpy_inputs['features_buffer'][0,1:]
        self.numpy_inputs['features_buffer'][0,-1] = vision_outputs_dict['hidden_state'][0, :]
        policy_output = self.run_policy(self.numpy_inputs)

        policy_outputs_dict = self.parser.parse_policy_outputs(self.slice_outputs(policy_output, self.policy_output_slices))
        

        # # TODO model only uses last value now
        self.numpy_inputs['prev_desired_curv'][0,:-1] = self.numpy_inputs['prev_desired_curv'][0,1:]
        self.numpy_inputs['prev_desired_curv'][0,-1,:] = policy_outputs_dict['desired_curvature'][0, :]

        combined_outputs_dict = {**vision_outputs_dict, **policy_outputs_dict}
        # if SEND_RAW_PRED:
        # combined_outputs_dict['raw_pred'] = np.concatenate([self.vision_output.copy(), self.policy_output.copy()])

        return combined_outputs_dict

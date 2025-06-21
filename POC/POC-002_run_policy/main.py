from model_runner.prepare import crop_and_resize
from model_runner.video_input import VideoInput
from model_runner.model_state import ModelState
from model_runner.constants import ModelConstants
from model_runner.fill_model_msg import get_leads
from model_runner.vars import basedir
import numpy as np



def main():
    sample = basedir + 'sample.mp4'
    tunnel = basedir + 'tunnel_right_curve.mp4'
    # Init
    vi = VideoInput(tunnel)
    model = ModelState()
    frameId = 0

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float16)
    # TODO: implement desire later
    vec_desire[0] = 1 # None
    
    traffic_convention = np.zeros(2)
    traffic_convention[0] = 1 # driver left hand  0=1, right hand 1=1

    steer_delay = 0.1 + 0.3 # mazda = 0.1 + 0.2 extra delay
    lateral_control_params = np.array([8.7, steer_delay], dtype=np.float16)

    inputs:dict[str, np.ndarray] = {
        'desire': vec_desire,
        'traffic_convention': traffic_convention,
        'lateral_control_params': lateral_control_params,
    }

    while True:
        frame_rgb = vi.capture()
        resized_frame_rgb = crop_and_resize(frame_rgb, 512, 256)

        # print(frameId)
        frameId = frameId + 1

        model_output = model.run(resized_frame_rgb, inputs)
        # print(model_output)

        leads = get_leads(model_output)
        print(leads[0].x[0])


      

if __name__ == "__main__":
    main()
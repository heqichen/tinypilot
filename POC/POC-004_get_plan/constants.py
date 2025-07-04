
import numpy as np

def index_function(idx, max_val=192, max_idx=32):
  return (max_val) * ((idx/max_idx)**2)

class ModelConstants:
  # time and distance indices
  IDX_N = 33
  T_IDXS = [index_function(idx, max_val=10.0) for idx in range(IDX_N)]
  X_IDXS = [index_function(idx, max_val=192.0) for idx in range(IDX_N)]
  LEAD_T_IDXS = [0., 2., 4., 6., 8., 10.]
  LEAD_T_OFFSETS = [0., 2., 4.]
  META_T_IDXS = [2., 4., 6., 8., 10.]

  # model inputs constants
  MODEL_FREQ = 20
  FEATURE_LEN = 512
  FULL_HISTORY_BUFFER_LEN = 25
  DESIRE_LEN = 8
  TRAFFIC_CONVENTION_LEN = 2
  LAT_PLANNER_STATE_LEN = 4
  LATERAL_CONTROL_PARAMS_LEN = 2
  PREV_DESIRED_CURV_LEN = 1

  # model outputs constants
  FCW_THRESHOLDS_5MS2 = np.array([.05, .05, .15, .15, .15], dtype=np.float32)
  FCW_THRESHOLDS_3MS2 = np.array([.7, .7], dtype=np.float32)
  FCW_5MS2_PROBS_WIDTH = 5
  FCW_3MS2_PROBS_WIDTH = 2

  DISENGAGE_WIDTH = 5
  POSE_WIDTH = 6
  WIDE_FROM_DEVICE_WIDTH = 3
  LEAD_WIDTH = 4
  LANE_LINES_WIDTH = 2
  ROAD_EDGES_WIDTH = 2
  PLAN_WIDTH = 15
  DESIRE_PRED_WIDTH = 8
  LAT_PLANNER_SOLUTION_WIDTH = 4
  DESIRED_CURV_WIDTH = 1

  NUM_LANE_LINES = 4
  NUM_ROAD_EDGES = 2

  LEAD_TRAJ_LEN = 6
  DESIRE_PRED_LEN = 4

  PLAN_MHP_N = 5
  LEAD_MHP_N = 2
  PLAN_MHP_SELECTION = 1
  LEAD_MHP_SELECTION = 3

  FCW_THRESHOLD_5MS2_HIGH = 0.15
  FCW_THRESHOLD_5MS2_LOW = 0.05
  FCW_THRESHOLD_3MS2 = 0.7

  CONFIDENCE_BUFFER_LEN = 5
  RYG_GREEN = 0.01165
  RYG_YELLOW = 0.06157

  POLY_PATH_DEGREE = 4
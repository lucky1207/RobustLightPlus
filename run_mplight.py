from utils.utils import pipeline_wrapper, merge
from utils import config
import time
from multiprocessing import Process
import argparse
import os
from utils.config import DMBP_config
import torch
import numpy as np 
import random
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-memo",       type=str,           default='MPLight')
    parser.add_argument("-mod",        type=str,           default="EfficientMPLight")
    parser.add_argument("-eightphase",  action="store_true", default=False)
    parser.add_argument("-gen",        type=int,            default=1)
    parser.add_argument("-multi_process", action="store_true", default=True)
    parser.add_argument("-workers",    type=int,            default=3)
    parser.add_argument("-hangzhou",    action="store_true", default=False)
    parser.add_argument("-jinan",       action="store_true", default=False)
    parser.add_argument("-new_york",       action="store_true", default=False)

    parser.add_argument("-inference", action="store_true", default=False)
    parser.add_argument("-traffic_file", type=str, default="anon_3_4_jinan_real")
    parser.add_argument("-NOISE_SCALE", type=float, choices=[3.5, 4.0])
    parser.add_argument("-NOISE_TYPE", type=int, choices=[-1, 0, 1, 2, 3])
    parser.add_argument("-NOISE_LEVEL", type=int, choices=[-1, 0, 1])
    parser.add_argument("-NOISE_DIRECTION", type=str, choices=["W", "W,E"])
    return parser.parse_args()


def main(in_args=None):
    if in_args.hangzhou:
        count = 3600
        road_net = "4_4"
        # traffic_file_list = ["anon_4_4_hangzhou_real.json",
        #                     "anon_4_4_hangzhou_real_5816.json"
        #                      ]
        traffic_file_list = [f"{in_args.traffic_file}.json"]
        num_rounds = 3
        template = "Hangzhou"
    elif in_args.jinan:
        count = 3600
        road_net = "3_4"
        # traffic_file_list = ["anon_3_4_jinan_real.json",
        #                   "anon_3_4_jinan_real_2000.json",
        #                   "anon_3_4_jinan_real_2500.json"
        #                      ]
        traffic_file_list = [f"{in_args.traffic_file}.json"]
        num_rounds = 3
        template = "Jinan"
    elif in_args.new_york:
        count = 3600
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json",
                            #"anon_28_7_newyork_real_triple.json"
                              ]
        num_rounds = 80
        template = "newyork_28_7"

    NUM_COL = int(road_net.split('_')[1])
    NUM_ROW = int(road_net.split('_')[0])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(traffic_file_list)
    process_list = []
    for traffic_file in traffic_file_list:
        dic_traffic_env_conf_extra = {

            "NUM_ROUNDS": num_rounds,
            "NUM_GENERATORS": in_args.gen,
            "NUM_AGENTS": 1,
            "NUM_INTERSECTIONS": num_intersections,
            "RUN_COUNTS": count,

            "MODEL_NAME": in_args.mod,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,
            'diffusion_path': f'./checkpoints/unet_mp/{in_args.traffic_file}/diffusion_model_meta_final.pth',
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
            "TRAFFIC_SEPARATE": traffic_file,
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "traffic_movement_pressure_num",
            ],
            "log_writer": False,
            "DIC_REWARD_INFO": {
                "pressure": -0.25,
            },

            'is_test': True,
            #"inference_epoch": 0,
            'is_inference': in_args.inference,
            "sota_path": '../RobustLight/model/MPLight/' + traffic_file[:-5],
            "NOISE_SCALE": in_args.NOISE_SCALE,
            "DETECT_RATE": 2,
            "NOISE_TYPE": in_args.NOISE_TYPE, # 每一个level有不同类型的noise，譬如level为0噪声类型有guassion，uniform，qmin，action_diff
            "NOISE_LEVEL": in_args.NOISE_LEVEL, # 0为state上面加噪声，1是mask， 2是 mask掉其他交叉口的
            "device":"cuda:1",
            "index_maps" : {
                "W": [0, 1, 2],
                "E": [3, 4, 5],
                "N": [6, 7, 8],
                "S": [9, 10, 11]
            },
            "NOISE_DIRECTION": in_args.NOISE_DIRECTION.split(","), # 某个方向出错了
            "inference_config": DMBP_config,
        }

        if in_args.eightphase:
            dic_traffic_env_conf_extra["PHASE"] = {
                    1: [0, 1, 0, 1, 0, 0, 0, 0],
                    2: [0, 0, 0, 0, 0, 1, 0, 1],
                    3: [1, 0, 1, 0, 0, 0, 0, 0],
                    4: [0, 0, 0, 0, 1, 0, 1, 0],
                    5: [1, 1, 0, 0, 0, 0, 0, 0],
                    6: [0, 0, 1, 1, 0, 0, 0, 0],
                    7: [0, 0, 0, 0, 0, 0, 1, 1],
                    8: [0, 0, 0, 0, 1, 1, 0, 0]
                }
            dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                        'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']

        dic_path_extra = {
            # "PATH_TO_MODEL": os.path.join("model", in_args.memo, traffic_file + "_"
            #                               + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            # "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, traffic_file + "_"
            #                                        + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
            "PATH_TO_MODEL": os.path.join("model", in_args.memo,
                                          f'{in_args.traffic_file}_{in_args.inference}_{in_args.NOISE_SCALE}_{in_args.NOISE_TYPE}_{in_args.NOISE_LEVEL}_{in_args.NOISE_DIRECTION}.json_{time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time()))}'),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo,
                                                   f'{in_args.traffic_file}_{in_args.inference}_{in_args.NOISE_SCALE}_{in_args.NOISE_TYPE}_{in_args.NOISE_LEVEL}_{in_args.NOISE_DIRECTION}.json_{time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time()))}'),
            "PATH_TO_DATA": os.path.join("data", template, str(road_net)),
            "PATH_TO_ERROR": os.path.join("errors", in_args.memo)
        }

        deploy_dic_agent_conf = getattr(config, "DIC_BASE_AGENT_CONF")
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

        if in_args.multi_process:
            ppl = Process(target=pipeline_wrapper,
                          args=(deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path))
            process_list.append(ppl)
        else:
            pipeline_wrapper(dic_agent_conf=deploy_dic_agent_conf,
                             dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                             dic_path=deploy_dic_path)

    if in_args.multi_process:
        for i in range(0, len(process_list), in_args.workers):
            i_max = min(len(process_list), i + in_args.workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)

    return in_args.memo


if __name__ == "__main__":
    args = parse_args()

    main(args)


import copy
import numpy as np
from parameters import get_parameters

def parameters_model(mode=None, params_prev=None):
    if params_prev is not None:
        params = copy.deepcopy(params_prev)
    else:
        params = get_parameters()

    if mode is None or mode == 0:                  # 이상적인 모델
        print("perfect model parameter is used")
        return params

    nlinks = params['LASDRA']['total_link_number']
    odars = params['ODAR']

    if mode == 1:                                  # link 길이 2cm 줄임
        for odar in odars:
            odar.length -= 0.02

    elif mode == 2:                                # link 길이 2cm 늘림
        for odar in odars:
            odar.length += 0.02

    elif mode == 3:                                # link 길이 2cm 줄이고, 질량 100g 증가
        for odar in odars:
            odar.length -= 0.02
            odar.mass += 0.1

    elif mode == 777:                              # link 길이, 질량 무작위 변경, 무게중심 위치 조정
        print("random mass and a bit biased")
        for i in range(nlinks):
            odars[i].mass += (np.random.rand() - 0.5) * 0.3
            odars[i].length += (np.random.rand() - 0.5) * 0.005 + 0.002
        odars[-1].joint_to_com[0] -= 0.05
        odars[0].joint_to_com[0] += 0.05

    elif mode == 1107:                            # link 질량 재분배
        print("masses are biased to base")
        odars[-1].joint_to_com[0] -= 0.05
        odars[3].mass -= 0.16
        odars[2].mass += 0.31
        odars[1].mass -= 0.31
        odars[0].joint_to_com[0] += 0.05
        odars[0].mass += 0.43

    else:
        print("default param error are used")
        for odar in odars:
            odar.length += 0.02
            odar.mass += 0.2

    return params

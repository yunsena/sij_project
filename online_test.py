import os
from multiprocessing import Pool
import time
import threading
import subprocess
import csv
import json
import math
import numpy as np
from socket import *

dir_path = '/home/jeong/fall_detection_online/test/rec'
csv_dir_path = '/home/jeong/fall_detection_online/test/sw_csv'
openpose_cmd_1 = './build/examples/openpose/openpose.bin --image_dir ../fall_detection_online/test/rec/'
openpose_cmd_2 = '--write_images.. /fall_detection_online/test/rec/o-'
openpose_cmd_3 = '/ --write_json ../fall_detection_online/test/rec/o-'
openpose_cmd_4 = '/ --part_candidates'

LABELS = [
    "LYING",
    "FALLING",
    "ADL"
]


def search_dir_num(dirname):
    filenames = os.listdir(dirname)
    # print(len(filenames))
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)

        # ext = os.path.splitext(full_filename)[-1]
        # if ext == '.py':
        #     print(full_filename)

        # print(full_filename)

    return len(filenames)

#test1
def get_dir_path(dirname, index_num):
    dirnames = os.listdir(dirname)
    dirnames.sort()
    print(dirnames[index_num - 1])

    return dirnames[index_num - 1]


def parse_skeleton(csv_path, json_path):
    f = open(csv_path, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)

    # 폴더 내 파일 리스트 읽어오기
    path_dir = json_path
    # path_dir = '/home/jeong/다운로드/fall-detection-dataset/adl/o_adl-' + str(tmp) + '-cam0-rgb'
    file_list = os.listdir(path_dir)
    file_list.sort()

    for item in file_list:
        if item.find('keypoints') is not -1:
            # print(item)
            file_name = path_dir + '/' + item
            # test_file_name = './COCO_val2014_000000000474_keypoints.json'

            with open(file_name) as json_file:
                # json file load
                data = json.load(json_file)
                # joint list
                read_jnt = [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36,
                            37,
                            39, 40, 42, 43]
                w_set = []

                for i in read_jnt:
                    try:
                        # print(data['people'][0]['pose_keypoints_2d'][i])
                        w_set.append(data['people'][0]['pose_keypoints_2d'][i])

                    except IndexError:
                        # print(item + ' %d의 값을 가져오지 못함' % i)
                        w_set.append(0)

                # print(w_set)
                wr.writerow(w_set)
    f.close()


def csvtoArray(csv_path, data_type):
    with open(csv_path, 'r', encoding='utf-8') as dest_f:
        data_iter = csv.reader(dest_f, delimiter=',')
        data = [data for data in data_iter]
    data_array = np.asarray(data, dtype=data_type)
    return data_array


def interp(csv_path):
    # csv 읽어오기
    # csv_path = str(csv_path).replace(u'\ufeff', '')
    fr = open(csv_path, 'r', encoding='utf-8')
    rdr = csv.reader(fr)

    result = []
    for line in rdr:
        # print(line)
        result.extend(line)
    fr.close()

    # print(len(result) / 30)
    i_len = int(len(result) / 30)

    arr = np.array(result).reshape(i_len, 30)  # 30
    # 보간법 사용 지점
    # print(arr)
    # print(arr.shape)

    row = arr.shape[0]  # 행
    col = arr.shape[1]  # 열

    float_arr = arr.astype(np.float)

    # del abnormal data before interpolation
    # del_zero_data(float_arr, label)
    # del_broken_data(float_arr, label)

    print(float_arr.shape)
    # 열 단위로 읽어오기
    for j in range(0, col):  # 0, col
        # print('---------------------------col = %d-------------------------------' %(j))
        # print(float_arr[:, j])
        for i in range(0, row):  # 0, row
            if float_arr[i][j] == 0.0 and float_arr[i - 1][j] != 0:
                # 초기화
                front_val = 0.0
                f_row_num = 0
                back_val = 0.0
                b_row_num = 0
                dif_val = 0.0
                dif_row = 0

                # 현재 행이 0.0일 경우
                now_row = i

                while (True):
                    if now_row + 1 == row or now_row == 0:
                        break
                    else:
                        if float_arr[now_row + 1][j] == 0.0:
                            now_row = now_row + 1
                            pass
                        else:
                            back_val = float_arr[now_row + 1][j]
                            b_row_num = now_row + 1
                            break

                front_val = float_arr[i - 1][j]
                f_row_num = i - 1
                dif_val = float(format(abs(front_val - back_val), ".3f"))
                # print(front_val)
                # print(back_val)
                # print(dif_val)
                dif_row = b_row_num - f_row_num

                if (dif_val) == 0:
                    for dif_i in range(0, dif_row - 1):
                        float_arr[i + dif_i][j] = front_val
                else:
                    for dif_i in range(0, dif_row - 1):
                        if front_val < back_val:
                            val = (dif_val) * (dif_i + 1) / (dif_row)
                            float_arr[i + dif_i][j] = float(format(front_val + val, ".3f"))
                        else:
                            val = (dif_val) * (dif_i + 1) / (dif_row)
                            float_arr[i + dif_i][j] = float(format(front_val - val, ".3f"))
            else:
                pass

        # print(float_arr[:, j])
        # print('------------------------------------------------------------------')

    # # Neck(2,3)과 기준점(320, 160) 차이를 구하고 d_x, d_y 노말라이제이션 하기 (640,480)
    # for r in range(0, row):
    #     neck_x = float_arr[r][2]
    #     neck_y = float_arr[r][3]
    #     # neck 좌표 x, y 구하기
    #     if neck_x != 0 and neck_y != 0:
    #         # 기준점 좌표(320, 190)
    #         stand_x = 640
    #         stand_y = 480
    #         # neck과 기준점 사이의 차이
    #         d_x = stand_x - neck_x
    #         d_y = stand_y - neck_y
    #
    #         for c in range(0, col):
    #             if c % 2 == 0:
    #                 if float_arr[r][c] != 0:
    #                     float_arr[r][c] = float_arr[r][c] + d_x
    #                 else:
    #                     pass
    #
    #                 if float_arr[r][c] > 1280 or float_arr[r][c] < 0:
    #                     print('x의 범위를 벗어났습니다. 조정 좌표 x : %f' % (float_arr[r][c]))
    #             else:
    #                 if float_arr[r][c] != 0:
    #                     float_arr[r][c] = float_arr[r][c] + d_y
    #                 else:
    #                     pass
    #
    #                 if float_arr[r][c] > 960 or float_arr[r][c] < 0:
    #                     print('y의 범위를 벗어났습니다. 조정 좌표 y : %f' % (float_arr[r][c]))
    #     else:
    #         print('x = %f, y = %f' % (neck_x, neck_y))

    # print(float_arr)

    # print(float_arr.shape)
    return np.asarray(float_arr, dtype=float)


def data_normalization(arr):
    for j in range(0, arr.shape[1]):
        for i in range(0, arr.shape[0]):
            if j % 2 == 0:
                # print(arr[i][j])
                res = float(arr[i][j]) / 640
                arr[i][j] = format(float(res), ".6f")
            else:
                # print(arr[i][j])
                res = float(arr[i][j]) / 480
                arr[i][j] = format(float(res), ".6f")

            # if j % 2 == 0:
            #     #print(arr[i][j])
            #     res = float(arr[i][j]) / 1280
            #     arr[i][j] = format(float(res), ".6f")
            # else:
            #     # print(arr[i][j])
            #     res = float(arr[i][j]) / 960
            #     arr[i][j] = format(float(res), ".6f")

    return np.asarray(arr, dtype=float)


# 중심점 계산
def hm_mid_cal(r_point, l_point, i, j):
    try:
        res = float((abs(float(r_point) + float(l_point))) / 2)
    except ZeroDivisionError:
        # err_msg = csv_name +'%d행 %d열 값이 0입니다.' %(i,j)
        # ferr.write(str(err_msg))
        return 0

    return format(float(res), ".6f")


# 좌표점이 0인 부분 로그 작성
def zero_detect(pnt, i, j):
    try:
        res = float(float(pnt) / 1)
    except ZeroDivisionError:
        # err_msg = csv_name + '%d행 %d열 값이 0입니다.' % (i, j)
        # ferr.write(err_msg)
        return 0

    return format(float(res), ".6f")


# 인체 중심점 속력 변화 데이터 계산
def hm_mid_vel_var_cal(xt1, xt, yt1, yt, fps):
    try:
        # print(float(xt1), float(xt), float(yt1), float(yt), fps)
        res = (math.sqrt(math.pow((float(xt1) - float(xt)), 2) + math.pow((float(yt1) - float(yt)), 2))) / (1 / fps)
    except ZeroDivisionError:
        # err_msg = csv_name + '%d행 %d열 값이 0입니다.' % (i, j)
        # ferr.write(err_msg)
        return 0

    return format(float(res), ".6f")


def cal_angle(arr):
    buf = []
    for i in range(0, arr.shape[0]):
        neck_x = float(arr[i][2])
        neck_y = float(arr[i][3])
        mhip_x = float(arr[i][16])
        mhip_y = float(arr[i][17])
        # print('%f %f %f %f' % (neck_x, neck_y, mhip_x, mhip_y))
        vec_HN_x = float(neck_x - mhip_x)
        vec_HN_y = float(neck_y - mhip_y)
        vec_HG_x = float(mhip_x - mhip_x)
        vec_HG_y = float(480 - mhip_y)
        try:
            rad = math.acos(((vec_HN_x * vec_HG_x) + (vec_HN_y * vec_HG_y)) / (
                    math.sqrt(math.pow(vec_HN_x, 2) + math.pow(vec_HN_y, 2)) * math.sqrt(
                math.pow(vec_HG_x, 2) + math.pow(vec_HG_y, 2))))
            angle = math.degrees(rad)
            # normalization
            # buf.append(format(float(angle/360), ".6f"))
            buf.append(format(float(angle / 360), ".6f"))

        except:
            buf.append(format(float(0), ".6f"))
            # pass
            # print('float division by zero')

        # print(angle)

    buf = np.asarray(buf, dtype=float).reshape(arr.shape[0], -1)
    # print(buf)
    arr = np.concatenate((arr, buf), axis=1)
    # print(buf.shape)
    # print(arr.shape)

    return np.asarray(arr, dtype=float)


def sliding_window(arr, window_size, stride_size):
    # 기준선과 상체 사이의 각도 구하기
    # arr = cal_angle(arr)

    # print('----------------------------------------------------------- arr.shape[1] : %d ' %(arr.shape[1]))

    # 9프레임
    # win_size = 9
    win_size = window_size + 1
    isize = arr.itemsize
    i_len = arr.shape[0]
    # print(isize)

    windowed = np.lib.stride_tricks.as_strided(arr,
                                               shape=(int((arr.shape[0] - win_size + 1) / stride_size), win_size,
                                                      arr.shape[1]),
                                               strides=(
                                                   arr.shape[1] * isize * stride_size, arr.shape[1] * isize, isize))

    # print(windowed.shape)
    # 인체 중심점 데이터 추출
    w_vel = []  # 인체 중심점 속력 변화량
    w_hm_cor = []  # 인체 중심점 속력 변화량 좌표
    # print(i_len - win_size + 1)
    for i in range(0, i_len - win_size + 1):
        w_hm = []  # 인체 중심점 좌표 : 어깨 엉덩이 무릎 발 중심점 좌표 8개 리스트 9개 포함 9x8
        w_hm2 = []  # 인체 중심점 좌표 : 어깨 엉덩이 무릎 발 중심점 좌표 8개 리스트 9개 포함 9x8

        for j in range(0, win_size):
            hm_buf = []  # 인체 중심점 계산을 위한 좌표 저장 리스트 12개
            hm = []  # 인체 중심점 좌표 저장 리스트 8개
            # 인체 중심점 저장 0~9프레임 (12개)
            read_joint = [2, 3, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29]
            for rj in read_joint:
                hm_buf.append(windowed[i][j][rj])
            # print(hm_buf)

            # 인체 중심점 좌표 저장 (값이 0인지도 체크해야함-> 리턴 형태로 함수 구현하기***)
            hm.append(zero_detect(hm_buf[0], i, j))  # 어깨 x 좌표 0
            hm.append(zero_detect(hm_buf[1], i, j))  # 어깨 y 좌표 1
            hm.append(zero_detect(hm_buf[2], i, j))  # 엉덩이 x 좌표 2
            hm.append(zero_detect(hm_buf[3], i, j))  # 엉덩이 y 좌표 3
            hm.append(hm_mid_cal(hm_buf[8], hm_buf[4], i, j))  # 무릎 x 좌표 4
            hm.append(hm_mid_cal(hm_buf[9], hm_buf[5], i, j))  # 무릎 y 좌표 5
            hm.append(hm_mid_cal(hm_buf[10], hm_buf[6], i, j))  # 발목 x 좌표 6
            hm.append(hm_mid_cal(hm_buf[11], hm_buf[7], i, j))  # 발목 y 좌표 7
            # print(hm)
            w_hm.extend(hm)
            w_hm2.append(hm)
        # print('-----------------------------------------------------------------------------------')
        # print(len(w_hm))
        # print(w_hm)
        # print('-----------------------------------------------------------------------------------')
        # print(w_hm[0][0])
        # print(w_hm[1][0])
        w_hm_cor.append(w_hm2)
        # print('-----------------------------------------------------------------------------------')
        # print(w_hm_cor)
        # print(len(w_hm_cor[0]))
        # print(len(w_hm_cor[0][0]))
        # print(w_hm_cor[0][0][0])
        # print('-----------------------------------------------------------------------------------')
        fps = 30
        # 인체 중심점 속력 변화량 데이터 4개 (0인지 체크 위해 함수로 구현)
        for l in range(0, win_size - 1):
            m = l + 1
            vel_var = []  # 인체 중심점 속력 변화량
            for n in range(0, 4):
                vel_var.append(hm_mid_vel_var_cal(w_hm[((8 * m) + n)], w_hm[((8 * l) + n)], w_hm[((8 * m) + 1 + n)],
                                                  w_hm[((8 * l) + 1 + n)], fps))
            w_vel.append(vel_var)

        # print('-----------------------------------------------------------------------------------')
        # print(len(vel_var))
        # print(vel_var)
        # print('%d-----------------------------------------------------------------------------------' % i)
        # w_vel.extend(vel_var)

    # fw.close()
    # print('-----final----------------------------------------------------------------------------------')
    # print(len(w_vel))
    # print(w_vel[1][3])
    # print(w_vel)
    # ferr.close()  # 값이 0인 에러 저장 파일 close

    # print('-----final----------------------------------------------------------------------------------')
    cnt = 0
    whole_buf = []
    for i in range(0, i_len - win_size + 1):
        buf = []
        # print(i)
        for j in range(0, win_size - 1):
            # print(j)
            # 인체 중심점 데이터만 추출한다면 k 인덱스 for문 주석 처리
            for k in range(0, arr.shape[1]):
                buf.append(windowed[i][j][k])
            # for k in range(0, arr.shape[1]):
            #    buf.append(windowed[i][j][k])
            # angle 값 추가
            # buf.append(windowed[i][j][arr.shape[1]-1])
            # buf.append(windowed[i][j][arr.shape[1] - 1])
            # 인체 중심점 속력 변화량 데이터
            # for m in range(0, 8):
            #     buf.append(w_hm_cor[i][j][m])
            for l in range(0, 4):
                buf.append(w_vel[cnt][l])
                # print(w_vel[cnt][l])
            cnt = cnt + 1
        # print(len(buf))
        # print(buf)
        whole_buf.append(buf)
        # wr.writerow(buf)
    # print(len(whole_buf[0]))
    whole_list = sum(whole_buf, [])
    # print(len(whole_buf))
    # print(len(whole_buf[0]))z
    res_arr = np.array(whole_list).reshape((len(whole_buf), int(len(whole_buf[0]))))

    # print(res_arr.shape)
    # print(res_arr)

    return np.asarray(res_arr, dtype=float)


#
# def inference(sw_csv_path):
#     ################inference 단계
#     ###### SDUFall Dataset 영상 fall detection URFD + SDUFall
#     n_steps = win_size  # 128 timesteps per series
#     n_input = pdata_length  # 9 input parameters per timestep
#
#     # LSTM Neural Network's internal structure
#     n_hidden = 256  # Hidden layer num of features
#     n_classes = class_num  # Total classes (should go up, or should go down)
#
#     # Graph input/output
#     x = tf.placeholder(tf.float32, [None, n_steps, n_input], name="x")
#     y = tf.placeholder(tf.float32, [None, n_classes], name="y")
#
#     # Graph weights
#     weights = {
#         'hidden': tf.Variable(tf.random_normal([n_input, n_hidden]), name="weights_hidden"),
#         # Hidden layer weights
#         'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0), name="weights_out")
#     }
#
#     print(type(weights['out']))
#
#     biases = {
#         'hidden': tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden"),
#         'out': tf.Variable(tf.random_normal([n_classes]), name="bias_out")
#     }
#
#     pred = LSTM_RNN(x, weights, biases)
#
#     correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
#     sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
#     init = tf.global_variables_initializer()
#     sess.run(init)
#
#     # y_test_ = one_hot(y_test)
#     ###################### ckpt_10f_34_1_3c_acc9880_URFD_SDUFall
#     save_file = './ckpt_10f_34_1_3c_acc9880_URFD_SDUFall/train_model.ckpt'
#     # save_file = './ckpt_33f_8_3c_acc9864_URFD_SDUF/train_model.ckpt'
#     # save_file = './ckpt_30f_13_3c_acc9935/train_model.ckpt'
#     saver = tf.train.Saver()
#
#     alarm_cnt = 0
#
#     with tf.Session() as sess:
#         saver.restore(sess, save_file)
#
#         c_path = sw_csv_path
#
#         one_hot_predictions = sess.run(
#             pred,
#             feed_dict={
#                 x: csv_read(c_path, 34),
#                 #             y: y_test_
#             }
#         )
#
#         predictions = one_hot_predictions.argmax(1)
#         print('=========================================================================================')
#         print('%d frame, class : %d, feature_num : %d' % (win_size, class_num, pdata_length))
#         print('%s 영상입니다.' % (c_path))
#         print('추론 결과')
#         print(predictions)
#
#         win = 31
#         chk_alarm = 0
#
#         for j in range(0, len(predictions)):
#             start_index = j
#
#             if (j + win) >= len(predictions):
#                 end_index = len(predictions)
#
#             else:
#                 end_index = j + win
#
#             #     print(end_index)
#             #     print(predictions[j:end_index])
#             if (predictions[j:end_index].sum()) == 1 and predictions[j] == 1:
#                 if j != (end_index - 1):
#                     chk_alarm = chk_alarm + 1
#                     print('Fall Detection!!! %d index' % (j + 1))
#
#         if chk_alarm == 1:
#             print('Successfully Fall Detection!!!')
#
#


class OpenPoseExecution(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while True:
            # 최초 dir 수
            global init_dir_cnt
            # 가공 완료 dir 수 // 최초엔 fin_dir_cnt == init_dir_cnt
            global fin_dir_cnt
            tmp = search_dir_num(dir_path)
            global flag

            cur_dir_cnt = init_dir_cnt
            if cur_dir_cnt != tmp:
                # print("new dir appear num : %d" % (tmp))
                cur_dir_cnt = tmp

                if fin_dir_cnt < cur_dir_cnt:
                    fin_dir_cnt = fin_dir_cnt + 1
                    # fin_dir_cnt 폴더 OpenPose 수행
                    # 디렉토리 이름 받아오기
                    tmp_dir_name = get_dir_path(dir_path, fin_dir_cnt)
                    tmp_dir_path = dir_path + '/' + str(tmp_dir_name)

                    if flag == 0:
                        time.sleep(8)
                        flag = 1

                    wait_cnt = 0
                    while True:
                        tmp_dir_num = search_dir_num(tmp_dir_path)

                        if tmp_dir_num >= 150 or wait_cnt == 5:

                            if wait_cnt == 5:
                                flag = 0
                            # OpenPose 실행
                            # subprocess.call(
                            #    './build/examples/openpose/openpose.bin --image_dir ../fall_detection_online/test/rec/' + str(
                            #        tmp_dir_name) + ' --write_images ../fall_detection_online/test/rec/' + str(
                            #        tmp_dir_name) + '/ --write_json ../fall_detection_online/test/rec/' + str(
                            #        tmp_dir_name) + '/ --part_candidates', shell=True)
                            subprocess.call(
                                './build/examples/openpose/openpose.bin --image_dir ../fall_detection_online/test/rec/' + str(
                                    tmp_dir_name) + ' --write_json ../fall_detection_online/test/rec/' + str(
                                    tmp_dir_name) + '/ --part_candidates', shell=True)
                            wait_cnt = 0
                            break

                        else:
                            wait_cnt = wait_cnt + 1
                            time.sleep(0.5)

                    ############# Parsing 실행
                    parse_dir_name = get_dir_path(dir_path, fin_dir_cnt)
                    parse_csv_path = '/home/jeong/fall_detection_online/test/csv/' + str(parse_dir_name) + '.csv'
                    tmp_dir_name = get_dir_path(dir_path, fin_dir_cnt)
                    tmp_dir_path = dir_path + '/' + str(tmp_dir_name)
                    parse_skeleton(parse_csv_path, tmp_dir_path)

                    # Parsing된 csv를 통해 데이터 가공
                    # data array
                    arr = csvtoArray(parse_csv_path, str)
                    interp_arr = interp(parse_csv_path)
                    # data normalization
                    interp_arr = data_normalization(interp_arr)
                    # window sliding
                    window_size = 10
                    sw_arr = sliding_window(interp_arr, window_size, 1)

                    sw_csv_path = '/home/jeong/fall_detection_online/test/sw_csv/' + str(parse_dir_name) + '.csv'
                    with open(sw_csv_path, 'w') as test_csv:
                        for i in range(0, sw_arr.shape[0]):
                            csv_writer = csv.writer(test_csv)
                            csv_writer.writerow(sw_arr[i])

                    py_cmd = 'python online_test_inference.py'
                    subprocess.call("python online_test_inference.py '" + sw_csv_path + "'", shell=True)
                    """
                    f=open('/home/jeong/fall_detection_online/test/out/output.txt', 'w')
                    res = subprocess.check_output("python online_test_inference.py '" + sw_csv_path + "'", shell=True, universal_newlines=True)
                    f.write(res)
                    f.close()
                    #print('****************************************************************************')
                    #print(res)
                    #print(type(res))
                    #print('****************************************************************************')
                    # time.sleep(1)
                    """
                    # socket test
                    # clientSock.send('normal from online_test.py'.encode('utf-8'))

                    txt_path = '/home/jeong/fall_detection_online/test/out/output.txt'
                    flag = 0

                    with open(txt_path, 'r') as rtxt:
                        res = rtxt.read().splitlines()
                        res_num = len(res)
                        # print(res_num)
                        integrated_pred_win = 121

                        if res_num == 0:
                            pass

                        elif res_num == 1:
                            l1 = res[res_num - 1].split(',')
                            l1 = [int(x) for x in l1]

                            predictions = l1
                            predictions = np.asarray(predictions, dtype=int)

                            for j in range(0, len(predictions)):
                                start_index = j

                                if (j + integrated_pred_win) >= len(predictions):
                                    end_index = len(predictions)

                                else:
                                    end_index = j + integrated_pred_win

                                #     print(end_index)
                                #     print(predictions[j:end_index])
                                if (predictions[j:end_index].sum()) == 1 and predictions[j] == 1:
                                    if j != (end_index - 1):
                                        print('Integrated Pred : Fall Detection!!! %d index' % (j + 1))
                                        print(predictions)
                                        clientSock.send('warning from online_test.py'.encode('utf-8'))

                        elif res_num >= 2:
                            l1 = res[res_num - 2].split(',')
                            l1 = [int(x) for x in l1]

                            predictions = l1
                            predictions = np.asarray(predictions, dtype=int)

                            for j in range(0, len(predictions)):
                                start_index = j

                                if (j + integrated_pred_win) >= len(predictions):
                                    end_index = len(predictions)

                                else:
                                    end_index = j + integrated_pred_win

                                #     print(end_index)
                                #     print(predictions[j:end_index])
                                if (predictions[j:end_index].sum()) == 1 and predictions[j] == 1:
                                    if j != (end_index - 1):
                                        flag = 1
                                        # print('Integrated Pred : Fall Detection!!! %d index' % (j + 1))
                                        # print(predictions)
                                        # clientSock.send('warning from online_test.py'.encode('utf-8'))

                            if flag == 1:
                                l1 = res[res_num - 2].split(',')
                                l1 = [int(x) for x in l1]
                                l2 = res[res_num - 1].split(',')
                                l2 = [int(x) for x in l2]

                                predictions = l1 + l2

                                predictions = np.asarray(predictions, dtype=int)
                                # print(len(predictions))

                                for j in range(0, len(predictions)):
                                    start_index = j

                                    if (j + integrated_pred_win) >= len(predictions):
                                        end_index = len(predictions)

                                    else:
                                        end_index = j + integrated_pred_win

                                    #     print(end_index)
                                    #     print(predictions[j:end_index])
                                    if (predictions[j:end_index].sum()) == 1 and predictions[j] == 1:
                                        if j != (end_index - 1):
                                            # chk_alarm = chk_alarm + 1
                                            if flag == 1:
                                                print('Integrated Pred : Fall Detection!!! %d index' % (j + 1))
                                                print(predictions)
                                                clientSock.send('danger from online_test.py'.encode('utf-8'))

                            elif flag == 0:
                                l1 = res[res_num - 1].split(',')
                                l1 = [int(x) for x in l1]

                                predictions = l1
                                predictions = np.asarray(predictions, dtype=int)

                                for j in range(0, len(predictions)):
                                    start_index = j

                                    if (j + integrated_pred_win) >= len(predictions):
                                        end_index = len(predictions)

                                    else:
                                        end_index = j + integrated_pred_win

                                    #     print(end_index)
                                    #     print(predictions[j:end_index])
                                    if (predictions[j:end_index].sum()) == 1 and predictions[j] == 1:
                                        if j != (end_index - 1):
                                            print('Integrated Pred : Fall Detection!!! %d index' % (j + 1))
                                            print(predictions)
                                            clientSock.send('warning from online_test.py'.encode('utf-8'))

            else:
                time.sleep(1)


#
# class InferenceExecution (threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#
#
#     def run(self):
#         ################inference 단계
#         ###### SDUFall Dataset 영상 fall detection URFD + SDUFall
#         n_steps = win_size  # 128 timesteps per series
#         n_input = pdata_length  # 9 input parameters per timestep
#
#         # LSTM Neural Network's internal structure
#         n_hidden = 256  # Hidden layer num of features
#         n_classes = class_num  # Total classes (should go up, or should go down)
#
#         # Graph input/output
#         x = tf.placeholder(tf.float32, [None, n_steps, n_input], name="x")
#         y = tf.placeholder(tf.float32, [None, n_classes], name="y")
#
#         # Graph weights
#         weights = {
#             'hidden': tf.Variable(tf.random_normal([n_input, n_hidden]), name="weights_hidden"),
#             # Hidden layer weights
#             'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0), name="weights_out")
#         }
#
#         print(type(weights['out']))
#
#         biases = {
#             'hidden': tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden"),
#             'out': tf.Variable(tf.random_normal([n_classes]), name="bias_out")
#         }
#
#         pred = LSTM_RNN(x, weights, biases)
#
#         correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
#         sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
#         init = tf.global_variables_initializer()
#         sess.run(init)
#
#         # y_test_ = one_hot(y_test)
#         ###################### ckpt_10f_34_1_3c_acc9880_URFD_SDUFall
#         save_file = './ckpt_10f_34_1_3c_acc9880_URFD_SDUFall/train_model.ckpt'
#         # save_file = './ckpt_33f_8_3c_acc9864_URFD_SDUF/train_model.ckpt'
#         # save_file = './ckpt_30f_13_3c_acc9935/train_model.ckpt'
#         saver = tf.train.Saver()
#
#         alarm_cnt = 0
#         fail_list = []
#
#         while True:
#             #print("Inference")
#             global init_sw_csv_dir_cnt
#             global fin_sw_csv_cnt
#             tmp = search_dir_num(csv_dir_path)
#             global flag2
#
#             cur_sw_csv_cnt = init_sw_csv_dir_cnt
#             if cur_sw_csv_cnt != tmp:
#                 cur_sw_csv_cnt = tmp
#
#                 if fin_sw_csv_cnt < cur_sw_csv_cnt:
#                     fin_sw_csv_cnt = fin_sw_csv_cnt + 1
#
#                     tmp_dir_name = get_dir_path(csv_dir_path, fin_sw_csv_cnt)
#
#                     sw_csv_path = '/home/jeong/fall_detection_online/test/sw_csv/' + str(tmp_dir_name)
#
#                     print('-------------------------------------------------------------------------------------------')
#                     print(sw_csv_path)
#
#                     wait_cnt = 0
#                     while True:
#                         arr = csvtoArray(sw_csv_path, str)
#                         print(arr.shape[0])
#                         print('-------------------------------------------------------------------------------------------')
#
#                         if arr.shape[0] > 0:
#                             with tf.Session() as sess:
#                                 saver.restore(sess, save_file)
#
#                                 c_path = sw_csv_path
#
#                                 one_hot_predictions = sess.run(
#                                     pred,
#                                     feed_dict={
#                                         x: csv_read(c_path, 34),
#                                         #             y: y_test_
#                                     }
#                                 )
#
#                                 predictions = one_hot_predictions.argmax(1)
#                                 print('=========================================================================================')
#                                 print('%d frame, class : %d, feature_num : %d' % (win_size, class_num, pdata_length))
#                                 print('%s 영상입니다.' % (tmp_dir_name))
#                                 print('추론 결과')
#                                 print(predictions)
#
#                                 win = 31
#                                 chk_alarm = 0
#
#                                 for j in range(0, len(predictions)):
#                                     start_index = j
#
#                                     if (j + win) >= len(predictions):
#                                         end_index = len(predictions)
#
#                                     else:
#                                         end_index = j + win
#
#                                     #     print(end_index)
#                                     #     print(predictions[j:end_index])
#                                     if (predictions[j:end_index].sum()) == 1 and predictions[j] == 1:
#                                         if j != (end_index - 1):
#                                             chk_alarm = chk_alarm + 1
#                                             print('Fall Detection!!! %d index' % (j + 1))
#
#                                 if chk_alarm == 1:
#                                     alarm_cnt = alarm_cnt + 1
#                                 else:
#                                     fail_list.append(tmp)
#
#                                 print('=========================================================================================')
#                                 print(' ')
#                                 wait_cnt = 0
#                                 break
#
#                         else:
#                             time.sleep(0.5)
#
#             else:
#                 time.sleep(1)


if __name__ == '__main__':
    clientSock = socket(AF_INET, SOCK_STREAM)
    clientSock.connect(('192.168.0.72', 5000))

    flag = 0
    cur_dir_cnt = 0
    init_dir_cnt = search_dir_num(dir_path)
    fin_dir_cnt = init_dir_cnt
    print('init dir cnt : %d' % (init_dir_cnt))

    flag2 = 0
    cur_sw_csv_cnt = 0
    init_sw_csv_dir_cnt = search_dir_num(csv_dir_path)
    # init_sw_csv_dir_cnt = 0
    fin_sw_csv_cnt = init_sw_csv_dir_cnt
    print('fin_sw_csv_cnt : %d' % (fin_sw_csv_cnt))

    time.sleep(2)
    opc_thread = OpenPoseExecution()
    # inf_thread = InferenceExecution()
    opc_thread.start()
    # inf_thread.start()



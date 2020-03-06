import os
import sys
import cv2
import datetime
import math
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.ParkingSpot import ParkingSpot
from src.Config import Config
from src.Logger import Logger
from src.DnnModel import DnnModel
from src.Motion import Motion
from src.Camera import Camera
from timeit import default_timer as timer
from src.Bussiness_logic import *
from src.ObjectTracker import tracker, create_tracker


def flow_handler(args):
    if len(args) == 1:
        print("1 positional argument missing!")
        print("\tuse_captured_video: detection on captured video\n\tuse_onboard_camera: detection on real-time video "
              "from tx2 camera\n\tuse_local_rtsp: detection on captured video with rtsp server")
        return

    father_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    log = Logger(father_dir + '/results/log/smart_parking.log', 'info')

    if args[1] == 'use_captured_video' or args[1] == 'use_onboard_camera' or args[1] == 'use_local_rtsp':
        detection_pipeline(log, args=args)
    else:
        print("\tuse_captured_video: detection on captured video\n\tuse_onboard_camera: detection on real-time video "
              "from tx2 camera\n\tuse_local_rtsp: detection on captured video with rtsp server")
        return


def detection_pipeline(log, args=None):
    """
    Detection main function
    :param log:
    :param args:
    :return:
    """
    config = Config(log)

    img_archive_dir = config.img_archive_dir
    if not os.path.exists(img_archive_dir):
        os.makedirs(img_archive_dir)

    father_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    '''would need a starting time for video'''
    config.st = datetime.datetime.now()
    if config.socket_enable:
        config.start_status_report_timer()
        if config.reupload_enbale:
            config.re_upload_cache()
        config.start_log_upload_timer()

    # Initialize for each camera
    if args[1] in ['use_onboard_camera', 'use_local_rtsp']:
        camera = Camera(log, args[1], local=False, cfg_path=father_dir + '/camera_info.txt')
    elif args[1] == 'use_captured_video':
        camera = Camera(log, args[1], local=True, cfg_path=father_dir + '/camera_info.txt')
    else:
        log.logger.error('Unrecognized execution mode argument {}'.format(args[1]))
        exit()

    dnn_model = DnnModel(log, config)
    for cam_id in camera.info:
        motion_roi = camera.info[cam_id]['coord']['bounding_rect']
        cov_roi, spot_roi1, spot_roi2 = camera.info[cam_id]['coord']['roi_rects']
        parking_ids = camera.info[cam_id]['parking_ids']
        transformation_matrix = camera.info[cam_id]['transformation_matrix']

        # convert to (left, top, right, bottom), may change in yaml file later.
        motion_roi = [motion_roi[0], motion_roi[1], motion_roi[0] + motion_roi[2], motion_roi[1] + motion_roi[3]]
        cov_roi = [cov_roi[0], cov_roi[1], cov_roi[0] + cov_roi[2], cov_roi[1] + cov_roi[3]]
        spot_roi1 = [spot_roi1[0], spot_roi1[1], spot_roi1[0] + spot_roi1[2], spot_roi1[1] + spot_roi1[3]]
        spot_roi2 = [spot_roi2[0], spot_roi2[1], spot_roi2[0] + spot_roi2[2], spot_roi2[1] + spot_roi2[3]]
        # trace_roi are used for backtrace
        trace_roi = [cov_roi[0] * 2 - cov_roi[2], cov_roi[1], cov_roi[0], cov_roi[3]]

        # set parameters for different cameras
        if int(cam_id) == 1:
            distance_thres = [-0.2, 0.9]
        elif int(cam_id) == 2:
            distance_thres = [-0.3, 0.9]
        elif int(cam_id) == 3:
            distance_thres = [-0.3, 0.9]
        else:
            distance_thres = [-0.3, 0.9]
        parking_spot = ParkingSpot(log, config, cov_roi, spot_roi1, spot_roi2, parking_ids, trace_roi, distance_thres,
                                   camera.info[cam_id]['initial_state'])
        logic = B_logic(log, 8)
        motion = Motion(log, motion_roi, dnn_model, parking_spot, logic, transformation_matrix)
        camera.info[cam_id]['ParkingSpot'] = motion

    if args[1] == 'use_onboard_camera' or args[1] == 'use_local_rtsp':
        camera.capture_pic_multithread()

    while config.cap_on is True:
        # camera robin-round
        for cam_id in camera.info:
            motion = camera.info[cam_id]['ParkingSpot']
            parking_draw = camera.info[cam_id]['parking_draw']
            # Read frame-by-frame
            if args[1] == 'use_onboard_camera' or args[1] == 'use_local_rtsp':
                ret, video_cur_frame, video_cur_time, video_cur_cnt = camera.get_frame_from_queue(cam_id)
            else:
                cap = camera.info[cam_id]['video_cap']
                if cap is None:
                    continue
                # Current position of the video file in seconds, indicate current time
                video_cur_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                # Index of the frame to be decoded/captured next
                video_cur_cnt = cap.get(cv2.CAP_PROP_POS_FRAMES)
                ret, video_cur_frame = cap.read()

            if ret is False:
                log.logger.info('Video Analysis Finished!')
                config.cap_on = False
                break
            elif ret is None:
                continue

            '''run each ParkingSpot'''
            frame_info = {'cur_frame': video_cur_frame, 'cur_time': video_cur_time, 'cur_cnt': video_cur_cnt}
            if video_cur_cnt % motion.read_step == 0:
                # frame_for_display is returned for debug, need to change later
                frame_for_display = detection_flow(log, motion, dnn_model, frame_info)

                # the rest is added only for debug
                show_camera(motion, frame_for_display, parking_draw)

    camera.release_video_cap()
    config.stop_status_report_timer()
    config.ftp_upload_enable = False


def detection_flow(log, motion, dnn_model, frame_info):
    log.logger.debug('Run detection flow for one frame start...')

    motion.cur_frame = frame_info['cur_frame'].copy()
    motion.cur_time = frame_info['cur_time']
    motion.cur_cnt = frame_info['cur_cnt']
    motion.frame_buffer.append(motion.cur_frame)

    log.logger.info('Current cnt: {}.'.format(motion.cur_cnt))

    start_program = timer()

    # this is the start of the program
    if len(motion.frame_buffer) == 1:
        return motion.cur_frame

    motion_boxes, trace_motion_boxes, frame_for_display = motion.motion_detection()

    call_vehicle_det, det_frame, det_motion_boxes = motion.call_vehicle_det(motion_boxes)
    if call_vehicle_det:
        img_name = motion.parking_spot.config.src_img_dir + 'inp_0.jpg'
        cv2.imwrite(img_name, det_frame)
        vehicle_boxes = dnn_model.vehicle_det()

        for spot_id in range(2):
            if motion_boxes:
                for motion_box in motion_boxes:
                    motion.filter_motion_box(motion_box, vehicle_boxes[0], spot_id, spot_identify=False)
            else:
                # this is only used to identify the spot
                # only need to remove unrelated/parked cars, do not need to check coverage
                for motion_box in det_motion_boxes:
                    motion.filter_motion_box(motion_box, vehicle_boxes[0], spot_id, spot_identify=True)
    elif 'empty' in motion.parking_spot.state and trace_motion_boxes:
        for spot_id in range(2):
            motion.update_backtrace(trace_motion_boxes, spot_id)

    motion.parking_spot.state_bk = [s for s in motion.parking_spot.state]

    for spot_id in range(2):
        if motion.motion_coverage[spot_id]:
            # enter_flag, _ = motion.parking_spot.check_enter(motion.motion_coverage[spot_id])
            leave_flag, _ = motion.parking_spot.check_leave(motion.motion_coverage[spot_id])
            # weak_leave_flag = motion.parking_spot.check_leave_loose(motion.motion_coverage[spot_id])
            # #check for missing cases
            # if motion.parking_spot.state[spot_id] == 'parking':
            #     if motion.motion_coverage[spot_id][-1][0] >= 0.3 and enter_flag:
            #       motion.logic.potential_enter[spot_id] = motion.cur_time
            #     if leave_flag == False and weak_leave_flag:
            #       motion.logic.potential_leave[spot_id] = motion.cur_time

        vehicles4plate, street_view_vehicles = motion.update_state(motion_boxes, spot_id)

        # #update the logic's local state based on update_state result
        # motion.logic.update_local_state(motion.parking_spot.state[spot_id], motion.parking_spot.parking_ids[spot_id])

        if motion.call_plate_identify:
            motion.save_cropped_vehicles(vehicles4plate, motion.parking_spot.config.cropped_vehicle_dir)
            motion.save_original_images(street_view_vehicles, motion.parking_spot.config.street_view_dir)

            frame_lp_dict, lp_maj, lp_type = motion.plate_identifier()

            # triggered tracker when no license plate or incomplete license plate is detected
            if leave_flag:
                if frame_lp_dict is None or len(lp_maj) < 7 and leave_flag:
                    motion.call_tracker = True

            # if motion.parking_spot.state[spot_id] == 'parking':
            #   motion.logic.update_plate_info(lp_maj, motion.parking_spot.parking_ids[spot_id])
            # #result check and correction by logic
            # if motion.parking_spot.state[spot_id] == 'empty':
            #   motion.logic.leave_spot_check(motion.parking_spot,lp_maj,  motion.parking_spot.parking_ids[spot_id])

            log.logger.info('Car info: {}, {}'.format(lp_maj, lp_type))
            log.logger.info('Spot {}, state changes: {} -> {}'.format(motion.parking_spot.parking_ids[spot_id],
                                                                      motion.parking_spot.state_bk[spot_id],
                                                                      motion.parking_spot.state[spot_id]))

            motion.parking_spot.update_upload(frame_lp_dict, lp_maj)
            motion.parking_spot.clear_cache_folder()

            motion.parking_spot.upload()

        if motion.call_tracker and street_view_vehicles is not None and leave_flag:
            print('start SORT tracking for the first time')
            bbx_tracker = []
            for item in street_view_vehicles[-1][0][2]:
                bbx_tracker.append(item)
            bbx_tracker.append(street_view_vehicles[-1][0][4])
            bbx_and_confidence = np.array([bbx_tracker])
            img_name_cache = street_view_vehicles[-1][0][1]

            # create instance of the SORT tracker
            motion.sort_tracker[spot_id] = create_tracker()
            frame_for_display, trackers = tracker(motion.sort_tracker[spot_id], frame_for_display, bbx_and_confidence)
            for item in trackers:
                id = int(item[4])
                # store ID of the target we cared about
                # and store the moment we start the tracker
                motion.target_id[spot_id] = id
                motion.tracker_triggered_time[spot_id] = motion.cur_time

                if id not in motion.tracker_dict[spot_id].keys():
                    motion.tracker_dict[spot_id][id] = dict()
                    motion.tracker_dict[spot_id][id][img_name_cache] = [int(item[0]), int(item[1]), int(item[2]), int(item[3])]
                else:
                    motion.tracker_dict[spot_id][id][img_name_cache] = [int(item[0]), int(item[1]), int(item[2]), int(item[3])]

            motion.call_tracker = False
            motion.call_tracker_subsequent = True

        if motion.call_tracker_subsequent and motion.sort_tracker[spot_id] is not None:
            cam_id = str(motion.parking_spot.parking_ids[1] // 2)
            img_id = str(round(motion.cur_time % 1000, 1))
            img_tracker_filename = 'img_' + cam_id + '_' + str(spot_id) + '_' + '0' * (5 - len(img_id)) + img_id + '.jpg'
            img_tracker = motion.parking_spot.config.tracker_view_dir + img_tracker_filename
            cv2.imwrite(img_tracker, motion.cur_frame)

            img_name = motion.parking_spot.config.src_img_dir + 'inp_0.jpg'
            cv2.imwrite(img_name, motion.cur_frame)
            vehicle_boxes_tracker = dnn_model.vehicle_det()
            bbx_tracker_subsequent = np.array(vehicle_boxes_tracker[0])[:, 1]
            confidence_tracker_subsequent = np.array(vehicle_boxes_tracker[0])[:, 3]
            for iii in range(len(bbx_tracker_subsequent)):
                bbx_tracker_subsequent[iii].append(confidence_tracker_subsequent[iii])
            bbx_and_confidence_subsequent = []
            for item in bbx_tracker_subsequent:
                bbx_and_confidence_subsequent.append(np.array(item))
            frame_for_display, trackers = tracker(motion.sort_tracker[spot_id], frame_for_display, np.array(bbx_and_confidence_subsequent))
            for item in trackers:
                id = int(item[4])
                if id not in motion.tracker_dict[spot_id].keys():
                    motion.tracker_dict[spot_id][id] = dict()
                    motion.tracker_dict[spot_id][id][img_tracker] = [int(item[0]), int(item[1]), int(item[2]), int(item[3])]
                else:
                    motion.tracker_dict[spot_id][id][img_tracker] = [int(item[0]), int(item[1]), int(item[2]), int(item[3])]

            motion.call_make_up_lp_det = True

        if motion.call_make_up_lp_det and motion.tracker_triggered_time[spot_id] is not None \
                and motion.cur_time - motion.tracker_triggered_time[spot_id] >= 8:
            tracker_info_dict = motion.tracker_dict[spot_id][motion.target_id[spot_id]]
            # match the data structure with that save_cropped_vehicles needs
            tracker_info_array = []
            for key in tracker_info_dict.keys():
                tmp_list = [[None], key, tracker_info_dict[key], 'car', 1]
                tracker_info_array.append(tmp_list)

            # only use half of the pictures we tracked
            half_vehicles = math.floor(len(tracker_info_array)/2)
            motion.save_cropped_vehicles_tracker([tracker_info_array[half_vehicles:]], motion.parking_spot.config.cropped_vehicle_dir)

            frame_lp_dict, lp_maj, lp_type = motion.plate_identifier()
            print(lp_maj, lp_type)
            log.logger.info('At time: {}'.format(round(motion.cur_time, 1)))
            log.logger.info('Tracked Car info: {}, {}'.format(lp_maj, lp_type))
            # motion.parking_spot.update_upload_lp(frame_lp_dict, lp_maj, spot_id)

            # TODO maybe clear only of frame_lp_dict is not None and len(lp_maj) >= 7? Will it ends in an endless tracking?
            motion.clear_tracker(spot_id)
            motion.parking_spot.clear_tracker_folder()
            # motion.parking_spot.upload()
            print('Tracker Out')

    end_program = timer()

    log.logger.debug('To process one frame need {}s'.format(end_program - start_program))
    log.logger.debug('Run detection flow for one frame done.')

    return frame_for_display


def show_camera(motion, frame, parking_draw):
    """
    Used to show detected frames on screen.
    :param motion:
    :param frame:
    :param parking_draw:
    :return:
    """
    color_box = [(0, 255, 0)] * 2
    for index, state in enumerate(motion.parking_spot.state):
        if state == 'parking':
            color_box[index] = (0, 0, 255)
        elif state == 'waiting':
            color_box[index] = (0, 69, 188)
        else:
            color_box[index] = (0, 255, 0)
    color = (255, 1, 158)
    color_coverage_roi = (250, 180, 50)

    cv2.drawContours(frame, [parking_draw[0][0]], contourIdx=-1, color=color, thickness=2,
                     lineType=cv2.LINE_8)
    cv2.drawContours(frame, [parking_draw[0][1]], contourIdx=-1, color=color_box[0],
                     thickness=2, lineType=cv2.LINE_8)
    cv2.drawContours(frame, [parking_draw[0][2]], contourIdx=-1, color=color_box[1],
                     thickness=2, lineType=cv2.LINE_8)
    cv2.drawContours(frame, [parking_draw[0][3]], contourIdx=-1, color=color_coverage_roi,
                     thickness=2, lineType=cv2.LINE_8)

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    cv2.imshow('camera', frame)
    cv2.waitKey(40)


if __name__ == '__main__':
    sys.path.append('src')
    flow_handler(sys.argv)

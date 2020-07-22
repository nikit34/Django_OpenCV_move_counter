import numpy as np
import cv2
import pandas as pd
import os
import time


class Interface:
    def __init__(self, space):
        if space in ['Camera', 'VideoStatistic', 'FileStatistic']:
            self.env = '/local/'
            self.root_path = os.path.dirname(os.path.abspath(__file__)) + '/data'

        if space in ['LineBounds', 'CountCrossLine']:
            self.lines = [
                {
                    'id_': 'top-left',
                    'p1': (70, 40),
                    'p2': (99, 30),
                    'rgb': (0, 0, 255),  # red
                    'bond': 2,
                    'cross': 2,
                },
                # {
                #     'id_': 'bottom-left',
                #     'p1': (70, 70),
                #     'p2': (99, 80),
                #     'rgb': (0, 0, 255),
                #     'bond': 2,
                #     'cross': 1,
                # },
                # {
                #     'id_': 'bottom-right',
                #     'p1': (30, 70),
                #     'p2': (1, 80),
                #     'rgb': (0, 0, 255),
                #     'bond': 2,
                #     'cross': 1,
                # },
                {
                    'id_': 'top-right',
                    'p1': (30, 40),
                    'p2': (1, 30),
                    'rgb': (0, 0, 255),
                    'bond': 2,
                    'cross': 2,
                },
            ]

        if space in ['main', 'LineBounds']:
            # resize
            self.ratio = 0.5

        if space == 'main':
            # createBackgroundSubtractorMOG2
            self.history = 100  # учет прошлых шагов
            self.varThreshold = 50
            self.detectShadows = True  # отсекает тени

            # get_structuring_element
            self.shape = cv2.MORPH_ELLIPSE
            self.ksize = (20, 20)
            self.anchor = (-1, -1)

            # threshold
            self.thresh = 200  # порог разницы пикселей
            self.maxval_threshold = 255
            self.type = cv2.THRESH_TRIANGLE

            # count_cross_line
            self.min_area = 500  # мин макс площадь объекта
            self.max_area = 6000

            # if old_center
            self.max_rad = 1  # порог для определения уникальности объектов

        elif space == 'CountCrossLine':
            self.epsilon = 400  # мин расстояние до границы пересечения
            self.timeout = 0.2  # таймаут приостановки учета пересечений

        elif space == 'Camera':
            self.record = 'buffer.txt'

        elif space == 'VideoStatistic':
            self.name_video = 'test0.avi'

        elif space == 'FileStatistic':
            self.statistic = 'test0.csv'


# class Camera(Interface):
#     def __init__(self):
#         super(Camera, self).__init__('Camera')
#         self.cap = cv2.VideoCapture(self.get_full_path_input())
#
#     def get_full_path_input(self) -> str:
#         path = self.root_path + self.env
#         file = open(path + self.record)
#         full_path = path + 'in/' + file.read()
#         return full_path


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def get_param_camera(self):
        params_ = {
            'frames_count': self.cap.get(cv2.CAP_PROP_FRAME_COUNT),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        }
        return params_

    def read(self):
        return self.cap.read()

    def stop_record(self):
        self.cap.release()


class Console:
    def __init__(self, cam):
        super(Console, self).__init__()
        self.params = cam.get_param_camera()

    def log_input(self):
        print('****************************************************')
        print('**************** Параметры видео *******************')
        print('* Количество кадров:                          ', self.params['frames_count'])
        print('* FPS:                                        ', self.params['fps'])
        print('* разрешение:                                 ',
              self.params['width'], ' x, ', self.params['height'], ' px')
        print('* Продолжительность:                     ',
              round((self.params['frames_count'] / (self.params['fps'] + 1)), 1), ' сек.')
        print('****************************************************')


class FileStatistic(Interface):
    def __init__(self):
        self.df = pd.DataFrame()
        self.frame_number = 0
        self.obj_cross_up = 0
        self.obj_cross_down = 0
        self.obj_id = []
        self.obj_crossed = []
        self.obj_total = 0
        super(FileStatistic, self).__init__('FileStatistic')

    def set_index(self):
        self.df.index.name = 'Frames'

    def save_data(self):
        self.df.to_csv(self.root_path + self.env + 'out/' + self.statistic, mode='w+')


class VideoStatistic(Interface):
    def __init__(self, cam):
        Interface.__init__(self, 'VideoStatistic')
        self.params = cam.get_param_camera()
        self.video = None

    def set_record(self):
        path = self.root_path + self.env + 'out/' + self.name_video
        self.video = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*'XVID'),
            int(self.params['fps']),
            (int(self.params['width']), int(self.params['height']))
        )

    def write_record(self, img):
        self.video.write(img)

    def stop_record(self):
        self.video.release()


class LineBounds(Interface):
    def __init__(self, cam):
        Interface.__init__(self, 'LineBounds')
        self.param = cam.get_param_camera()
        self.count_lines = len(self.lines)
        self.coord_p1 = [(0, 0) for _ in range(self.count_lines)]
        self.coord_p2 = [(0, 0) for _ in range(self.count_lines)]
        self.rgb = [(0, 0, 0) for _ in range(self.count_lines)]
        self.bond = [0 for _ in range(self.count_lines)]

    def create_lines(self):
        for i_, line in enumerate(self.lines):
            self.coord_p1[i_] = (
                int(self.param['width'] * line['p1'][0] / 100 * self.ratio),
                int(self.param['height'] * line['p1'][1] / 100 * self.ratio)
            )
            self.coord_p2[i_] = (
                int(self.param['width'] * line['p2'][0] / 100 * self.ratio),
                int(self.param['height'] * line['p2'][1] / 100 * self.ratio)
            )
            self.rgb[i_] = line['rgb']
            self.bond[i_] = line['bond']

    def update_lines(self, img):
        for i_ in range(self.count_lines):
            cv2.line(img, self.coord_p1[i_], self.coord_p2[i_], self.rgb[i_], self.bond[i_])

    def get_coord_lines(self):
        return self.coord_p1, self.coord_p2


def follow_rectangle(img, cx_, cy_, cnt_):
    x, y, w, h = cv2.boundingRect(cnt_)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, str(cx_) + ', ' + str(cy_), (cx_ + 10, cy_ + 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 1)
    cv2.drawMarker(img, (cx_, cy_), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)


class CountCrossLine(Interface):
    def __init__(self, lines):
        super(CountCrossLine, self).__init__('CountCrossLine')
        self.coord_lines = lines.get_coord_lines()
        self.count_cross = [0 for _ in range(len(self.lines))]
        self.done_cross = [False for _ in range(len(self.lines))]
        self.total = 0
        self.last_time = [0. for _ in range(len(self.lines))]

    def filter_cross(self, cx_, cy_):
        for i_line, line in enumerate(self.lines):
            if self.timeout < time.time() - self.last_time[i_line]:
                if square_dist_point_line((cx_, cy_), self.coord_lines[0][i_line], self.coord_lines[1][i_line]) < self.epsilon and \
                        square_min_dist_extreme_point_circles((cx_, cy_), self.coord_lines[0][i_line], self.coord_lines[1][i_line]) <= \
                        square_radius_circles(self.coord_lines[0][i_line], self.coord_lines[1][i_line]) + self.epsilon:
                    self.count_cross[i_line] += 1
                    self.last_time[i_line] = time.time()

                    if self.count_cross[i_line] >= line['cross']:
                        self.done_cross[i_line] = True
                        if False not in self.done_cross:
                            self.update()

    def switch_color_line(self, obj_line_bounds):
        for i_ in range(len(self.lines)):
            if self.count_cross[i_] == 1:
                obj_line_bounds.rgb[i_] = (0, 100, 100)
            elif self.count_cross[i_] == 2:
                obj_line_bounds.rgb[i_] = (0, 200, 200)
            elif self.count_cross[i_] == 3:
                obj_line_bounds.rgb[i_] = (0, 255, 255)  # light green
            elif self.done_cross[i_]:
                obj_line_bounds.rgb[i_] = (0, 255, 0)  # green
            else:
                obj_line_bounds.rgb[i_] = (0, 0, 255)  # red

    def update(self):
        self.count_cross = [0 for _ in range(len(self.lines))]
        self.done_cross = [False for _ in range(len(self.lines))]
        self.total += 1


def filter_area(current_contours, min_area, max_area):
    for i_contour in range(len(current_contours)):
        if hierarchy[0][i_contour][3] == -1:
            area = cv2.contourArea(current_contours[i_contour])
            if min_area < area < max_area:
                yield current_contours[i_contour]


def get_center_moment(current_contours, min_area, max_area):
    for cnt_ in filter_area(current_contours, min_area, max_area):
        moment = cv2.moments(array=cnt_, binaryImage=False)
        cx_ = int(moment['m10'] / moment['m00'])
        cy_ = int(moment['m01'] / moment['m00'])
        yield cx_, cy_, cnt_


def square_min_dist_extreme_point_circles(point, line1, line2):
    sq_dist_p1 = pow((line1[0] - point[0]), 2) + pow((line1[1] - point[1]), 2)
    sq_dist_p2 = pow((line2[0] - point[0]), 2) + pow((line2[1] - point[1]), 2)
    return min(sq_dist_p1, sq_dist_p2)


def square_radius_circles(line1, line2):
    return (pow((line1[0] - line2[0]), 2) + pow((line1[1] - line2[1]), 2)) / 4


def square_dist_point_line(point, line1, line2):
    area_double_triangle = abs(
        (line2[1] - line1[1]) * point[0] -
        (line2[0] - line1[0]) * point[1] +
        line2[0] * line1[1] - line2[1] * line1[0]
    )
    dist_line = pow((line2[1] - line1[1]), 2) + pow((line2[0] - line1[0]), 2)
    return int(pow(area_double_triangle, 2) / (dist_line + 1))


if __name__ == "__main__":
    interface = Interface('main')
    camera = Camera()
    console = Console(camera)
    file_statistic = FileStatistic()
    video_statistic = VideoStatistic(camera)
    line_bounds = LineBounds(camera)
    count_cross_line = CountCrossLine(line_bounds)

    foreground_bg = cv2.createBackgroundSubtractorMOG2(
        history=interface.history,
        varThreshold=interface.varThreshold,
        detectShadows=interface.detectShadows
    )  # разделить на два слоя bg и fg

    console.log_input()
    params = camera.get_param_camera()
    file_statistic.set_index()
    video_statistic.set_record()

    line_bounds.create_lines()

    while camera.cap.isOpened():
        ret, frame = camera.read()
        if ret:
            try:
                image = cv2.resize(src=frame, dsize=(0, 0), fx=interface.ratio, fy=interface.ratio)
            except cv2.error as e:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            foreground_mask = foreground_bg.apply(gray)  # создание фона вычитания ч/б изображения

            kernel = cv2.getStructuringElement(
                shape=interface.shape,
                ksize=interface.ksize,
                anchor=interface.anchor
            )  # применение морфологического ядра

            closing = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)  # удаляем черный шум у белых частей
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)  # удаляем белый шум снаружи черных частей
            dilation = cv2.dilate(opening, kernel)  # выравниваем границы по внешнему контуру
            _, arr_bins = cv2.threshold(
                src=dilation,
                thresh=interface.thresh,
                maxval=interface.maxval_threshold,
                type=interface.type
            )  # разделение по thresh с присвоением 0 или max из всех значений
            contours, hierarchy = cv2.findContours(arr_bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            hull = [cv2.convexHull(c) for c in filter_area(
                contours, interface.min_area, interface.max_area
            )]
            cv2.drawContours(image, hull, -1, (200, 0, 0), 1)

            line_bounds.update_lines(image)

            cxx = np.zeros(len(contours))
            cyy = np.zeros(len(contours))
            for i, (cx, cy, cnt) in enumerate(get_center_moment(contours, interface.min_area, interface.max_area)):
                count_cross_line.filter_cross(cx, cy)
                follow_rectangle(image, cx, cy, cnt)
                cxx[i] = cx
                cyy[i] = cy
            count_cross_line.switch_color_line(line_bounds)
            cxx = [i for i in cxx if i]
            cyy = [i for i in cyy if i]
            add_min_x_i = []
            add_min_y_i = []
            current_cx_cy = []
            old_cx_cy = []

            if len(cxx):
                if not file_statistic.obj_id:
                    for i in range(len(cxx)):
                        file_statistic.obj_id.append(i)
                        file_statistic.df[str(file_statistic.obj_id[i])] = ''
                        file_statistic.df \
                            .at[int(file_statistic.frame_number), str(file_statistic.obj_id[i])] = [cxx[i], cyy[i]]
                        file_statistic.obj_total = file_statistic.obj_id[i] + 1
                else:
                    dx = np.zeros((len(cxx), len(file_statistic.obj_id)))
                    dy = np.zeros((len(cyy), len(file_statistic.obj_id)))

                    for i in range(len(cxx)):
                        for j in range(len(file_statistic.obj_id)):
                            for f in range(file_statistic.frame_number):
                                if file_statistic.frame_number - f in file_statistic.df.index:
                                    old_cx_cy = file_statistic.df \
                                        .loc[int(file_statistic.frame_number - f)][str(file_statistic.obj_id[j])]
                                    current_cx_cy = np.array([cxx[i], cyy[i]])
                                    if not isinstance(old_cx_cy, list) or len(old_cx_cy) != 2:
                                        continue
                                    else:
                                        dx[i, j] = old_cx_cy[0] - current_cx_cy[0]
                                        dy[i, j] = old_cx_cy[1] - current_cx_cy[1]
                                    break

                    for j in range(len(file_statistic.obj_id)):
                        sum_dx_dy = np.abs(dx[:, j]) + np.abs(dy[:, j])

                        min_sum_i = np.argmin(np.abs(sum_dx_dy))
                        min_dx = dx[min_sum_i, j]
                        min_dy = dy[min_sum_i, j]

                        if min_dx == 0 and min_dy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                            continue
                        else:
                            if np.abs(min_dx) < interface.max_rad and np.abs(min_dy) < interface.max_rad:
                                file_statistic.df \
                                    .at[int(file_statistic.frame_number),
                                        str(file_statistic.obj_id[j])] = [cxx[int(min_sum_i)], cyy[int(min_sum_i)]]
                                add_min_x_i.append(min_sum_i)
                                add_min_y_i.append(min_sum_i)

                    for i in range(len(cxx)):
                        if (i not in add_min_x_i and add_min_y_i) or \
                                (not old_cx_cy and not add_min_x_i and not add_min_y_i):
                            file_statistic.df[str(file_statistic.obj_total)] = ''
                            file_statistic.obj_id.append(file_statistic.obj_total)
                            file_statistic.df \
                                .at[int(file_statistic.frame_number),
                                    str(file_statistic.obj_total)] = [cxx[i], cyy[i]]
                            file_statistic.obj_total += 1

            current_objects = 0
            current_objects_index = []

            for i in range(len(file_statistic.obj_id)):
                if file_statistic.frame_number \
                        in file_statistic.df.index \
                        and file_statistic.df \
                        .at[int(file_statistic.frame_number), str(file_statistic.obj_id[i])] != '':
                    current_objects += 1
                    current_objects_index.append(i)
            for i in range(current_objects):
                current_center = file_statistic.df \
                    .loc[int(file_statistic.frame_number)][str(file_statistic.obj_id[current_objects_index[i]])]
                for f in range(file_statistic.frame_number):
                    if file_statistic.frame_number - f - 1 in file_statistic.df.index:
                        old_center = file_statistic.df \
                            .loc[
                                int(file_statistic.frame_number-f-1)
                            ][
                                str(file_statistic.obj_id[current_objects_index[i]])
                            ]

                        if isinstance(old_center, list) and len(old_center) == 2:
                            x_start = old_center[0] - interface.max_rad
                            y_start = old_center[1] - interface.max_rad
                            x_width = old_center[0] + interface.max_rad
                            y_height = old_center[1] + interface.max_rad
                            cv2.rectangle(
                                image,
                                (int(x_start), int(y_start)),
                                (int(x_width), int(y_height)),
                                (0, 125, 0),
                                1
                            )
                        break

                if isinstance(current_center, list):
                    cv2.putText(
                        image,
                        'Cd: ' + str(current_center[0]) + ', ' + str(current_center[1]),
                        (int(current_center[0]), int(current_center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .5,
                        (0, 255, 255),
                        1
                    )
                    cv2.putText(
                        image,
                        'ID: ' + str(file_statistic.obj_id[current_objects_index[i]]),
                        (int(current_center[0]), int(current_center[1] - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .5,
                        (0, 255, 255),
                        1
                    )
                    cv2.drawMarker(
                        image,
                        (int(current_center[0]), int(current_center[1])),
                        (0, 0, 255),
                        cv2.MARKER_STAR,
                        markerSize=5,
                        thickness=1,
                        line_type=cv2.LINE_AA
                    )

            field_t = np.zeros((700, 700, 3), np.uint8)

            cv2.rectangle(field_t, (0, 0), (700, 700), (255, 255, 255), -1)

            cv2.putText(field_t, '******************** Video options ***************************',
                        (0, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, "* Objects in area: " + str(current_objects),
                        (0, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, "* Total object detect: " + str(len(file_statistic.obj_id)),
                        (0, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, "* Frame: " + str(file_statistic.frame_number) + ' of ' + str(params['frames_count']),
                        (0, 80), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, '* Time: ' + str(round(file_statistic.frame_number / params['fps'], 2)) + ' sec of ' +
                        str(round(params['frames_count'] / params['fps'], 2)) + ' sec',
                        (0, 100), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, '**************************************************************',
                        (0, 120), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, '* Number of frames:                      ' + str(params['frames_count']),
                        (0, 140), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, '* FPS:                                    ' + str(params['fps']),
                        (0, 160), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, '* Extension:                         ' +
                        str(params['width']) + ' x ' + str(params['height']) + ' px',
                        (0, 180), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, '**************************************************************',
                        (0, 200), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, "* CURRENT LIFTS: {}".format(*count_cross_line.done_cross),
                        (0, 240), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, "* TOTAL LIFTS: " + str(count_cross_line.total),
                        (0, 260), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            cv2.putText(field_t, '**************************************************************', (0, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)

            image = cv2.flip(image, 1)
            image = cv2.resize(image, (1200, 800))

            cv2.imshow("contours", image)
            cv2.moveWindow("contours",
                           0,
                           0)

            # cv2.imshow("fgmask", foreground_mask)
            # cv2.moveWindow("fgmask",
            #                int(params['width'] * interface.ratio),
            #                0)
            #
            # cv2.imshow("closing", closing)
            # cv2.moveWindow("closing",
            #                0,
            #                int(params['height'] * interface.ratio))
            #
            # cv2.imshow("opening", opening)
            # cv2.moveWindow("opening",
            #                int(params['width'] * interface.ratio),
            #                int(params['height'] * interface.ratio))
            #
            # cv2.imshow("dilation", dilation)
            # cv2.moveWindow("dilation",
            #                0,
            #                2 * int(params['height'] * interface.ratio))
            #
            # cv2.imshow("binary", arr_bins)
            # cv2.moveWindow("binary",
            #                int(params['width'] * interface.ratio),
            #                2 * int(params['height'] * interface.ratio))
            #
            cv2.imshow("field text", field_t)
            cv2.moveWindow("field text",
                           4 * int(params['width'] * interface.ratio),
                           0)

            if cv2.waitKey(int(1000 / camera.get_param_camera()['fps'])) & 0xff == 27:  # 0xff <-> 255
                break

            try:
                frame = cv2.resize(src=image, dsize=(0, 0), fx=(1 / interface.ratio), fy=(1 / interface.ratio))
            except cv2.error as e:
                continue

            file_statistic.frame_number += 1

            video_statistic.write_record(frame)
            video_statistic.write_record(field_t)

        else:
            break

    camera.stop_record()
    video_statistic.stop_record()
    cv2.destroyAllWindows()

    file_statistic.save_data()

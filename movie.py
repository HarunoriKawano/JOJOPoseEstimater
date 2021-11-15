import cv2
import numpy as np
from matplotlib import pyplot as plt
import io
import tqdm
import opencv_functions as cvF


class MovieCreator:
    stand_information = [
        {'name': 'スティッキーフィンガーズ', 'top_x': 700, 'top_y': 70, 'person_top_x': 700, 'person_top_y': 70,
         'move_x': 620, 'move_y': -50, 'person_move_x': 500, 'person_move_y': 0}
    ]

    def __init__(self, video, person_image, detection_result, name, movie_time, maxparam=False):
        self.video = video
        self.person_image = person_image
        self.back_ground = cv2.imread(f'data/back_ground/{str(detection_result)}.jpg')
        self.stand = cv2.imread(f'data/stand/{str(detection_result)}.png')
        if self.stand.ndim == 3:
            mask = self.stand.copy()
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            index = np.where(np.logical_or(mask == 0, mask == 255))
            self.stand = cv2.cvtColor(self.stand, cv2.COLOR_RGB2RGBA)
            self.stand[index] = 0
        self.name = name
        self.movie_time = movie_time
        self.detection_result = detection_result
        if maxparam:
            self.param = np.array([5, 5, 5, 5, 5, 5])
        else:
            self.param = np.random.randint(1, 6, 6)

    def forward(self):
        iteration = 30 * self.movie_time
        stand_information = self.stand_information[self.detection_result]
        stand_plus_x = stand_information['move_x'] / iteration
        stand_plus_y = stand_information['move_y'] / iteration
        person_plus_x = stand_information['person_move_x'] / iteration
        person_plus_y = stand_information['person_move_y'] / iteration
        half_time = iteration / 2
        stand_title = '[STAND NAME]'
        person_title = '[STAND MASTER]'
        print('動画作成中...')
        for i in tqdm.tqdm(range(iteration)):
            stand_alpha = i / iteration if i / iteration <= 1.0 else 1.0
            person_alpha = i * 4 / iteration if i * 4 / iteration <= 1.0 else 1.0
            movie = cvF.image_synthesis(self.stand, self.back_ground,
                                        stand_information['top_x'] + stand_plus_x * i,
                                        stand_information['top_y'] + stand_plus_y * i,
                                        stand_alpha)
            movie = cvF.image_synthesis(self.person_image, movie,
                                        stand_information['person_top_x'] + person_plus_x * i,
                                        stand_information['person_top_y'] + person_plus_y * i,
                                        person_alpha)
            radar_chart = self.make_radar_chart(iteration, i, 750, '#0000ff')
            movie = cvF.image_synthesis(radar_chart, movie, 0, 330, 1)
            if i < half_time:
                stand_limit = round(len(stand_title) * i / half_time)
                stand_text = stand_title[:stand_limit]
                movie = cvF.cv2_putText(movie, stand_text, org=(100, 100), fontFace="C:/Windows/Fonts/msgothic.ttc",
                                        fontScale=50,
                                        color=(255, 0, 0))
                person_limit = round(len(person_title) * i / half_time)
                person_text = person_title[:person_limit]
                movie = cvF.cv2_putText(movie, person_text, org=(1000, 100), fontFace="C:/Windows/Fonts/msgothic.ttc",
                                        fontScale=50,
                                        color=(255, 0, 0))
            else:
                movie = cvF.cv2_putText(movie, stand_title, org=(100, 100), fontFace="C:/Windows/Fonts/msgothic.ttc",
                                        fontScale=50,
                                        color=(255, 0, 0))
                movie = cvF.cv2_putText(movie, person_title, org=(1000, 100), fontFace="C:/Windows/Fonts/msgothic.ttc",
                                        fontScale=50,
                                        color=(255, 0, 0))

            self.video.write(movie)
        return self.video, movie

    def make_radar_chart(self, iteration, i, size, color):
        param = self.param * i / iteration
        labels = ['破壊力', 'スピード', '射程距離', '持続力', '精密動作性', '成長性']
        radar_values = np.concatenate([param, [param[0]]])

        angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
        rgrids = list(range(8))
        str_rgrids = ['', 'E', 'D', 'C', 'B', 'A', '', '']

        fig = plt.figure(facecolor="w")
        ax = fig.add_subplot(1, 1, 1, polar=True, label='first')
        ax.plot(angles, radar_values, linewidth=0.2)
        ax.fill(angles, radar_values, color=color)
        ax.spines['polar'].set_color('gray')
        ax.spines['polar'].set_linewidth(5.0)
        ax.spines['polar'].set_zorder(1)
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontname="UD Digi Kyokasho N-B",
                          fontsize=20, color="black", zorder=2)
        ax.set_rgrids([])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        for grid_value in rgrids[:-2]:
            grid_values = [grid_value] * (len(labels) + 1)
            ax.plot(angles, grid_values, color="black", linewidth=1.5)

        for y, s in zip(rgrids, str_rgrids):
            ax.text(x=0, y=y, s=s, fontsize=12)

        for i, value in enumerate(param):
            text = str_rgrids[round(value)]
            ax.text(x=i * np.pi / 3, y=6, s=text, fontsize=25, fontname='Franklin Gothic Medium',
                    horizontalalignment="center", verticalalignment='center')

        ax.set_rlim([min(rgrids), max(rgrids)])
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        enc = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        dst = cv2.imdecode(enc, 1)
        plt.clf()
        plt.close()
        buf.close()
        if dst.ndim == 3:
            mask = dst.copy()
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            index = np.where(mask == 255)
            dst = cv2.cvtColor(dst, cv2.COLOR_RGB2RGBA)
            dst[index] = 0
        return cvF.scale_to_height(dst, height=size)

import cv2
import numpy as np
from matplotlib import pyplot as plt
import io
import tqdm
import opencv_functions as cvF
from PIL import Image, ImageDraw, ImageFont


class MovieCreator:
    stand_information = [
        {'name': 'スティッキーフィンガーズ', 'top_x': 1100, 'top_y': 20, 'person_top_x': 1050, 'person_top_y': 80,
         'move_x': 80, 'move_y': 0, 'person_move_x': -135, 'person_move_y': 10, 'person_scale_width': False,
         'person_scale': 990, 'figure_color': '#4682b4', 'text_color': (255, 255, 255), 'text_edge_color': (0, 255, 255),
         'stand_scale_width': False, 'stand_scale': 990},

        {'name': 'ザ・ワールド', 'top_x': 1020, 'top_y': 45, 'person_top_x': 885, 'person_top_y': 150,
         'move_x': 130, 'move_y': -45, 'person_move_x': -125, 'person_move_y': 35, 'person_scale_width': False,
         'person_scale': 895, 'figure_color': '#f4a460', 'text_color': (255, 255, 226), 'text_edge_color': (255, 255, 0),
         'stand_scale_width': False, 'stand_scale': 970},

        {'name': 'ゴールド・エクスペリエンス', 'top_x': 1280, 'top_y': 0, 'person_top_x': 855, 'person_top_y': 65,
         'move_x': -80, 'move_y': 0, 'person_move_x': 60, 'person_move_y': 5, 'person_scale_width': False,
         'person_scale': 990, 'figure_color': '#e7a118', 'text_color': (255, 255, 226), 'text_edge_color': (255, 255, 0),
         'stand_scale_width': False, 'stand_scale': 1040},

        {'name': 'ハイウェイ・スター', 'top_x': 1135, 'top_y': 105, 'person_top_x': 835, 'person_top_y': 50,
         'move_x': 0, 'move_y': 0, 'person_move_x': 0, 'person_move_y': 0, 'person_scale_width': False,
         'person_scale': 1020, 'figure_color': '#DA70D6', 'text_color': (181, 190, 251), 'text_edge_color': (138, 43, 226),
         'stand_scale_width': False, 'stand_scale': 914},

        {'name': 'クレイジー・ダイヤモンド', 'top_x': 800, 'top_y': 10, 'person_top_x': 915, 'person_top_y': 90,
         'move_x': 400, 'move_y': 0, 'person_move_x': -90, 'person_move_y': 5, 'person_scale_width': False,
         'person_scale': 980, 'figure_color': '#cc99cc', 'text_color': (204, 204, 255), 'text_edge_color': (204, 51, 255),
         'stand_scale_width': False, 'stand_scale': 1060},

        {'name': 'スタープラチナ', 'top_x': 735, 'top_y': 10, 'person_top_x': 1290, 'person_top_y': 120,
         'move_x': 0, 'move_y': 0, 'person_move_x': 0, 'person_move_y': 0, 'person_scale_width': False,
         'person_scale': 960, 'figure_color': '#9966cc', 'text_color': (221, 160, 221), 'text_edge_color': (199, 21, 133),
         'stand_scale_width': False, 'stand_scale': 900},

        {'name': 'ハイエロファント・グリーン', 'top_x': 800, 'top_y': 30, 'person_top_x': 1540, 'person_top_y': 80,
         'move_x': 85, 'move_y': 20, 'person_move_x': -195, 'person_move_y': 0, 'person_scale_width': False,
         'person_scale': 1000, 'figure_color': '#adff2f', 'text_color': (226, 255, 226), 'text_edge_color': (0, 250, 154),
         'stand_scale_width': False, 'stand_scale': 780},

        {'name': 'キラークイーン', 'top_x': 690, 'top_y': 60, 'person_top_x': 690, 'person_top_y': 150,
         'move_x': 450, 'move_y': -45, 'person_move_x': 205, 'person_move_y': 5, 'person_scale_width': False,
         'person_scale': 890, 'figure_color': '#cc99cc', 'text_color': (221, 160, 221), 'text_edge_color': (238, 130, 238),
         'stand_scale_width': False, 'stand_scale': 970},

        {'name': 'ヘブンズドア', 'top_x': 910, 'top_y': 70, 'person_top_x': 930, 'person_top_y': 30,
         'move_x': 420, 'move_y': -70, 'person_move_x': 120, 'person_move_y': 25, 'person_scale_width': False,
         'person_scale': 1025, 'figure_color': '#a4b5a3', 'text_color': (255, 255, 224), 'text_edge_color': (255, 255, 255),
         'stand_scale_width': False, 'stand_scale': 720},

        {'name': 'シルバー・チャリオッツ', 'top_x': 810, 'top_y': 30, 'person_top_x': 785, 'person_top_y': 40,
         'move_x': 475, 'move_y': -30, 'person_move_x': 245, 'person_move_y': 25, 'person_scale_width': False,
         'person_scale': 1015, 'figure_color': '#6086ac', 'text_color': (176, 196, 222), 'text_edge_color': (93, 163, 237),
         'stand_scale_width': False, 'stand_scale': 1050},

        {'name': 'スパイス・ガール', 'top_x': 930, 'top_y': 0, 'person_top_x': 830, 'person_top_y': 45,
         'move_x': 525, 'move_y': 30, 'person_move_x': 190, 'person_move_y': 95, 'person_scale_width': False,
         'person_scale': 940, 'figure_color': '#eeadcb', 'text_color': (255, 240, 245), 'text_edge_color': (255, 20, 147),
         'stand_scale_width': False, 'stand_scale': 1050},
    ]
    font_path = "data/yumindb.ttf"

    def __init__(self, video, person_image, detection_result, name, movie_time, maxparam=False):
        self.video = video
        self.back_ground = cv2.imread(f'data/back_ground/{str(detection_result)}.jpg')
        self.stand_image = cv2.imread(f'data/stand/{str(detection_result)}.png')
        if self.stand_image.ndim == 3:
            mask = self.stand_image.copy()
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            index = np.where(mask == 0)
            self.stand_image = cv2.cvtColor(self.stand_image, cv2.COLOR_RGB2RGBA)
            self.stand_image[index] = 0
        self.name = name
        self.stand = self.stand_information[detection_result]
        if self.stand['stand_scale_width']:
            self.stand_image = cvF.scale_to_width(self.stand_image, self.stand['stand_scale'])
        else:
            self.stand_image = cvF.scale_to_height(self.stand_image, self.stand['stand_scale'])
        self.movie_time = movie_time

        if self.stand['person_scale_width']:
            self.person_image = cvF.scale_to_width(person_image, self.stand['person_scale'])
        else:
            self.person_image = cvF.scale_to_height(person_image, self.stand['person_scale'])
        self.movie_time = movie_time
        if maxparam:
            self.param = np.array([5, 5, 5, 5, 5, 5])
        else:
            self.param = np.random.randint(1, 6, 6)

    def forward(self):
        iteration = round(30 * self.movie_time)
        stand_plus_x = self.stand['move_x'] / iteration
        stand_plus_y = self.stand['move_y'] / iteration
        person_plus_x = self.stand['person_move_x'] / iteration
        person_plus_y = self.stand['person_move_y'] / iteration
        half_time = iteration / 2
        stand_title = '[STAND NAME]'
        person_title = '[STAND MASTER]'
        print('動画作成中...')
        for i in tqdm.tqdm(range(iteration)):
            stand_alpha = i / iteration if i / iteration <= 1.0 else 1.0
            person_alpha = i * 2 / iteration if i * 2 / iteration <= 1.0 else 1.0
            movie = cvF.image_synthesis(self.stand_image, self.back_ground,
                                        self.stand['top_x'] + stand_plus_x * i,
                                        self.stand['top_y'] + stand_plus_y * i,
                                        stand_alpha)

            movie = cvF.image_synthesis(self.person_image, movie,
                                        self.stand['person_top_x'] + person_plus_x * i,
                                        self.stand['person_top_y'] + person_plus_y * i,
                                        person_alpha)

            radar_chart = self.make_radar_chart(iteration, i, 700, self.stand['figure_color'])
            movie = cvF.image_synthesis(radar_chart, movie, 0, 380, 1)

            if i < half_time:
                stand_limit = round(len(stand_title) * i / half_time)
                stand_text = stand_title[:stand_limit]
                movie = cvF.cv2_putText(stand_text, movie, org=(80, 70),
                                        font_path=self.font_path, font_size=50,
                                        color=self.stand['text_color'], edge_color=self.stand['text_edge_color'])

                person_limit = round(len(person_title) * i / half_time)
                person_text = person_title[:person_limit]
                movie = cvF.cv2_putText(person_text, movie, org=(1250, 750),
                                        font_path=self.font_path, font_size=50,
                                        color=self.stand['text_color'], edge_color=self.stand['text_edge_color'])
            else:
                movie = cvF.cv2_putText(stand_title, movie, org=(80, 70),
                                        font_path=self.font_path, font_size=50,
                                        color=self.stand['text_color'], edge_color=self.stand['text_edge_color'])

                movie = cvF.cv2_putText(person_title, movie, org=(1250, 750),
                                        font_path=self.font_path, font_size=50,
                                        color=self.stand['text_color'], edge_color=self.stand['text_edge_color'])

                stand_limit = round(len(stand_title) * (i - half_time + 1) / half_time)
                stand_text = self.stand['name'][:stand_limit]
                if iteration == i + 1:
                    stand_text = self.stand['name']
                movie = cvF.cv2_putText(stand_text, movie, org=(40, 170),
                                        font_path=self.font_path, font_size=80,
                                        color=self.stand['text_color'], edge_color=self.stand['text_edge_color'])

                image = Image.fromarray(movie)
                font = ImageFont.truetype(self.font_path, 80)
                draw_dummy = ImageDraw.Draw(image)
                w, h = draw_dummy.textsize(self.name, font)
                person_limit = round(len(self.name) * (i - half_time + 1) / half_time)
                person_text = self.name[:person_limit]
                if iteration == i + 1:
                    person_text = self.name
                movie = cvF.cv2_putText(person_text, movie, org=(1730 - w, 850),
                                        font_path=self.font_path, font_size=80,
                                        color=self.stand['text_color'], edge_color=self.stand['text_edge_color'])

            self.video.write(movie)
        return self.video, movie

    def make_radar_chart(self, iteration, i, size, color):
        param = self.param * i / iteration
        labels = ['破壊力', 'スピード', '射程距離', '持続力', '精密動作性', '成長性']
        radar_values = np.concatenate([param, [param[0]]])

        angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
        rgrids = list(range(8))
        str_rgrids = ['?', 'E', 'D', 'C', 'B', 'A', '', '']

        fig = plt.figure(facecolor="w", dpi=140)
        ax = fig.add_subplot(1, 1, 1, polar=True, label='first')
        ax.plot(angles, radar_values, linewidth=0.2)
        ax.fill(angles, radar_values, color=color)
        ax.spines['polar'].set_color('#708090')
        ax.spines['polar'].set_linewidth(5.0)
        ax.spines['polar'].set_zorder(1)
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontname="UD Digi Kyokasho N-B",
                          fontsize=20, color='black', zorder=2)
        # Hiragino Sans
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
            ax.text(x=i * np.pi / 3, y=6, s=text, fontsize=25, fontname='UD Digi Kyokasho N-B',
                    horizontalalignment="center", verticalalignment='center', color='black')
        # Hiragino Sans

        ax.set_rlim([min(rgrids), max(rgrids)])
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', transparent=True)
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
        return dst  # cvF.scale_to_height(dst, height=size)

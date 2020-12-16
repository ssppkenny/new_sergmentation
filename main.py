import cv2
import matplotlib.pyplot as plt
import numpy as np
import cyflann
import matplotlib.patches as patches

import heapq


def ratio(a, b):
    return min(a, b) / max(a, b)


def similar_sizes(h1, w1, h2, w2):
    return ratio(h1, h2) >= 0.6 and ratio(w1, w2) >= 0.1


class LineFinder(object):
    def __init__(self, filename):
        self.__filename = filename
        self.preprocess()

    def preprocess(self):
        self.__cy = cyflann.FLANNIndex(algorithm='kdtree_single')
        img = cv2.imread(self.__filename, 0)
        img = cv2.bitwise_not(img)

        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.__img = thresh
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        new_centroids = []
        self.__rects = []
        self.__center_to_rect = dict()
        self.__center_to_rect = dict()
        for i in range(1, len(stats)):
            h = stats[i][cv2.CC_STAT_HEIGHT]
            w = stats[i][cv2.CC_STAT_WIDTH]
            t = stats[i][cv2.CC_STAT_TOP]
            l = stats[i][cv2.CC_STAT_LEFT]
            if (h >= 6 or w >= 6):
                self.__rects.append((l, t, l + w, t + h))
                self.__center_to_rect[tuple(centroids[i])] = (l, t, l + w, t + h)
                new_centroids.append(centroids[i])

        self.__centroids = np.array(new_centroids)
        self.__cy.build_index(self.__centroids)

    def find_parallel_lines(self, a, b):
        l1, t1, r1, b1 = a
        l2, t2, r2, b2 = b

        top_points = [((l1 + r1) / 2, t1), ((l2 + r2) / 2, t2)]
        bottom_points = [((l1 + r1) / 2, b1), ((l2 + r2) / 2, b2)]

        sumxi = sum(map(lambda x: x[0], top_points))
        sumxk = sum(map(lambda x: x[0], bottom_points))

        sumyi = sum(map(lambda x: x[1], top_points))
        sumyk = sum(map(lambda x: x[1], bottom_points))

        sumxi2 = sum(map(lambda x: x[0] ** 2, top_points))
        sumxk2 = sum(map(lambda x: x[0] ** 2, bottom_points))

        sumyixi = sum(map(lambda x: x[0] * x[1], top_points))
        sumykxk = sum(map(lambda x: x[0] * x[1], bottom_points))

        N = 2

        delta = np.array([[sumxi2 + sumxk2, sumxi, sumxk], [sumxi, N, 0], [sumxk, 0, N]])

        deltam = np.array([[sumyixi + sumykxk, sumxi, sumxk], [sumyi, N, 0], [sumyk, 0, N]])

        deltabt = np.array([[sumxi2 + sumxk2, sumyixi + sumykxk, sumxk], [sumxi, sumyi, 0], [sumxk, sumyk, N]])

        deltabd = np.array([[sumxi2 + sumxk2, sumxi, sumyixi + sumykxk], [sumxi, N, sumyi], [sumxk, 0, sumyk]])

        bt = np.linalg.det(deltabt) / np.linalg.det(delta)
        bd = np.linalg.det(deltabd) / np.linalg.det(delta)

        m = np.linalg.det(deltam) / np.linalg.det(delta)

        return m, bt, bd

    def find_letter_left(self, current_block):
        centroids = self.__centroids
        left_letter, right_letter = current_block
        l1, t1, r1, b1 = left_letter
        m, bt, bd = self.find_parallel_lines(left_letter, right_letter)
        query = np.array([[(l1 + r1) / 2, (t1 + b1) / 2]])
        cy_ids, cy_dists = self.__cy.nn_index(query, 6)
        nnb = None
        for ind in cy_ids[0]:
            if (m * centroids[ind][0] + bt < centroids[ind][1] < m * centroids[ind][0] + bd) and centroids[ind][0] < l1:
                nnb = ind
                break
        return centroids[nnb] if not nnb is None else None

    def find_letter_right(self, current_block):
        centroids = self.__centroids
        left_letter, right_letter = current_block
        l1, t1, r1, b1 = right_letter
        m, bt, bd = self.find_parallel_lines(left_letter, right_letter)
        query = np.array([[(l1 + r1) / 2, (t1 + b1) / 2]])
        cy_ids, cy_dists = self.__cy.nn_index(query, 6)
        nnb = None
        for ind in cy_ids[0]:
            if (m * centroids[ind][0] + bt < centroids[ind][1] < m * centroids[ind][0] + bd) and centroids[ind][0] > r1:
                nnb = ind
                break
        return centroids[nnb] if not nnb is None else None

    def find_lines(self):
        fig, ax = plt.subplots(1)
        centroids = self.__centroids
        q = []
        max_prio = -float('inf')

        for i in range(0, len(centroids)):
            query = np.array([[centroids[i, 0], centroids[i, 1]]])
            current_rect = self.__rects[i]
            cy_ids, cy_dists = self.__cy.nn_index(query, 2)
            ind = list(filter(lambda x: x != i, cy_ids[0]))[0]
            # print(i, ind)
            rect = self.__rects[ind]
            current_rect = self.__rects[i]
            if similar_sizes(rect[0], rect[1], current_rect[0], current_rect[1]):
                prio = ratio(rect[0], current_rect[0]) + ratio(rect[1], current_rect[1])
                if prio > max_prio:
                    max_prio = prio
                heapq.heappush(q, (-prio, tuple(sorted((self.__rects[i], self.__rects[ind])))))

        visited = set()

        textlines = []

        while q:
            p, (a, b) = heapq.heappop(q)
            if a in visited or b in visited:
                continue

            textline = [a, b]

            visited.add(a)
            visited.add(b)

            ## form a line
            current_left = (a, b)
            current_right = (a, b)

            self.form_text_line(a, b, current_left, current_right, textline, visited)
            ax.imshow(self.__img)
            self.draw_text_line(ax, textline)
            textlines.append(textline)
        print(len(textlines))


    def draw_text_line(self, ax, textline):
        for letter in textline:
            x1, y1, x2, y2 = letter
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='w', facecolor='none')
            ax.add_patch(rect)

    def form_text_line(self, a, b, current_left, current_right, textline, visited):
        while True:
            ind1 = self.find_letter_left(current_left)
            if not ind1 is None:
                textline.insert(0, self.__center_to_rect[tuple(ind1)])
                visited.add(self.__center_to_rect[tuple(ind1)])
                current_left = (self.__center_to_rect[tuple(ind1)], a)
            else:
                break
        while True:
            ind1 = self.find_letter_right(current_right)
            if not ind1 is None:
                textline.append(self.__center_to_rect[tuple(ind1)])
                visited.add(self.__center_to_rect[tuple(ind1)])
                current_right = (b, self.__center_to_rect[tuple(ind1)])
            else:
                break


if __name__ == '__main__':
    lf = LineFinder('curved_lines.png')
    lf.find_lines()
    plt.show()

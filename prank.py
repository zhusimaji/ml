# -*- coding: utf-8 -*-
# @Time    : 2018/3/21 下午9:30
# @Author  : Tomcj
# @File    : prank.py
# @Software: PyCharm
import os
from tqdm import tqdm


class prank(object):
    '''
    prank算法，基于pointwise，测试数据使用微软测试数据
    '''
    rank_num = 10000
    rank_cate = 136
    rank_iter = 20
    rank_label = 5
    weight = []
    source_data = []
    br = [0, 0, 0, 0, float('inf')]

    def readFile(self, path):
        '''
        :param path:文件路径
        :return:
        '''
        test_rank=set()
        if not os.path.exists(path):
            raise AttributeError(u'请输入正确的样本路径')
        fp = open(path, 'r')
        for x in range(self.rank_num):
            tmp_list = []

            linedata = fp.readline().strip().split(" ")
            # 获取真实样本label
            tmp_list.append(int(linedata[0]))
            test_rank.add(int(linedata[0]))
            # 获取pid
            tmp_list.append(int((linedata[1].split(':'))[1]))
            # 获取特征维数
            for y in linedata[2:]:
                tmp_list.append(float(y.split(':')[1]))
            self.source_data.append(tmp_list)
        fp.close()
        print  test_rank

    def learn_to_rank(self):
        print 'start to learn rank'
        new_label = [0 for x in range(self.rank_label)]
        tao = []

        self.weight = [0.0 for x in range(self.rank_cate)]
        for num in range(self.rank_iter):
            for i in tqdm(range(self.rank_num)):
                predict_rank = 0
                sumwx = sum([self.weight[x] * self.source_data[i][x + 2]
                             for x in range(len(self.weight))])

                # 预测排名
                for r in range(self.rank_label):
                    if sumwx - self.br[r] < 0:
                        predict_rank = r
                        break
                # 获取真实label
                if self.source_data[i][0] != predict_rank:
                    for r in range(self.rank_label):
                        if self.source_data[i][0] - r < 0:
                            new_label[r] = -1
                        else:
                            new_label[r] = 1

                tao = [new_label[x] if (
                    sumwx - self.br[x]) * new_label[x] <= 0 else 0.0 for x in range(self.rank_label)]
                tao_sum = sum(tao)
                new_weight = [self.weight[x] + tao_sum * self.source_data[i][x+2]
                              for x in range(self.rank_cate)]
                self.weight = new_weight
                for  r in  range(self.rank_label):
                    self.br[r] = self.br[r] - tao[r]

    def predict_label(self):
        print 'start to predict'
        rightcount = 0
        for i in tqdm(range(self.rank_num)):
            predict_r = 0
            sumwx = sum([self.weight[x] * self.source_data[i][x]
                         for x in range(len(self.weight))])
            for j in range(self.rank_label):
                if sumwx < self.br[j]:
                    score = sumwx
                    predict_r = j
                    break
            if predict_r == self.source_data[i][0]:
                rightcount += 1

        print u'准确率为%{:.2f}'.format(1.0 * rightcount / self.rank_num * 100)


if __name__ == '__main__':
    pprank = prank()
    pprank.readFile('/Users/leiyang/Downloads/Fold1/test.txt')
    pprank.learn_to_rank()
    pprank.predict_label()

import numpy as np
import cv2


class Q2:
    def __init__(self, filename, subSampleR, width=352, height=288):
        # 构造函数，初始化对象
        # filename: 输入的YUV文件名
        # subSampleR: 子采样率，0->4:4:4, 1->4:2:2, 2->4:2:0
        # width: 帧宽度，默认为288
        # height: 帧高度，默认为352
        self.heightV = height
        self.widthV = width
        self.uv_height = height // 2
        self.uv_width = width // 2

        # 根据子采样率计算帧大小
        temp_d = {0: 3, 1: 2, 2: 1.5}
        self.frameSize = int(width * height * (temp_d[int(subSampleR)]))
        print("frames size: ", self.frameSize)

        # 从文件中读取YUV数据并存储在numpy数组中
        self.yuv_data = np.fromfile(filename, dtype=np.uint8, count=-1)
        print("total len: ", len(self.yuv_data))
        self.frameNum = len(self.yuv_data) // self.frameSize
        self.yFrame = []
        self.uFrame = []

        self.vFrame = []
        self.frameSpiliting()

    def frameSpiliting(self):
        # 分割帧的YUV数据
        for i in range(40):  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # print("this is the", str(i + 1) + "th frame")
            frame_start = i * (self.frameSize)

            # 读取Y分量
            y_data = self.yuv_data[frame_start:frame_start +
                                   self.heightV * self.widthV].reshape((self.heightV, self.widthV))
            self.yFrame.append(y_data)
            u_data = self.yuv_data[int(frame_start + self.heightV * self.widthV):int(
                frame_start + 1.25 * self.heightV * self.widthV)].reshape((self.uv_width, self.uv_height))
            self.uFrame.append(u_data)
            v_data = self.yuv_data[int(frame_start + 1.25 * self.heightV * self.widthV):int(
                frame_start + 1.5 * self.heightV * self.widthV)].reshape((self.uv_width, self.uv_height))
            self.vFrame.append(v_data)

        return

    def dispalyY(self):
        # 显示Y分量
        for i in range(self.frameNum):
            print("this is the", str(i + 1) + "th frame")
            frame_start = i * (self.frameSize)

            # 读取Y分量
            y_data = self.yuv_data[frame_start:frame_start +
                                   self.heightV * self.widthV].reshape((self.heightV, self.widthV))
            cv2.imshow('Y分量', y_data)
            if cv2.waitKey(80) & 0xFF == ord('q'):  # 按 'q' 键退出
                break
        cv2.destroyAllWindows()

        return

    def combineY(self):
        # 将Y分量组合成YUV格式的数据
        # my_array = np.array([89, 85, 86, 52, 77, 80, 69, 71, 50, 32, 87, 51, 53, 50, 32, 72, 50, 56, 56, 32,
        #                      70, 51, 48, 48, 48, 48, 58, 49, 48, 48, 49, 32, 73, 112, 32, 65, 49, 50, 56, 58, 49, 49, 55, 10])

        for i in range(self.frameNum):
            print("this is the", str(i + 1) + "th frame")
            frame_start = i * (self.frameSize)
            # 读取Y分量
            y_data = self.yuv_data[frame_start:frame_start +
                                   self.heightV * self.widthV]
            my_array = np.hstack((my_array, Frame))
            my_array = np.hstack((my_array, y_data))

        my_array = my_array.astype(np.uint8)
        return (my_array)
        return None

    def blockSpliting(self, blockSize):
        """
        _summary_

        Args:
            blockSize 分块操作中每一个block的大小，默认block的形状为正方形
        """

        blockSize = int(blockSize)
        self.blockSize = blockSize
        self.blockedYF = []

        # 分别计算出长宽方向分块的数量，以及长宽方向剩余的余数，需要填补的数量
        # 把横向和纵向的分块数量作为实例的一个属性，方便后续ME使用
        # 因为前面反过来了，这里用正确的称法，待后续组员修改__init__方法中的称呼
        self.blockNumInWidth = int(self.widthV // blockSize)
        self.blockNumInHeight = int(self.heightV // blockSize)

        remainderInWidth = self.widthV % blockSize
        remainderInHeight = self.heightV % blockSize
        paddingInWidth = blockSize - remainderInWidth
        paddingInHeight = blockSize - remainderInHeight

        # res用于存储每一帧分块后的Y数据,每一个元素是一个block，每一个block是一个二维数组，padding部分则可能为长方形的矩阵
        # 所有帧分块的结果存储在self.blockedYF中

        # todo: 以前5帧作为示例，后续进行修改
        for fr in range(5):
            # print("frame:", fr)
            frame = self.yFrame[fr]
            # 初始化当前帧分块以后的结果数组
            res = []

            # 如果需要水平向填补
            if remainderInWidth != 0:
                grayBlock = np.full(
                    (self.heightV, paddingInWidth), 128, np.uint8)
                frame = np.hstack((frame, grayBlock))
                # 更新横向分块数量
                self.blockNumInWidth += 1
            # 如果需要竖向填补
            if remainderInHeight != 0:
                # res.shape[1]这里获取了res的列数
                grayBlock = np.full(
                    (paddingInHeight, frame.shape[1]), 128, np.uint8)
                frame = np.vstack(frame, grayBlock)
                # 更新竖向分块数量
                self.blockNumInHeight += 1

            # 填补以后再分块
            for i in range(self.blockNumInHeight):
                res.append([])
            for i1 in range(self.blockNumInHeight):
                x_start = i1 * blockSize
                x_end = (i1 + 1) * blockSize
                for i2 in range(self.blockNumInWidth):
                    y_start = i2 * blockSize
                    y_end = (i2 + 1) * blockSize
                    block = frame[x_start:x_end, y_start:y_end]
                    res[i1].append(block)

            # 把这一帧的结果存储在blockedYF中
            self.blockedYF.append(res)

        return None

    def MAE(self, block_A=np.array([[]]), block_B=np.array([[]])):
        """
        计算均方误差

        Args:
            block_A (2D数组): 第一个块
            block_B (2D数组): 第二个块
        """
        absD = np.abs(block_A - block_B)
        mae = absD.mean()
        return mae

    def Get_ref_block(self, rx, ry):
        return

    def Full_search(self, currFnum, iRange):
        """
        _summary_

        Args:
            pre_f (2d array): refernce frame from self.blockedYF
            curr_f (2d array): blocked current frame from self.blockedYF
            iRange (odd number): i
        """
        curr_f = self.blockedYF[currFnum]
        pre_f = self.yFrame[currFnum - 1]
        searchR = iRange // 2  # 计算搜索范围
        print("search range", searchR)
        motion_V = []  # 初始化运动矢量列表
        for i in range(self.blockNumInHeight):
            motion_V.append([])
            for j in range(self.blockNumInWidth):
                min_mae = 12312312  # 初始化最小均方误差为一个较大的值
                min_axy = 12312312
                currentBlock = curr_f[i][j]
                tempMV = None
                for ry in range(searchR, -searchR - 1, -1):
                    for rx in range(searchR, -searchR - 1, -1):
                        ref_x = j * self.blockSize + rx
                        ref_y = i * self.blockSize + ry
                        # print("refxy: ", ref_x, ref_y)

                        # Ensure that ref_x and ref_y are within bounds
                        if (
                            0 <= ref_x + self.blockSize <= self.widthV and 0 <= ref_y + self.blockSize <= self.heightV) and (
                                0 <= ref_x <= self.widthV and 0 <= ref_y <= self.heightV):
                            refBlock = pre_f[ref_y:ref_y + self.blockSize,
                                             ref_x:ref_x + self.blockSize]
                            maeT = self.MAE(currentBlock, refBlock)
                            axy = np.abs(ry) + np.abs(rx)
                            # print("block:", (i, j), "MV:",
                            #       (rx, ry), "mae", maeT, "axy:", axy, "min_axy:", min_axy)
                            if (maeT <= min_mae) and (axy <= min_axy):
                                min_mae = maeT
                                min_axy = axy
                                tempMatch = refBlock
                                tempMV = (rx, ry)
                # print("MV", tempMV)
                motion_V[i].append(tempMV)
        print(motion_V)  # 打印运动矢量


if __name__ == "__main__":
    newO = Q2('foreman_cif-1.yuv', 2)
    # newO.dispalyY()
    newO.blockSpliting(8)
    print(newO.blockNumInHeight, newO.blockNumInWidth)
    newO.Full_search(1, 5)
    # newO.combineY()

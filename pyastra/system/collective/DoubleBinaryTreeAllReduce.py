# 头文件部分的模拟，Python 不需要像 C++ 那样的头文件包含机制，但需要导入模块
import sys

# 假设这些类和枚举在其他地方定义
class EventType:
    PacketReceived = 1
    General = 2

class CallData:
    pass

class BinaryTree:
    class Type:
        Leaf = 1
        Intermediate = 2
        Root = 3

    def __init__(self):
        pass

    def get_parent_id(self, id):
        pass

    def get_left_child_id(self, id):
        pass

    def get_right_child_id(self, id):
        pass

    def get_node_type(self, id):
        pass

    def is_enabled(self, id):
        pass

class Algorithm:
    def __init__(self, layer_num):
        self.layer_num = layer_num

class PacketBundle:
    def __init__(self, owner, stream, flag1, flag2, data_size, transmission_type):
        self.owner = owner
        self.stream = stream
        self.flag1 = flag1
        self.flag2 = flag2
        self.data_size = data_size
        self.transmission_type = transmission_type

    def send_to_MA(self):
        pass

    def send_to_NPU(self):
        pass

class RecvPacketEventHadndlerData:
    def __init__(self, stream, id, event_type, queue_id, stream_num):
        self.stream = stream
        self.id = id
        self.event_type = event_type
        self.queue_id = queue_id
        self.stream_num = stream_num

class Sys:
    dummy_data = None

    @staticmethod
    def handleEvent():
        pass


class DoubleBinaryTreeAllReduce(Algorithm):
    class State:
        Begin = 1
        WaitingForTwoChildData = 2
        WaitingForOneChildData = 3
        SendingDataToParent = 4
        WaitingDataFromParent = 5
        SendingDataToChilds = 6
        End = 7

    def __init__(self, id, layer_num, tree, data_size, boost_mode):
        super().__init__(layer_num)
        self.id = id
        self.logicalTopology = tree
        self.data_size = data_size
        self.state = self.State.Begin
        self.reductions = 0
        self.parent = tree.get_parent_id(id)
        self.left_child = tree.get_left_child_id(id)
        self.right_child = tree.get_right_child_id(id)
        self.type = tree.get_node_type(id)
        self.final_data_size = data_size
        self.comType = "All_Reduce"
        self.name = "DoubleBinaryTree"
        self.enabled = True
        if boost_mode:
            self.enabled = tree.is_enabled(id)

    def run(self, event, data):
        if self.state == self.State.Begin and self.type == BinaryTree.Type.Leaf:
            PacketBundle(self.stream.owner, self.stream, False, False, self.data_size, "Usual").send_to_MA()
            self.state = self.State.SendingDataToParent
            return
        elif self.state == self.State.SendingDataToParent and self.type == BinaryTree.Type.Leaf:
            snd_req = {
                "srcRank": self.stream.owner.id,
                "dstRank": self.parent,
                "tag": self.stream.stream_num,
                "reqType": "UINT8",
                "vnet": self.stream.current_queue_id,
                "layerNum": self.layer_num
            }
            # 模拟 front_end_sim_send
            # self.stream.owner.front_end_sim_send(0, Sys.dummy_data, self.data_size, "UINT8", self.parent, self.stream.stream_num, snd_req, Sys.handleEvent, None)

            rcv_req = {
                "vnet": self.stream.current_queue_id,
                "layerNum": self.layer_num
            }
            ehd = RecvPacketEventHadndlerData(self.stream, self.stream.owner.id, EventType.PacketReceived,
                                              self.stream.current_queue_id, self.stream.stream_num)
            # 模拟 front_end_sim_recv
            # self.stream.owner.front_end_sim_recv(0, Sys.dummy_data, self.data_size, "UINT8", self.parent, self.stream.stream_num, rcv_req, Sys.handleEvent, ehd)

            self.state = self.State.WaitingDataFromParent
            return
        elif self.state == self.State.WaitingDataFromParent and self.type == BinaryTree.Type.Leaf:
            PacketBundle(self.stream.owner, self.stream, False, False, self.data_size, "Usual").send_to_NPU()
            self.state = self.State.End
            return
        elif self.state == self.State.End and self.type == BinaryTree.Type.Leaf:
            sys.exit()
            return

        elif self.state == self.State.Begin and self.type == BinaryTree.Type.Intermediate:
            rcv_req = {
                "vnet": self.stream.current_queue_id,
                "layerNum": self.layer_num
            }
            ehd = RecvPacketEventHadndlerData(self.stream, self.stream.owner.id, EventType.PacketReceived,
                                              self.stream.current_queue_id, self.stream.stream_num)
            # 模拟 front_end_sim_recv
            # self.stream.owner.front_end_sim_recv(0, Sys.dummy_data, self.data_size, "UINT8", self.left_child, self.stream.stream_num, rcv_req, Sys.handleEvent, ehd)

            rcv_req2 = {
                "vnet": self.stream.current_queue_id,
                "layerNum": self.layer_num
            }
            ehd2 = RecvPacketEventHadndlerData(self.stream, self.stream.owner.id, EventType.PacketReceived,
                                               self.stream.current_queue_id, self.stream.stream_num)
            # 模拟 front_end_sim_recv
            # self.stream.owner.front_end_sim_recv(0, Sys.dummy_data, self.data_size, "UINT8", self.right_child, self.stream.stream_num, rcv_req2, Sys.handleEvent, ehd2)

            self.state = self.State.WaitingForTwoChildData
            return
        elif self.state == self.State.WaitingForTwoChildData and self.type == BinaryTree.Type.Intermediate and event == EventType.PacketReceived:
            PacketBundle(self.stream.owner, self.stream, True, False, self.data_size, "Usual").send_to_NPU()
            self.state = self.State.WaitingForOneChildData
            return
        elif self.state == self.State.WaitingForOneChildData and self.type == BinaryTree.Type.Intermediate and event == EventType.PacketReceived:
            PacketBundle(self.stream.owner, self.stream, True, True, self.data_size, "Usual").send_to_NPU()
            self.state = self.State.SendingDataToParent
            return
        elif self.reductions < 1 and self.type == BinaryTree.Type.Intermediate and event == EventType.General:
            self.reductions += 1
            return
        elif self.state == self.State.SendingDataToParent and self.type == BinaryTree.Type.Intermediate:
            snd_req = {
                "srcRank": self.stream.owner.id,
                "dstRank": self.parent,
                "tag": self.stream.stream_num,
                "reqType": "UINT8",
                "vnet": self.stream.current_queue_id,
                "layerNum": self.layer_num
            }
            # 模拟 front_end_sim_send
            # self.stream.owner.front_end_sim_send(0, Sys.dummy_data, self.data_size, "UINT8", self.parent, self.stream.stream_num, snd_req, Sys.handleEvent, None)

            rcv_req = {
                "vnet": self.stream.current_queue_id,
                "layerNum": self.layer_num
            }
            ehd = RecvPacketEventHadndlerData(self.stream, self.stream.owner.id, EventType.PacketReceived,
                                              self.stream.current_queue_id, self.stream.stream_num)
            # 模拟 front_end_sim_recv
            # self.stream.owner.front_end_sim_recv(0, Sys.dummy_data, self.data_size, "UINT8", self.parent, self.stream.stream_num, rcv_req, Sys.handleEvent, ehd)

            self.state = self.State.WaitingDataFromParent
        elif self.state == self.State.WaitingDataFromParent and self.type == BinaryTree.Type.Intermediate and event == EventType.PacketReceived:
            PacketBundle(self.stream.owner, self.stream, True, True, self.data_size, "Usual").send_to_NPU()
            self.state = self.State.SendingDataToChilds
            return
        elif self.state == self.State.SendingDataToChilds and self.type == BinaryTree.Type.Intermediate:
            snd_req = {
                "srcRank": self.stream.owner.id,
                "dstRank": self.left_child,
                "tag": self.stream.stream_num,
                "reqType": "UINT8",
                "vnet": self.stream.current_queue_id,
                "layerNum": self.layer_num
            }
            # 模拟 front_end_sim_send
            # self.stream.owner.front_end_sim_send(0, Sys.dummy_data, self.data_size, "UINT8", self.left_child, self.stream.stream_num, snd_req, Sys.handleEvent, None)

            snd_req2 = {
                "srcRank": self.stream.owner.id,
                "dstRank": self.right_child,
                "tag": self.stream.stream_num,
                "reqType": "UINT8",
                "vnet": self.stream.current_queue_id,
                "layerNum": self.layer_num
            }
            # 模拟 front_end_sim_send
            # self.stream.owner.front_end_sim_send(0, Sys.dummy_data, self.data_size, "UINT8", self.right_child, self.stream.stream_num, snd_req2, Sys.handleEvent, None)

            sys.exit()
            return

        elif self.state == self.State.Begin and self.type == BinaryTree.Type.Root:
            only_child_id = self.left_child if self.left_child >= 0 else self.right_child
            rcv_req = {
                "vnet": self.stream.current_queue_id,
                "layerNum": self.layer_num
            }
            ehd = RecvPacketEventHadndlerData(self.stream, self.stream.owner.id, EventType.PacketReceived,
                                              self.stream.current_queue_id, self.stream.stream_num)
            # 模拟 front_end_sim_recv
            # self.stream.owner.front_end_sim_recv(0, Sys.dummy_data, self.data_size, "UINT8", only_child_id, self.stream.stream_num, rcv_req, Sys.handleEvent, ehd)

            self.state = self.State.WaitingForOneChildData
        elif self.state == self.State.WaitingForOneChildData and self.type == BinaryTree.Type.Root:
            PacketBundle(self.stream.owner, self.stream, True, True, self.data_size, "Usual").send_to_NPU()
            self.state = self.State.SendingDataToChilds
            return
        elif self.state == self.State.SendingDataToChilds and self.type == BinaryTree.Type.Root:
            only_child_id = self.left_child if self.left_child >= 0 else self.right_child
            snd_req = {
                "srcRank": self.stream.owner.id,
                "dstRank": only_child_id,
                "tag": self.stream.stream_num,
                "reqType": "UINT8",
                "vnet": self.stream.current_queue_id,
                "layerNum": self.layer_num
            }
            # 模拟 front_end_sim_send
            # self.stream.owner.front_end_sim_send(0, Sys.dummy_data, self.data_size, "UINT8", only_child_id, self.stream.stream_num, snd_req, Sys.handleEvent, None)

            sys.exit()
            return

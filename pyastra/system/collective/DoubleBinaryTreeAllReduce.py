import sys

from system.topology.RingTopology import RingTopology
from AstraNetworkAPI import sim_request
from system.MockNcclChannel import *
from system.SendPacketEventHandlerData import SendPacketEventHandlerData
from PacketBundle import PacketBundle
from system.RecvPacketEventHandlerData import RecvPacketEventHandlerData
from Algorithm import Algorithm
from system.Common import ComType, EventType
from system.topology.BinaryTree import BinaryTree

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
            ehd = RecvPacketEventHandlerData(self.stream, self.stream.owner.id, EventType.PacketReceived,
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
            ehd = RecvPacketEventHandlerData(self.stream, self.stream.owner.id, EventType.PacketReceived,
                                              self.stream.current_queue_id, self.stream.stream_num)
            # 模拟 front_end_sim_recv
            # self.stream.owner.front_end_sim_recv(0, Sys.dummy_data, self.data_size, "UINT8", self.left_child, self.stream.stream_num, rcv_req, Sys.handleEvent, ehd)

            rcv_req2 = {
                "vnet": self.stream.current_queue_id,
                "layerNum": self.layer_num
            }
            ehd2 = RecvPacketEventHandlerData(self.stream, self.stream.owner.id, EventType.PacketReceived,
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
            ehd = RecvPacketEventHandlerData(self.stream, self.stream.owner.id, EventType.PacketReceived,
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
            ehd = RecvPacketEventHandlerData(self.stream, self.stream.owner.id, EventType.PacketReceived,
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

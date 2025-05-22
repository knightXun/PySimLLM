import enum
from typing import Tuple, Dict, Callable

from AstraNetworkAPI import AstraNetworkAPI
from AstraMemoryAPI import AstraMemoryAPI


class time_type_e(enum.Enum):
    NS = 0

class timespec_t:
    def __init__(self):
        self.time_res = None
        self.time_val = None



class WrapperData:
    class Type(enum.Enum):
        FastSendRecv = 1
        DetailedSend = 2
        DetailedRecv = 3
        Undefined = 4

    def __init__(self, type, msg_handler: Callable, fun_arg):
        self.type = type
        self.msg_handler = msg_handler
        self.fun_arg = fun_arg


class WrapperRelayData(WrapperData):
    def __init__(self, fast_backend, type, creation_time, partner_node, comm_size, msg_handler: Callable, fun_arg):
        super().__init__(type, msg_handler, fun_arg)
        self.partner_node = partner_node
        self.comm_size = comm_size
        self.creation_time = creation_time
        self.fast_backend = fast_backend


class InflightPairsMap:
    def __init__(self):
        self.inflightPairs = {}

    def insert(self, src, dest, tag, communicationSize, simulationType):
        key = (src, dest, tag, communicationSize)
        assert key not in self.inflightPairs
        self.inflightPairs[key] = simulationType

    def pop(self, src, dest, tag, communicationSize):
        key = (src, dest, tag, communicationSize)
        if key in self.inflightPairs:
            simulationType = self.inflightPairs[key]
            del self.inflightPairs[key]
            return (True, simulationType)
        return (False, WrapperData.Type.Undefined)

    def print(self):
        for key, value in self.inflightPairs.items():
            src, dest, tag, communicationSize = key
            print(f"src: {src}, dest: {dest}, tag: {tag}, communicationSize: {communicationSize}")


class DynamicLatencyTable:
    def __init__(self):
        self.latencyTables = {}
        self.latencyDataCountTable = {}

    def insertLatencyData(self, nodesPair, communicationSize, latency):
        if nodesPair not in self.latencyTables:
            self.insertNewTableForNode(nodesPair)
        latencyTable = self.latencyTables[nodesPair]
        if communicationSize not in latencyTable:
            latencyTable[communicationSize] = latency
            self.latencyDataCountTable[nodesPair] = self.latencyDataCountTable.get(nodesPair, 0) + 1

    def lookupLatency(self, nodesPair, communicationSize):
        if nodesPair in self.latencyTables:
            latencyTable = self.latencyTables[nodesPair]
            if communicationSize in latencyTable:
                return (True, latencyTable[communicationSize])
        return (False, -1)

    def predictLatency(self, nodesPair, communicationSize):
        assert self.canPredictLatency(nodesPair)
        assert not self.lookupLatency(nodesPair, communicationSize)[0]
        latencyTable = self.latencyTables[nodesPair]
        keys = sorted(latencyTable.keys())
        smallerPoint = None
        largerPoint = None
        for i, key in enumerate(keys):
            if key > communicationSize:
                if i == 0:
                    smallerPoint = keys[0]
                    largerPoint = keys[1]
                else:
                    smallerPoint = keys[i - 1]
                    largerPoint = key
                break
        if smallerPoint is None:
            smallerPoint = keys[-2]
            largerPoint = keys[-1]
        x1 = smallerPoint
        y1 = latencyTable[x1]
        x2 = largerPoint
        y2 = latencyTable[x2]
        slope = (y2 - y1) / (x2 - x1)
        predictedLatency = int(slope * (communicationSize - x1) + y1)
        assert predictedLatency > 0
        return predictedLatency

    def canPredictLatency(self, nodesPair):
        return self.latencyDataCountTable.get(nodesPair, 0) >= 2

    def print(self):
        for nodesPair, latencyTable in self.latencyTables.items():
            src, dest = nodesPair
            datapoints = self.latencyDataCountTable[nodesPair]
            print(f"src: {src}, dest: {dest}, datapoints: {datapoints}")
            for commSize, latency in latencyTable.items():
                print(f"\t- commSize: {commSize} - latency: {latency}")

    def insertNewTableForNode(self, nodesPair):
        assert nodesPair not in self.latencyTables
        assert nodesPair not in self.latencyDataCountTable
        self.latencyTables[nodesPair] = {}
        self.latencyDataCountTable[nodesPair] = 0


class FastBackEnd(AstraNetworkAPI):
    inflightPairsMap = InflightPairsMap()
    dynamicLatencyTable = DynamicLatencyTable()

    def __init__(self, rank, wrapped_backend):
        super().__init__(rank)
        self.wrapped_backend = wrapped_backend

    @staticmethod
    def handleEvent(arg):
        wrapperData = arg
        if wrapperData.type == WrapperData.Type.FastSendRecv:
            wrapperData.msg_handler(wrapperData.fun_arg)
        elif wrapperData.type == WrapperData.Type.DetailedRecv:
            wrapperRelayData = arg
            current_time = wrapperRelayData.fast_backend.wrapped_backend.sim_get_time()
            wrapperRelayData.fast_backend.update_table_recv(
                wrapperRelayData.creation_time,
                current_time.time_val,
                wrapperRelayData.partner_node,
                wrapperRelayData.comm_size
            )
            wrapperRelayData.msg_handler(wrapperRelayData.fun_arg)
        elif wrapperData.type == WrapperData.Type.DetailedSend:
            wrapperRelayData = arg
            current_time = wrapperRelayData.fast_backend.wrapped_backend.sim_get_time()
            wrapperRelayData.fast_backend.update_table_send(
                wrapperRelayData.creation_time,
                current_time.time_val,
                wrapperRelayData.partner_node,
                wrapperRelayData.comm_size
            )
            wrapperRelayData.msg_handler(wrapperRelayData.fun_arg)
        else:
            print("Event type undefined!")

    def sim_time_resolution(self):
        return self.wrapped_backend.sim_time_resolution()

    def sim_finish(self):
        self.wrapped_backend.sim_finish()
        return 1

    def sim_comm_size(self, comm, size):
        return self.wrapped_backend.sim_comm_size(comm, size)

    def sim_get_time(self):
        return self.wrapped_backend.sim_get_time()

    def sim_init(self, MEM):
        return self.wrapped_backend.sim_init(MEM)

    def sim_schedule(self, delta, fun_ptr, fun_arg):
        self.wrapped_backend.sim_schedule(delta, fun_ptr, fun_arg)

    def update_table_send(self, start, finished, dst, comm_size):
        latency = finished - start
        src = self.rank
        self.dynamicLatencyTable.insertLatencyData((src, dst), comm_size, latency)

    def update_table_recv(self, start, finished, src, comm_size):
        latency = finished - start
        dst = self.rank
        self.dynamicLatencyTable.insertLatencyData((src, dst), comm_size, latency)

    def relay_recv_request(self, buffer, count, type, src, tag, request, msg_handler, fun_arg):
        current_time = self.wrapped_backend.sim_get_time()
        wrapperData = WrapperRelayData(
            self,
            WrapperData.Type.DetailedRecv,
            current_time.time_val,
            src,
            count,
            msg_handler,
            fun_arg
        )
        return self.wrapped_backend.sim_recv(
            buffer,
            count,
            type,
            src,
            tag,
            request,
            self.handleEvent,
            wrapperData
        )

    def relay_send_request(self, buffer, count, type, dst, tag, request, msg_handler, fun_arg):
        current_time = self.wrapped_backend.sim_get_time()
        wrapperData = WrapperRelayData(
            self,
            WrapperData.Type.DetailedSend,
            current_time.time_val,
            dst,
            count,
            msg_handler,
            fun_arg
        )
        return self.wrapped_backend.sim_send(
            buffer,
            count,
            type,
            dst,
            tag,
            request,
            self.handleEvent,
            wrapperData
        )

    def fast_send_recv_request(self, delay, msg_handler, fun_arg):
        delta = timespec_t()
        delta.time_res = time_type_e.NS
        delta.time_val = delay
        wrapperData = WrapperData(WrapperData.Type.FastSendRecv, msg_handler, fun_arg)
        self.wrapped_backend.sim_schedule(delta, self.handleEvent, wrapperData)
        return 1

    def sim_send(self, buffer, count, type, dst, tag, request, msg_handler, fun_arg):
        src = self.rank
        srcDestPair = (src, dst)
        inflightPair = self.inflightPairsMap.pop(src, dst, tag, count)
        if inflightPair[0]:
            if inflightPair[1] == WrapperData.Type.FastSendRecv:
                lookupResult = self.dynamicLatencyTable.lookupLatency(srcDestPair, count)
                if lookupResult[0]:
                    return self.fast_send_recv_request(lookupResult[1], msg_handler, fun_arg)
                predictedLatency = self.dynamicLatencyTable.predictLatency(srcDestPair, count)
                return self.fast_send_recv_request(predictedLatency, msg_handler, fun_arg)
            elif inflightPair[1] == WrapperData.Type.DetailedRecv:
                return self.relay_send_request(buffer, count, type, dst, tag, request, msg_handler, fun_arg)
            else:
                print("sim_send inflight pair error")
                exit(-1)
        lookupResult = self.dynamicLatencyTable.lookupLatency(srcDestPair, count)
        if lookupResult[0]:
            self.inflightPairsMap.insert(src, dst, tag, count, WrapperData.Type.FastSendRecv)
            return self.fast_send_recv_request(lookupResult[1], msg_handler, fun_arg)
        if self.dynamicLatencyTable.canPredictLatency(srcDestPair):
            import random
            if random.randint(0, 99) < 10:
                self.inflightPairsMap.insert(src, dst, tag, count, WrapperData.Type.DetailedSend)
                return self.relay_send_request(buffer, count, type, dst, tag, request, msg_handler, fun_arg)
            self.inflightPairsMap.insert(src, dst, tag, count, WrapperData.Type.FastSendRecv)
            predictedLatency = self.dynamicLatencyTable.predictLatency(srcDestPair, count)
            return self.fast_send_recv_request(predictedLatency, msg_handler, fun_arg)
        self.inflightPairsMap.insert(src, dst, tag, count, WrapperData.Type.DetailedSend)
        return self.relay_send_request(buffer, count, type, dst, tag, request, msg_handler, fun_arg)

    def sim_recv(self, buffer, count, type, src, tag, request, msg_handler, fun_arg):
        dst = self.rank
        srcDestPair = (src, dst)
        inflightPair = self.inflightPairsMap.pop(src, dst, tag, count)
        if inflightPair[0]:
            if inflightPair[1] == WrapperData.Type.FastSendRecv:
                lookupResult = self.dynamicLatencyTable.lookupLatency(srcDestPair, count)
                if lookupResult[0]:
                    return self.fast_send_recv_request(lookupResult[1], msg_handler, fun_arg)
                predictedLatency = self.dynamicLatencyTable.predictLatency(srcDestPair, count)
                return self.fast_send_recv_request(predictedLatency, msg_handler, fun_arg)
            elif inflightPair[1] == WrapperData.Type.DetailedSend:
                return self.relay_recv_request(buffer, count, type, src, tag, request, msg_handler, fun_arg)
            else:
                print("sim_recv inflight pair error")
                exit(-1)
        lookupResult = self.dynamicLatencyTable.lookupLatency(srcDestPair, count)
        if lookupResult[0]:
            self.inflightPairsMap.insert(src, dst, tag, count, WrapperData.Type.FastSendRecv)
            return self.fast_send_recv_request(lookupResult[1], msg_handler, fun_arg)
        if self.dynamicLatencyTable.canPredictLatency(srcDestPair):
            import random
            if random.randint(0, 99) < 10:
                self.inflightPairsMap.insert(src, dst, tag, count, WrapperData.Type.DetailedRecv)
                return self.relay_recv_request(buffer, count, type, src, tag, request, msg_handler, fun_arg)
            self.inflightPairsMap.insert(src, dst, tag, count, WrapperData.Type.FastSendRecv)
            predictedLatency = self.dynamicLatencyTable.predictLatency(srcDestPair, count)
            return self.fast_send_recv_request(predictedLatency, msg_handler, fun_arg)
        self.inflightPairsMap.insert(src, dst, tag, count, WrapperData.Type.DetailedRecv)
        return self.relay_recv_request(buffer, count, type, src, tag, request, msg_handler, fun_arg)

    
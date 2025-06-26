import math
import time
import enum 

from ns import ns
from system import Common

class RingFlowModel:
    def __init__(self, 
                nodes,
                nvswitchs,
                ns3_nodes,
                portNumber,
                pairBdp,
                has_win,
                global_t,
                pairRtt,
                maxRtt,
                serverAddress,
                maxBdp
                ):
        self.nodes = nodes
        self.nvswitchs = nvswitchs
        self.ns3_nodes = ns3_nodes
        self.portNumber = portNumber
        self.pairBdp = pairBdp
        self.has_win = has_win
        self.global_t = global_t
        self.pairRtt = pairRtt
        self.maxRtt = maxRtt
        self.serverAddress = serverAddress
        self.maxBdp = maxBdp


    def p2pSend(self, src, dst, packetSize, send_lat=3000):
        nvls_on = True
        # import pdb; pdb.set_trace()
        port = self.portNumber[src][dst] 
        self.portNumber[src][dst]  = self.portNumber[src][dst]  + 1
        pg = 3
        dport = 100
        # send_lat = 6000
        # send_lat = 3 * 1000 
        # import pdb; pdb.set_trace()

        win = 0
        if self.ns3_nodes.Get(src) not in self.pairBdp or \
            self.ns3_nodes.Get(dst) not in self.pairBdp[self.ns3_nodes.Get(src)]:
            if not self.has_win:
                win = 0
            elif self.global_t == 1:
                win = self.maxBdp
            else:
                win = 0
        else:
            if not self.has_win:
                win = 0
            elif self.global_t == 1:
                win = self.maxBdp
            else:
                win = self.pairBdp[ self.ns3_nodes.Get(src) ][ self.ns3_nodes.Get(dst) ] 

        rtt = 0
        if src in self.pairRtt and dst in self.pairRtt[src]:
            rtt = self.maxRtt if self.global_t == 1 else self.pairRtt[src][dst]
        else:
            rtt = self.maxRtt if self.global_t == 1 else 0

        if packetSize == 0:
            packetSize = 1
        
        clientHelper = ns.RdmaClientHelper(
            pg, 
            self.serverAddress[src], 
            self.serverAddress[dst], 
            int(port), 
            dport,
            int(packetSize), 
            win,
            rtt,
            0, 
            src,
            dst
        )

        # clientHelper.SetAttribute("NVLS_enable", ns.UintegerValue(1))
        clientHelper.SetAttribute("NVLS_enable",  ns.UintegerValue(1))

        appCon = clientHelper.Install(self.ns3_nodes.Get(src));
        
        appCon.Start(ns.NanoSeconds(send_lat));
        
    def runReduceScatter(self, data_size):
        single_batch = data_size / 8
        startT = ns.Simulator.Now().GetNanoSeconds()
        #TODO: compute this npu_time
        to_npu_time = 0

        for j in range( len(self.nodes) / 8 ):
            arr = list(range(j * 8, j*8 + 8 ) )
            for a in arr:
                for b in arr:
                    if a != b: 
                        self.p2pSend( self.nodes[a], self.nodes[b], single_batch)

            ns.Simulator.Run()

        endT = ns.Simulator.Now().GetNanoSeconds()
        return endT - startT + to_npu_time


    def runAllGather(self, data_size):
        single_batch = data_size / (8 * 8)

        startT = ns.Simulator.Now().GetNanoSeconds()
        #TODO: compute this npu_time
        to_npu_time = 0

        for t in range(7):
            for j in self.nodes:
                send_node = self.nodes[j]
                recv_node = self.nodes[j]
                if j % 8 == 0:
                    recv_node = self.nodes[j + 7]
                else:
                    recv_node = self.nodes[j - 1]

                for i in range(8):
                    self.p2pSend( send_node, recv_node, single_batch)
            
            ns.Simulator.Run()

        endT = ns.Simulator.Now().GetNanoSeconds()

        return endT - startT + to_npu_time  


    def runAllReduce(self, data_size):
        single_batch = data_size / (8 * 8)

        startT = ns.Simulator.Now().GetNanoSeconds()
        #TODO: compute this npu_time
        to_npu_time = 0

        for t in range(14):
            for j in self.nodes:
                send_node = self.nodes[j]
                recv_node = self.nodes[j]
                if j % 8 == 0:
                    recv_node = self.nodes[j + 7]
                else:
                    recv_node = self.nodes[j - 1]

                for i in range(8):
                    self.p2pSend( send_node, recv_node, single_batch)
            
            ns.Simulator.Run()


        endT = ns.Simulator.Now().GetNanoSeconds()
        
        return endT - startT + to_npu_time  


    def runAll2All(self, data_size): 
        single_batch = data_size / 8
        startT = ns.Simulator.Now().GetNanoSeconds()
        #TODO: compute this npu_time
        to_npu_time = 0

        for j in range( len(self.nodes) / 8 ):
            arr = list(range(j * 8, j*8 + 8 ) )
            for a in arr:
                for b in arr:
                    if a != b: 
                        self.p2pSend( self.nodes[a], self.nodes[b], single_batch)

            ns.Simulator.Run()

        endT = ns.Simulator.Now().GetNanoSeconds()
        return endT - startT + to_npu_time

    def runAllReduceAll2All(self, data_size):
        return 0  

    def runAllReduceNVLS(self, data_size):
        return 0 


class NcclTreeFlowModel:
    def __init__(self, 
                nodes,
                nvswitchs,
                ns3_nodes,
                portNumber,
                pairBdp,
                has_win,
                global_t,
                pairRtt,
                maxRtt,
                serverAddress,
                maxBdp
                ):
        self.nodes = nodes
        self.nvswitchs = nvswitchs
        self.ns3_nodes = ns3_nodes
        self.portNumber = portNumber
        self.pairBdp = pairBdp
        self.has_win = has_win
        self.global_t = global_t
        self.pairRtt = pairRtt
        self.maxRtt = maxRtt
        self.serverAddress = serverAddress
        self.maxBdp = maxBdp


    def p2pSend(self, src, dst, packetSize, send_lat=3000):
        nvls_on = True
        # import pdb; pdb.set_trace()
        port = self.portNumber[src][dst] 
        self.portNumber[src][dst]  = self.portNumber[src][dst]  + 1
        pg = 3
        dport = 100
        # send_lat = 6000
        # send_lat = 3 * 1000 

        # import pdb; pdb.set_trace()

        win = 0
        if self.ns3_nodes.Get(src) not in self.pairBdp or \
            self.ns3_nodes.Get(dst) not in self.pairBdp[self.ns3_nodes.Get(src)]:
            if not self.has_win:
                win = 0
            elif self.global_t == 1:
                win = self.maxBdp
            else:
                win = 0
        else:
            if not self.has_win:
                win = 0
            elif self.global_t == 1:
                win = self.maxBdp
            else:
                win = self.pairBdp[ self.ns3_nodes.Get(src) ][ self.ns3_nodes.Get(dst) ] 

        rtt = 0
        if src in self.pairRtt and dst in self.pairRtt[src]:
            rtt = self.maxRtt if self.global_t == 1 else self.pairRtt[src][dst]
        else:
            rtt = self.maxRtt if self.global_t == 1 else 0

        if packetSize == 0:
            packetSize = 1
        
        clientHelper = ns.RdmaClientHelper(
            pg, 
            self.serverAddress[src], 
            self.serverAddress[dst], 
            port, 
            dport,
            int(packetSize), 
            win,
            rtt,
            0, 
            src,
            dst
        )

        clientHelper.SetAttribute("NVLS_enable", ns.UintegerValue(1))
        
        appCon = clientHelper.Install(self.ns3_nodes.Get(src));
        
        appCon.Start(ns.NanoSeconds(send_lat));

    def runReduceScatter(self, data_size):
        single_batch = data_size / 8
        startT = ns.Simulator.Now().GetNanoSeconds()
        #TODO: compute this npu_time
        to_npu_time = 0

        for j in range( len(self.nodes) / 8 ):
            arr = list(range(j * 8, j*8 + 8 ) )
            for a in arr:
                for b in arr:
                    if a != b: 
                        self.p2pSend( self.nodes[a], self.nodes[b], single_batch)

            ns.Simulator.Run()

        endT = ns.Simulator.Now().GetNanoSeconds()
        return endT - startT + to_npu_time


    def runAllGather(self, data_size):
        single_batch = data_size / (8 * 8)

        startT = ns.Simulator.Now().GetNanoSeconds()
        #TODO: compute this npu_time
        to_npu_time = 0

        for t in range(7):
            for j in self.nodes:
                send_node = self.nodes[j]
                recv_node = self.nodes[j]
                if j % 8 == 0:
                    recv_node = self.nodes[j + 7]
                else:
                    recv_node = self.nodes[j - 1]

                for i in range(8):
                    self.p2pSend( send_node, recv_node, single_batch)
            
            ns.Simulator.Run()

        endT = ns.Simulator.Now().GetNanoSeconds()
        return endT - startT + to_npu_time  

    def runAllReduce(self, data_size):
        single_batch = data_size / 4

        startT = ns.Simulator.Now().GetNanoSeconds()
        #TODO: compute this npu_time
        to_npu_time = 0

        for t in range(4):
            for node in self.nodes:
                self.p2pSend( node, self.nvswitchs[0], single_batch)
            for node in self.nodes:
                self.p2pSend( self.nvswitchs[0], node, single_batch)
            
            ns.Simulator.Run()


        endT = ns.Simulator.Now().GetNanoSeconds()
        return endT - startT + to_npu_time  



    def runAll2All(self, data_size): 
        single_batch = data_size / 8

        startT = ns.Simulator.Now().GetNanoSeconds()

        #TODO: compute this npu_time
        to_npu_time = 0

        for j in range( len(self.nodes) / 8 ):
            arr = list(range(j * 8, j*8 + 8 ) )
            for a in arr:
                for b in arr:
                    if a != b: 
                        self.p2pSend( self.nodes[a], self.nodes[b], single_batch)

            ns.Simulator.Run()

        endT = ns.Simulator.Now().GetNanoSeconds()
        return endT - startT + to_npu_time

    def runAllReduceAll2All(self, data_size):
        return 0 

    def runAllReduceNVLS(self, data_size):
        return 0 


class FlowModel:

    def __init__(self, 
                nodes,
                nvswitchs,
                ns3_nodes,
                portNumber,
                pairBdp,
                has_win,
                global_t,
                pairRtt,
                maxRtt,
                serverAddress,
                maxBdp
                ):
        if len(nvswitchs) == 1:
            self.flow_model = NcclTreeFlowModel(nodes, \
                nvswitchs, ns3_nodes, portNumber, pairBdp, \
                has_win, global_t, pairRtt, maxRtt, serverAddress, maxBdp)
        else:
            # self.flow_model = RingFlowModel(nodes, nvswitchs, ns3_nodes) 
            self.flow_model = RingFlowModel(nodes, \
                nvswitchs, ns3_nodes, portNumber, pairBdp, \
                has_win, global_t, pairRtt, maxRtt, serverAddress, maxBdp)


    def runReduceScatter(self, data_size):
        return self.flow_model.runReduceScatter(data_size)


    def runAllGather(self, data_size):
        return self.flow_model.runAllGather(data_size)


    def runAllReduce(self, data_size):
        return self.flow_model.runAllReduce(data_size)


    def runAll2All(self, data_size): 
        return self.flow_model.runAll2All(data_size)


    def runAllReduceAll2All(self, data_size):
        return self.flow_model.runAllReduceAll2All(data_size)


    def runAllReduceNVLS(self, data_size):
        return self.flow_model.runAllReduceNVLS(data_size)

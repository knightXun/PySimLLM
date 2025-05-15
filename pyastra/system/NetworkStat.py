class NetworkStat:
    def __init__(self):
        self.net_message_latency = []
        self.net_message_counter = 0

    def update_network_stat(self, network_stat):
        if len(self.net_message_latency) < len(network_stat.net_message_latency):
            diff = len(network_stat.net_message_latency) - len(self.net_message_latency)
            self.net_message_latency.extend([0.0] * diff)
        
        for i, ml in enumerate(network_stat.net_message_latency):
            self.net_message_latency[i] += ml
        
        self.net_message_counter += 1

    def take_network_stat_average(self):
        if self.net_message_counter == 0:
            return 
        
        for i in range(len(self.net_message_latency)):
            self.net_message_latency[i] /= self.net_message_counter
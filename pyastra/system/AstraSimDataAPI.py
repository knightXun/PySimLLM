class LayerData:
    def __init__(self):
        self.layer_name = ""
        self.total_forward_pass_compute = 0.0
        self.total_weight_grad_compute = 0.0
        self.total_input_grad_compute = 0.0
        self.total_waiting_for_fwd_comm = 0.0
        self.total_waiting_for_wg_comm = 0.0
        self.total_waiting_for_ig_comm = 0.0
        self.total_fwd_comm = 0.0
        self.total_weight_grad_comm = 0.0
        self.total_input_grad_comm = 0.0
        self.avg_queuing_delay = []
        self.avg_network_message_dealy = []


class AstraSimDataAPI:
    def __init__(self):
        self.run_name = ""
        self.layers_stats = []
        self.avg_chunk_latency_per_logical_dimension = []
        self.workload_finished_time = 0.0
        self.total_compute = 0.0
        self.total_exposed_comm = 0.0
    
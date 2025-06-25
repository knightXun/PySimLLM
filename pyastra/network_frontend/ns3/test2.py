from ns import ns  

def get_pfc(fout, types):
    """Python版本PFC数据记录函数"""
    # 获取模拟时间步
    time_step = ns.Simulator.Now().GetTimeStep()
    
    # # 从设备获取节点信息
    # node = dev.GetNode()
    # node_id = node.GetId()
    # node_type = node.GetNodeType()  # 假设节点有此方法
    # if_index = dev.GetIfIndex()
    
    # 写入格式化数据到文件
    # fout.write(f"{time_step} {node_id} {node_type} {if_index} {type}\n")
    fout.write(f"{time_step} {types}\n")


with open("pfc.txt", "w") as fout:
    # 创建回调函数，绑定fout和type=2
    callback = ns.MakeBoundCallback(get_pfc, fout)
        
        # 示例：将回调关联到设备事件（需替换为实际设备）
        # dev = ...  # 获取QbbNetDevice实例
        # dev.TraceConnectWithoutContext("PfcEvent", callback)
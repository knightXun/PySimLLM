from ns import ns

n = ns.NodeContainer()
nodes = [] 
for i in range(10): 
  if i %3 ==0:
    a = ns.CreateObject[ns.Node]()
    n.Add(a)
  elif i % 3 == 1:
    sw = ns.CreateObject[ns.SwitchNode]()
    n.Add(sw)
  else: 
    sw = ns.CreateObject[ns.NVSwitchNode]()
    n.Add(sw)

internet = ns.InternetStackHelper()
internet.Install(n)

# from ns import ns
# node_num = 10
# n = ns.NodeContainer(node_num)


# internet = ns.InternetStackHelper()
# internet.Install(n)
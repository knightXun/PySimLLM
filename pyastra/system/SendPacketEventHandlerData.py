from enum import Enum

from BaseStream import * 
from BasicEventHandlerData import * 
from Common import * 
from Sys import *


class SendPacketEventHandlerData(BasicEventHandlerData, MetaData):
  def __init__(self, *args, **kwargs):
      if len(args) == 4 and all(isinstance(arg, int) for arg in args[1:]):
          node, senderNodeId, receiverNodeId, tag = args
          super().__init__(node, EventType.PacketSent)
          self.init_first_constructor(senderNodeId, receiverNodeId, tag)
      elif (len(args) == 5 and 
            isinstance(args[0], object) and 
            all(isinstance(arg, int) for arg in args[1:4]) and
            isinstance(args[4], EventType)):
          owner, senderNodeId, receiverNodeId, tag, event = args
          super().__init__(owner.owner, event)
          self.init_second_constructor(owner, senderNodeId, receiverNodeId, tag)
      else:
          raise ValueError("Invalid arguments for SendPacketEventHandlerData constructor")

  def init_first_constructor(self, senderNodeId, receiverNodeId, tag):
      self.senderNodeId = senderNodeId
      self.receiverNodeId = receiverNodeId
      self.tag = tag
      self.child_flow_id = -2

  def init_second_constructor(self, owner, senderNodeId, receiverNodeId, tag):
      self.owner = owner
      self.senderNodeId = senderNodeId
      self.receiverNodeId = receiverNodeId
      self.tag = tag
      self.channel_id = -1
      self.child_flow_id = -1
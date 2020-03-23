import gobang
import playerAI
import numpy as np
import torch

dev = torch.device("cpu")
size = (15, 15)

game = gobang.Game()

p1 = game.getPlayer1()
p2 = game.getPlayer2()

ai1 = playerAI.AI(p1)
ai2 = playerAI.AI(p2)

ai1.findLocation()
ai2.findLocation()
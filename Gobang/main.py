import gobang
import playerAI
import numpy as np
import torch

if __name__ == "__main__":
    dev = torch.device("cpu")
    size = (15, 15)

    game = gobang.Game()

    p1 = game.getPlayer1()
    p2 = game.getPlayer2()

    ai1 = playerAI.AI(p1)
    ai2 = playerAI.AI(p2)

    while (True):
        ptr = ai1.findLocation()
        (isWin, socre) = game.nextTurn(ai1.player, ptr)
        game.drawBoard()
        input("{}: {}".format(ptr, isWin))

        ptr = ai2.findLocation()
        (isWin, socre) = game.nextTurn(ai2.player, ptr)
        game.drawBoard()
        input("{}: {}".format(ptr, isWin))
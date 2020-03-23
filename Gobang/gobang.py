import numpy as np

class Board:
    def __init__(self, size:(int, int)):
        self.size = size
        self.board = np.zeros(size, dtype=np.uint8)
    
    def clear(self):
        self.board = np.zeros(size, dtype=np.uint8)
    
    def setChess(self, location:(int, int), chessType:(int)) -> bool:
        if (location[0] > self.size[0] or location[1] > self.size[1]):
            return False
        if (self.board[location] != 0):
            return False
        self.board[location] = chessType
        return True
    
    def draw(self):
        (y, x) = self.size
        for y_ in range(y):
            for x_ in range(x):
                print(self.board[y_][x_], end=' ')
            print('')

class Player:
    def __init__(self, id:int, board:Board):
        self.id = id
        self.board = board
        self.size = self.board.size
        self.nowChessLocation = (-1, -1)
    
    def getEmptyArea(self) -> np.array:
        '''
        Allocate np.zeros and set empty location as 1 and return this array.
        '''
        ptr = np.where(self.board.board == 0)
        area = np.zeros(self.size)
        area[ptr] = 1
        return area
    
    def getPlayerView(self) -> np.array:
        '''
        Allocate np.zeros, set self chess as 1 and opponent chess as -1, and return this array.
        '''
        self_area = np.zeros(self.size)
        self_area[np.where(self.board.board == self.id)] = 1

        opp_area = self.board.board.copy()
        opp_area[np.where(self.board.board == self.id)] = 0
        opp_area[np.where(opp_area != 0)] = -1

        area = self_area + opp_area

        return area
    
    def setChess(self, location:(int, int)) -> bool:
        '''
        If the location is not empty, return false. Else return true.
        '''
        self.nowChessLocation = location
        b = self.board.setChess(location, self.id)
        return b

class Judgment:
    def __init__(self, board:Board):
        self.board = board

    def directionChessNumber(self, player:Player, dx:int, dy:int, maxNum:int = float('+inf')) -> (int, int):
        '''
        Return (num, outChess). num means what's number of player's chess in direction, outChess means what make
        player's chess interrupt(-1 means out area, 0 means no chess, player.id means opponent's chess)
        '''
        num = 0
        outChess = -1
        (y_min, x_min) = (0, 0)
        (y_max, x_max) = self.board.size
        (y, x) = player.nowChessLocation
        while (y_min <= y < y_max and x_min <= x < x_max and num < maxNum):
            if (self.board.board[y][x] == player.id):
                num += 1
                y += dy
                x += dx
            else:
                outChess = self.board.board[y][x]
                break
        
        return (num, outChess)
    
    def win(self, player:Player) -> int:
        '''
        If win, return player's id, else return 0.
        '''
        direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for (dy, dx) in direction:
            (num, stat) = self.directionChessNumber(player, dx, dy, 6)
            if (num >= 5):
                return player.id
        return 0

class Game:
    def __init__(self, boardSize:int = (15, 15)):
        self.boardSize = boardSize
        self.board = Board(boardSize)
        self.judgment = Judgment(self.board)
        self.player1 = Player(1, self.board)
        self.player2 = Player(2, self.board)
    
    def getPlayer1(self):
        return self.player1
    
    def getPlayer2(self):
        return self.player2

    def nextTurn(self, player:Player, location:(int, int)):
        '''
        location: (y, x)
        '''
        isWin = bool()
        score = 0
        player.setChess(location)
        win = self.judgment.win(player)
        if (win > 0):
            isWin = True
        return (isWin, score)
    
    def drawBoard(self):
        self.board.draw()

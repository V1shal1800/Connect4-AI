import copy
import numpy as np
import random,math
import gzip
import json

#Utility functions for game board
def getValidMoves(state,num_row, num_col):
    validMoves = []
    top = np.zeros(num_col, dtype=np.int16)
    for i in range(num_col):
        top[i] = num_col
    for i in range(num_col):
        for j in range(num_row):
            if state[j][i] != 0:
                top[i] = j
                break
    for i in range(num_col):
        if top[i] > 0:
            validMoves.append(i)
    return validMoves

def checkResult(state, num_row, num_col):
    for i in range(num_row):
        for j in range(num_col):
            if(i+3 < num_row):
                if(state[i][j] == state[i+1][j] == state[i + 2][j] == state[i + 3][j]) and (state[i][j] != 0):
                    return state[i][j]
            if(j+3 < num_col):
                if(state[i][j] == state[i][j+1] == state[i][j+2] == state[i][j+3]) and (state[i][j] != 0):
                    return state[i][j]
        
    #Checking diagonal
    for i in range(num_row - 3):
        for j in range(num_col - 3):
            if(state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3]) and (state[i][j] != 0):
                return state[i][j]

    for i in range(3,num_row):
        for j in range(num_col-3):
            if(state[i][j] == state[i-1][j+1] == state[i-2][j+2] == state[i-3][j+3]) and (state[i][j] != 0):
                return state[i][j]

    #Checking if board full
    for i in range(num_row):
        for j in range(num_col):
            if(state[i][j] == 0):
                return -1
    
    return 0
          
def Move(state, player, col, num_row):
    top = num_row -1
    for i in range(num_row):
        if state[i][col] != 0:
            top = i - 1
            break
    newstate = copy.deepcopy(state)
    newstate[top][col] = player
    return newstate
    
def viewTable(state, num_row, num_col):
    for i in range(num_row):
        for j in range(num_col):
            print(state[i][j], end = " ")
        print(" ")


#MCTS Algorithm
class Node:
    def __init__(self,state,player,parent,action):
        self.state = state
        self.player = player
        self.reward = 0
        self.parent = parent
        self.children = []
        self.numVisits = 0
        self.action = action

class MCTS:
    def __init__(self,num_playouts,num_row,num_col):
        self.num_playouts = num_playouts
        self.root = Node(None,None,None,None)
        self.num_row = num_row
        self.num_col = num_col


    def getUCB(self,node):
        C = 0.8
        if(node.numVisits == 0 or node.parent.numVisits == 0):
            ucb = 10
        else:
            ucb = (node.reward/(node.numVisits)) + C*math.sqrt((math.log(node.parent.numVisits))/(node.numVisits))
        return ucb

    def select(self):
        currNode = self.root
        while(len(currNode.children) != 0):
            currNode.numVisits += 1
            children = currNode.children
            maxUCB = -math.inf
            maxNode = None
            for child in children:
                ucb = self.getUCB(child)
                if(ucb > maxUCB):
                    maxUCB = ucb
                    maxNode = child
            currNode = maxNode

        return currNode

    def expand(self, node):
        validMoves = getValidMoves(node.state,self.num_row,self.num_col)
        state = node.state
        player = None
        if checkResult(node.state,self.num_row,self.num_col) >= 0:
            return node
        if(node.player == 1):
            player = 2
        else:
            player = 1

        for move in validMoves:
            childState = Move(state,player,move,self.num_row)
            child = Node(childState,player,node,move)
            node.children.append(child)

        returnChild =  random.choice(node.children)
        returnChild.numVisits += 1
        return returnChild

    def nextNodeSimulate(self,node):
        validMoves = getValidMoves(node.state,self.num_row,self.num_col)
        state = node.state
        player = None
        if checkResult(node.state,self.num_row,self.num_col) >= 0:
            return node
        if(node.player == 1):
            player = 2
        else:
            player = 1

        children = []
        for move in validMoves:
            childState = Move(state,player,move,self.num_row)
            child = Node(childState,player,node,move)
            children.append(child)

        return random.choice(children)

    def simulate(self, node):
        currNode = node
        while checkResult(currNode.state,self.num_row,self.num_col) < 0:
            currNode = self.nextNodeSimulate(currNode)

        result = checkResult(currNode.state,self.num_row,self.num_col)
        return result

    def backProp(self, node, result):
        if(node == None):
            return
        if result == self.root.player:
            reward = 1
        elif result == 0:
            reward = 0
        else:
            reward = -10
        node.reward += reward

        self.backProp(node.parent,result)

    def run(self,state,player):
        self.root = Node(state,player,None,None)
        currNode = self.root
        for i in range(3):
            currNode = self.expand(currNode)
            
        while(self.num_playouts):
            leaf = self.select()
            child = self.expand(leaf)
            result = self.simulate(child)
            self.backProp(child,result)
            self.num_playouts -= 1
        maxNode = None
        maxUCB = -math.inf
        for child in self.root.children:
            ucb = self.getUCB(child)
            if(ucb > maxUCB):
                maxUCB = ucb
                maxNode = child

        return maxNode.action,maxUCB

class QL:
    def __init__(self, alpha, epsilon,gamma,player,num_rows):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_rows = num_rows
        self.num_cols = 5
        self.player = player
        self.Qmap = {}
        self.initQValue = 0
        
    def getNextState(self,state):
        validMoves = getValidMoves(state,self.num_rows,5)
        children = []
        moveMap = {}
        for move in validMoves:
            nextState = Move(state,self.player,move,self.num_rows)
            children.append(nextState)
            byteHash = str(nextState.tobytes())
            moveMap[byteHash] = move
            if byteHash not in self.Qmap.keys():
                self.Qmap[byteHash] = self.initQValue
        if random.random() < self.epsilon:
            returnChild = random.choice(children)
            return returnChild, moveMap[str(returnChild.tobytes())]
        else:
            maxQ = -math.inf
            maxChild = None
            moveChosen = 0
            for child in children:
                byteHash = str(child.tobytes())
                if self.Qmap[byteHash] > maxQ:
                    maxQ = self.Qmap[byteHash]
                    maxChild = child
                    moveChosen = moveMap[byteHash]
            return maxChild, moveChosen

    def getMaxQ(self, state):
        validMoves = getValidMoves(state,self.num_rows,5)
        children = []
        for move in validMoves:
            nextState = Move(state,self.player,move,self.num_rows)
            children.append(nextState)
            byteHash = str(nextState.tobytes())
            if byteHash not in self.Qmap.keys():
                self.Qmap[byteHash] = self.initQValue
        maxQ = -math.inf
        for child in children:
            byteHash = str(child.tobytes())
            if self.Qmap[byteHash] > maxQ:
                maxQ = self.Qmap[byteHash]
        return maxQ

    def step(self,state):
        byteHash = str(state.tobytes())
        if checkResult(state, self.num_rows,5) >= 0:
            # self.Qmap[byteHash] = 0
            return
        nextState,nextMove = self.getNextState(state)
        nextResult = checkResult(nextState, self.num_rows,5)
        reward = 0
        if nextResult >= 0:
            if nextResult == self.player:
                reward = 2
            elif not nextResult:
                reward = -1
            else:
                reward = -5

        if byteHash not in self.Qmap.keys():
            self.Qmap[byteHash] = self.initQValue
        self.Qmap[byteHash] = self.Qmap[byteHash] + self.alpha*(reward + (self.gamma * self.getMaxQ(nextState)) - self.Qmap[byteHash])
        return nextState

    def play(self,state):
        byteHash = str(state.tobytes())
        if checkResult(state, self.num_rows,5) >= 0:
            return
        nextState,nextMove = self.getNextState(state)
        if byteHash not in self.Qmap.keys():
            self.Qmap[byteHash] = self.initQValue
        return nextState,nextMove,self.Qmap[byteHash]
    
    def dumpQValues(self):
        with gzip.open('./2019A7PS0036G_VISHAL.dat.gz', 'wb') as f:
            f.write(bytes(json.dumps(self.Qmap),'utf-8'))

    def loadQValues(self):
        with gzip.open('./2019A7PS0036G_VISHAL.dat.gz', 'rb') as f:
            file = f.read()
            self.Qmap = json.loads(file.decode('utf-8'))

def PrintGrid(positions):
    print('\n'.join(' '.join(str(x) for x in row) for row in positions))
    print()

def main():
    choice = input("a or c: ")
    
    if choice == "a":
        for i in range(50):
            numMoves = 0
            state = np.zeros((6,5), dtype=np.int16)
            print()
            while(checkResult(state,6,5) < 0):
                player1 = MCTS(40,6,5)
                player2 = MCTS(200,6,5)
                player1Move, player1value = player1.run(state,1)
                state = Move(state,1,player1Move,6)
                numMoves += 1
                print('Player 1 (MCTS with 40 playouts)')
                print(f'Action selected: {player1Move+1}')
                print('Total playouts for next state: 40')
                print(f'Value of next state according to MCTS: {player1value}')
                viewTable(state,6,5)
                print()
                if checkResult(state,6,5) >= 0:
                    break
                player2Move,player2value = player2.run(state,2)
                state = Move(state,2,player2Move,6)
                numMoves += 1
                print('Player 2 (MCTS with 200 playouts)')
                print(f'Action selected: {player2Move+1}')
                print('Total playouts for next state: 40')
                print(f'Value of next state according to MCTS: {player2value}')
                viewTable(state,6,5)
                print()
            if checkResult(state,6,5) > 0:
                print(f'Player {checkResult(state,6,5)} has WON. Total moves = {numMoves}.')
            else:
                print(f'Game drawn. Total moves = {numMoves}.')
        for i in range(50):
            numMoves = 0
            state = np.zeros((6,5), dtype=np.int16)
            print()
            while(checkResult(state,6,5) < 0):
                player1 = MCTS(200,6,5)
                player2 = MCTS(40,6,5)
                player1Move, player1value = player1.run(state,1)
                state = Move(state,1,player1Move,6)
                numMoves += 1
                print('Player 1 (MCTS with 200 playouts)')
                print(f'Action selected: {player1Move+1}')
                print('Total playouts for next state: 40')
                print(f'Value of next state according to MCTS: {player1value}')
                viewTable(state,6,5)
                print()
                if checkResult(state,6,5) >= 0:
                    break
                player2Move,player2value = player2.run(state,2)
                state = Move(state,2,player2Move,6)
                numMoves += 1
                print('Player 2 (MCTS with 40 playouts)')
                print(f'Action selected: {player2Move+1}')
                print('Total playouts for next state: 40')
                print(f'Value of next state according to MCTS: {player2value}')
                viewTable(state,6,5)
                print()
            if checkResult(state,6,5) > 0:
                print(f'Player {checkResult(state,6,5)} has WON. Total moves = {numMoves}.')
            else:
                print(f'Game drawn. Total moves = {numMoves}.')

    
    if choice == "c":
        state = np.zeros((4,5), dtype=np.int16)
        print()
        n = int(input("Enter n(number of playouts): "))
        QAgent = QL(0,0,0,2,4) # 0 value passed as parameter because the Q agent only uses trained estimates here
        numMoves = 0
        while(checkResult(state,4,5) < 0):
            player1 = MCTS(n,4,5)
            player1Move,player1value = player1.run(state,1)
            state = Move(state,1,player1Move,4)
            numMoves += 1
            print(f'Player 1 (MCTS with {n} playouts)')
            print(f'Action selected: {player1Move+1}')
            print(f'Total playouts for next state: {n}')
            print(f'Value of next state according to MCTS: {player1value}')
            viewTable(state,4,5)
            if checkResult(state,4,5) >= 0:
                break
            QAgent.loadQValues()
            state,player2Move,player2value = QAgent.play(state)
            numMoves += 1
            print('Player 2 (Q-learning)')
            print(f'Action selected : {player2Move}')
            print(f'Value of next state according to Q-learning : {player2value}')
            viewTable(state,4,5)
        if checkResult(state,4,5) > 0:
                print(f'Player {checkResult(state,4,5)} has WON. Total moves = {numMoves}.')
        else:
            print(f'Game drawn. Total moves = {numMoves}.')
    
if __name__=='__main__':
    main()

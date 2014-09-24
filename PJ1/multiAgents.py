# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        oldGhostStates = currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        oldScaredTimes = [ghostState.scaredTimer for ghostState in oldGhostStates]

	"*** YOUR CODE HERE ***"
	newNumFood = successorGameState.getNumFood()
	oldNumFood = currentGameState.getNumFood()
	max = 500 
	min = -500
	total = 0
	eatGhost = 500 
	for index, ghost in enumerate(newGhostStates):
          dist = util.manhattanDistance(newPos, ghost.getPosition())
	  if newScaredTimes[index] == 0:
	    if (dist > 1):
	      total += 50 
	    if (ghost.getPosition() == newPos):
	      total += min
	  else:
	    total += 50
	    if (ghost.getPosition() == newPos):
	      total += eatGhost 
	    if oldScaredTimes[index] != 0:
	      total -= dist

	#countCloseFood = 0
	closestFood = 999999 
	for f in newFood.asList():
	    distance = util.manhattanDistance(f,newPos)
	    if distance < closestFood:
	      closestFood = distance
	return total + (100/(closestFood+1)) + 50*(oldNumFood - newNumFood) + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 7)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
	actions = gameState.getLegalActions(self.index)
	if len(actions) == 0:
	  return self.evaluationFunction(gameState)
	value = self.getMinValue(self.index + 1,gameState.generateSuccessor(self.index,actions[0]),self.depth)
	bestAction = actions[0]
	for a in actions[1:]:
	  currValue = self.getMinValue(self.index + 1, gameState.generateSuccessor(self.index,a),self.depth)
	  if currValue > value:
	    value = currValue
	    bestAction = a 
	return bestAction

    def getMinValue(self,agentIndex,gameState,depth):
      actions = gameState.getLegalActions(agentIndex)
      if len(actions) == 0:
	return self.evaluationFunction(gameState)
      if agentIndex == gameState.getNumAgents() - 1:
	value = self.getMaxValue(0, gameState.generateSuccessor(agentIndex,actions[0]),depth - 1)
      else:
	value = self.getMinValue(agentIndex + 1, gameState.generateSuccessor(agentIndex,actions[0]),depth)
      for a in actions[1:]:
	if agentIndex == gameState.getNumAgents() - 1:
	  currValue = self.getMaxValue(0, gameState.generateSuccessor(agentIndex,a),depth - 1)
	else:
	  currValue = self.getMinValue(agentIndex + 1, gameState.generateSuccessor(agentIndex,a),depth )
	value = min(value,currValue)
      return value

    def getMaxValue(self,agentIndex,gameState,depth):
      if (depth == 0):
        return self.evaluationFunction(gameState)
      actions = gameState.getLegalActions(agentIndex)
      if len(actions) == 0:
        return self.evaluationFunction(gameState)
      value = self.getMinValue(agentIndex + 1, gameState.generateSuccessor(agentIndex,actions[0]),depth)
      for a in actions[1:]:
        currValue = self.getMinValue(agentIndex + 1, gameState.generateSuccessor(agentIndex,a),depth)
        value = max(value,currValue)
      return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(self.index)
        if len(actions) == 0:
          return self.evaluationFunction(gameState)
        value = self.getExpectiValue(self.index + 1,gameState.generateSuccessor(self.index,actions[0]),self.depth)
        bestAction = actions[0]
        for a in actions[1:]:
          currValue = self.getExpectiValue(self.index + 1, gameState.generateSuccessor(self.index,a),self.depth)
          if currValue > value:
            value = currValue
            bestAction = a
        return bestAction

    def getExpectiValue(self,agentIndex,gameState,depth):
      actions = gameState.getLegalActions(agentIndex)
      if len(actions) == 0:
        return self.evaluationFunction(gameState)
      if agentIndex == gameState.getNumAgents() - 1:
        value = self.getMaxValue(0, gameState.generateSuccessor(agentIndex,actions[0]),depth - 1)
      else:
        value = self.getExpectiValue(agentIndex + 1, gameState.generateSuccessor(agentIndex,actions[0]),depth)
      for a in actions[1:]:
        if agentIndex == gameState.getNumAgents() - 1:
          currValue = self.getMaxValue(0, gameState.generateSuccessor(agentIndex,a),depth - 1)
        else:
          currValue = self.getExpectiValue(agentIndex + 1, gameState.generateSuccessor(agentIndex,a),depth )
        value += currValue
      return value*1.0/len(actions)

    def getMaxValue(self,agentIndex,gameState,depth):
      if (depth == 0):
        return self.evaluationFunction(gameState)
      actions = gameState.getLegalActions(agentIndex)
      if len(actions) == 0:
        return self.evaluationFunction(gameState)
      value = self.getExpectiValue(agentIndex + 1, gameState.generateSuccessor(agentIndex,actions[0]),depth)
      for a in actions[1:]:
        currValue = self.getExpectiValue(agentIndex + 1, gameState.generateSuccessor(agentIndex,a),depth)
        value = max(value,currValue)
      return value

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 9).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    numFood = currentGameState.getNumFood()
    locFood = currentGameState.getFood()
    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]


    max = 500 
    min = -500
    total = 0
    eatGhost = 100 #try 100, used to be 500
    for index, ghost in enumerate(ghostStates):
      dist = util.manhattanDistance(pacmanPos, ghost.getPosition())
      if scaredTimes[index] == 0:
        if (dist > 1):
          total += 50 #try 10
        if (ghost.getPosition() == pacmanPos):
          total += min
      else:
        total += 50
        if (ghost.getPosition() == pacmanPos):
          total += eatGhost 

    closestFood = 999999 
    for f in locFood.asList():
        distance = util.manhattanDistance(f,pacmanPos)
        if distance < closestFood:
          closestFood = distance

    numFoodWeight = 1000
    closestFoodWeight = 100
    if numFood <=5:
      numFoodWeight = 5000
      closestFoodWeight = 50
    return total + (closestFoodWeight/(closestFood+1)) + (numFoodWeight/(numFood + 1)) + score
    

# Abbreviation
better = betterEvaluationFunction


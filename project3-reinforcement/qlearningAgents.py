# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qvalue = util.Counter()
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qvalue[state,action]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
      
        # Get a list of legal actions for the given state.
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return 0.0
        max_q_value = float('-inf')
        # Iterate over all legal actions and update the maximum Q-value.
        for action in legal_actions:
            q_value = self.getQValue(state, action)
            if q_value > max_q_value:
                max_q_value = q_value
        return max_q_value
      # "Not sure if above is correct lol"
          
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        best_actions = []
         # Get a list of legal actions for the given state.
        legal_actions = self.getLegalActions(state)
        max_q_value = float('-inf')
        # Iterate over all legal actions and update the maximum Q-value.
        for action in legal_actions:
            q_value = self.getQValue(state, action)
            # If the Q-value is higher than the current maximum, update the maximum and best actions
            if q_value > max_q_value:
                max_q_value = q_value
                best_actions = [action]
            # If the Q-value is equal to the maximum, add the action to the best actions list
            elif q_value == max_q_value:
                best_actions.append(action)
        if best_actions:
            return random.choice(best_actions)
        else:
            return None


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

        #return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
    
        next_q_value = self.computeValueFromQValues(nextState)
        sample = reward + self.discount * next_q_value
        
        key = (state, action)
        current_q_value = self.getQValue(state, action)
        updated_q_value = (1.0 - self.alpha) * current_q_value + self.alpha * sample
        self.qvalue[key] = updated_q_value
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        qval = 0
        for f in self.featExtractor.getFeatures(state,action):
          qval += self.featExtractor.getFeatures(state,action)[f] * self.weights[f]
        return qval

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        max_q_value = 0 
        legal_actions = self.getLegalActions(nextState)
        if not legal_actions:
            max_q_value = 0.0
        else: 
          max_q_value = float('-inf')
          for actionitem in legal_actions:
              q_value = self.getQValue(nextState, actionitem)
              if q_value > max_q_value:
                  max_q_value = q_value

        difference = reward + max_q_value * self.discount - self.getQValue(state,action)

        for f in self.featExtractor.getFeatures(state,action):
            self.weights[f] += (self.featExtractor.getFeatures(state,action)[f] * (self.alpha*difference))

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

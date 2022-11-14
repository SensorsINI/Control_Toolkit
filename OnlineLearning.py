import tensorflow as tf

class OnlineLearning:
    def __init__(self, predictor):
        self.predictor = predictor

        self.s_previous = None
        self.u_previous = None

    def step(self, s, u, time, updated_attributes):
        """ Retraining of neural network used for predictions"""
        pass


        # net_input = tf.concat(self.s_previous, self.u_previous)
        #
        # self.s_predicted = self.predictor.net(net_input)
        #
        # loss = (s-self.s_predicted)**2
        #
        # # Train on this loss and modify self.predictor.net
        #
        #
        # self.s_previous = s
        # self.u_previous = u


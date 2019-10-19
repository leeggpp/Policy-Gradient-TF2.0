import tensorflow as tf
import numpy as np
import gym

tf.keras.backend.set_floatx('float64')

env = gym.make('CartPole-v1')
InputSize = 4
OutputSize = 2

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100, input_shape=(InputSize,) , activation='relu', kernel_initializer='glorot_uniform'))
model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(OutputSize, activation = 'softmax', kernel_initializer='glorot_uniform'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MSE)
#categorical_crossentropy


def discount_rewards(r, gamma = 0.8):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

scores = []
for ep in range(100000):
	s = env.reset()
	state = []
	reward = []
	action = []
	action_ = []
	score = 0
	done = False
	while not done:
		s = s.reshape([1, 4])
		a = model(s)
		a_ = np.random.choice(2, p = a.numpy()[0])
		s_, r, done, _ = env.step(a_)
		score += r
		r = r if not done or score == 500 else -10
		state.append(s)
		reward.append(r)
		action.append(a.numpy()[0])
		action_.append(a_)
		s = s_
	
	r_ = discount_rewards(np.array(reward))
	
	# r_ -= np.mean(r_)
	# r_ /= np.std(r_)
	# r_ = np.array(reward)
	action = np.array(action)
	x = np.array(state).reshape(len(state), InputSize)
	y = np.zeros((len(action_), OutputSize))
	r_ = r_.reshape(len(r_), 1)
	# y[np.arange(len(action_)), action_] = action[np.arange(len(action_)), action_]
	y[np.arange(len(action_)), action_] = 1
	y *= r_
	# r_ = r_.reshape(len(r_))
	# model.fit(x = x, y = y, sample_weight = r_, verbose = 0)
	# model.fit(x = x, y = y, class_weight = r_, verbose = 0)
	model.fit(x = x, y = y,  verbose = 0)
	scores.append(score)
	if ep % 50 == 0:
		print (np.mean(np.array(scores[-50:])))
	if np.mean(np.array(scores[-50:])) >= 490:
		model.save('C:/Users/paklo/OneDrive/Desktop/Python/Savemodel/CartPole-v0-p{}g.h5'.format(ep))
		print ('Success in ', ep)
		break
	

	
	
	


	

import tensorflow as tf
import numpy as np
import gym


env = gym.make('CartPole-v1')
InputSize = 4
OutputSize = 2

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, input_shape=(4,) , activation='relu'))
model.add(tf.keras.layers.Dense(2, activation = 'softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.categorical_crossentropy)


def discount_rewards(r, gamma = 0.8):
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r
	
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
		state.append(s)
		reward.append(r)
		action.append(a)
		action_.append(a_)
		score += r
		s = s_
		
	r_ = discount_rewards(np.array(reward))
	r_ -= np.mean(r_)
	r_ /= np.std(r_)
	x = np.zeros((len(state), InputSize))
	y =  np.zeros((len(action_), OutputSize))
	for i in range(len(state)):
		x[i] = state[i]
		y[i][action_[i]] = r_[i]
	if ep%100 == 0:
		print (score)	
	model.fit(x=x, y=y, verbose=0)
	
  
  https://sergioskar.github.io/Policy-Gradients/

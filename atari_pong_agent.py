import gym
import numpy as np

def reduce_image_resolution(image):
    # Take only alternate pixels and halves the resolution of the image
    return image[::2, ::2, :]

def remove_image_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_image_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image
def set_rest_to_one(image):
    image[image != 0] = 1 # Set everything else (ball, paddles) to 1
    return image

def image_preprocess_observations(current_observation_image, previous_image_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    image_processed_observation = current_observation_image[35:195] # crop
    image_processed_observation = reduce_image_resolution(image_processed_observation)
    image_processed_observation = remove_image_color(image_processed_observation)
    image_processed_observation = remove_image_background(image_processed_observation)
    image_processed_observation = set_rest_to_one(image_processed_observation)
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    image_processed_observation = image_processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one
    if previous_image_processed_observation is not None:
        image_current_observation = image_processed_observation - previous_image_processed_observation
    else:
        current_observation_image = np.zeros(input_dimensions)
    # update the previous_image_processed_observation
    image_processed_observation = image_processed_observation
    return current_observation_image, image_processed_observation


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        return 2 # signifies upward movement in OpenAI gym pong implmentation
    else:
        return 3 # signifies downward movement in OpenAI gym pong implmentation

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    """ Refer: http://neuralnetworksanddeeplearning.com/chap2.html"""
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    """ Refer: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop"""
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer

def discount_rewards(rewards, gamma):
    """ Actions taken 20 steps before the end result are less important to the overall result than an action taken a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # since this was a game boundary (pong specific!), reset the sum
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    """ discount the gradient with the normalized rewards """
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal which helps to control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards

# Define the model in keras (WIP)
def keras_learning_model(input_dim=80*80, model_type=1):
  model = Sequential()
  if model_type==0:
    model.add(Reshape((1,80,80), input_shape=(input_dim,)))
    model.add(Flatten())
    model.add(Dense(200, activation = 'relu'))
    model.add(Dense(number_of_inputs, activation='softmax'))
    opt = RMSprop(lr=learning_rate)
  else:
    model.add(Reshape((1,80,80), input_shape=(input_dim,)))
    model.add(Convolution2D(32, 9, 9, subsample=(4, 4), border_mode='same', activation='relu', init='he_uniform'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu', init='he_uniform'))
    model.add(Dense(number_of_inputs, activation='softmax'))
    opt = Adam(lr=learning_rate)
  model.compile(loss='categorical_crossentropy', optimizer=opt)
  if resume == True:
    model.load_weights('pong_model_checkpoint.h5')
  return model

model = learning_model()

# The execution starts here
def start():
    env = gym.make("Pong-v0") # start the OpenAI gym pong environment
    observation_image = env.reset() # get the image

    # hyperparameters
    episode_number = 0
    batch_size = 10
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80 * 80
    learning_rate = 0.001 # original value: 1e-4, adam optimizer used here in keras model. (Refer : https://keras.io/optimizers/)


    #Script Parameters for keras
    update_frequency = 1 # to decide frequency of update for the keras model parameters
    resume = False # to load a previous checkpoint model weights to run again.
    render = True # to render the OpenAI environment.
    train_X = []
    train_y = []

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
    xs, dlogps, drs, probs = [],[],[],[]

    episode_number = 0
    reward_sum = 0
    running_reward = None
    previous_processed_observations_image_vector = None

    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }

    # To be used with rmsprop algorithm. Refer:(http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])


    model = learning_model() # compile the model.

    while True:
        env.render()
        processed_observations_image_vector, previous_processed_observations_image_vector = image_preprocess_observations(observation_image, previous_processed_observations_image_vector, input_dimensions)

        # predict probabilities from the model
        up_probability = ((model.predict(processed_observations_image_vector.reshape([1,processed_observations_image_vector.shape[0]]), batch_size=1).flatten()))

        episode_observations.append(processed_observations_image_vector)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        # Implement the chosen action. Get back the details from the enviroment after performing the action
        observation_image, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        # Refer: http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)

        # check if an episode is finished
        if done:
            episode_number += 1

            # combine the following values for the episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)

            gradient = compute_gradient(
              episode_gradient_log_ps_discounted,
              episode_hidden_layer_values,
              episode_observations,
              weights
            )

            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # reset values
            observation_image = env.reset() # reset openAI environment
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print 'Resetting environment. Total Episode Reward: %f. Running Mean: %f' % (reward_sum, running_reward)
            reward_sum = 0
            previous_processed_observations_image_vector = None

start()

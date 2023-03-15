print("\nLOKAL/DIGITAL ANN RECOGNIZER\n")
print("Samuel Cho\n")
print("2020\n\n\n\n")

import sys
import numpy as np
from tensorflow.python.keras.models import load_model
from pythonosc.dispatcher import Dispatcher 
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer

#load max and min values of data set
max_values = np.load('max_values.npy')
min_values = np.load('min_values.npy')


#LOAD MODELS
print("Loading models..")
num_dir_model = load_model('models/num_dir.h5')
print("\n\n\n\n\nNum_dir_model loaded")

deposit_amt_model = load_model('models/deposit_amt.h5')
print("deposit_amt_model loaded")

max_speed_model = load_model('models/max_speed.h5')
print("max_speed_model loaded")

max_force_model = load_model('models/max_force.h5')
print("max_force_model loaded")

decay_rate_model = load_model('models/decay_rate.h5')
print("decay_rate_model loaded")

sense_dist_model = load_model('models/sense_dist.h5')
print("sense_dist_model loaded")

slope_step_model = load_model('models/slope_step.h5')
print("slope_step_model loaded")

displace_size_model = load_model('models/displace_size.h5')
print("displace_size_model loaded")
print("Models successfully loaded\n\n")


# DEFINE CALLBACKS

def send_prediction(p):
	client.send_message("/wek/outputs", p)
	

def get_predictions(in_vec):
	num_dirs = int(np.argmax(num_dir_model.predict(x=in_vec)))
	deposit_amt = int(np.argmax(deposit_amt_model.predict(x=in_vec)))
	max_speed = int(np.argmax(max_speed_model.predict(x=in_vec)))
	max_force = int(np.argmax(max_force_model.predict(x=in_vec)))
	decay_rate = int(np.argmax(decay_rate_model.predict(x=in_vec)))
	sense_dist = int(np.argmax(sense_dist_model.predict(x=in_vec)))
	slope_step = int(np.argmax(slope_step_model.predict(x=in_vec)))
	displace_size = int(np.argmax(displace_size_model.predict(x=in_vec)))
	prediction = [num_dirs, 
				deposit_amt, 
				max_speed, 
				max_force, 
				decay_rate, 
				sense_dist, 
				slope_step, 
				displace_size]
	print(prediction)
	for i in range(len(prediction)):
		prediction[i] += 1

	send_prediction(prediction)




#OSC call back
def normalize(address, *args):
	current_in_vector = []
	for i in range(len(args)):
		norm_arg = (args[i] - min_values[i]) / (max_values[i] - min_values[i])
		current_in_vector.append(norm_arg)
	current_in_vector = np.array(current_in_vector)
	current_in_vector = current_in_vector.reshape(-1,49)
	get_predictions(current_in_vector)

#SETUP OSC
print("Setting up osc...")
dispatcher = Dispatcher()
dispatcher.map("/wek/inputs", normalize)

SEND = 12000
RECEIVE = 6448

server = BlockingOSCUDPServer(("localhost", RECEIVE), dispatcher)
client = SimpleUDPClient("localhost", SEND)
print("OSC is done setting. Listening at port {} and sending at port {}\n".format(RECEIVE,SEND))
print("Turn on puredata to play")


server.serve_forever()
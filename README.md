# NILM
This project aims to identify events like switching on and off time of various appliances in a house from the main utility meter data.
The concept behind this framework is visualizing a window of meter data as an image and to segment out or decompose it into its constituent features. Thus we have used two convolution blocks to filter the data and make higher level features out of it.
Next we have also placed some RNN blocks so that the neural network has a memory to keep the devices on at a given point of time in consideration.

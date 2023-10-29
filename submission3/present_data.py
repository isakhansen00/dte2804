#!/usr/bin/env python

# Import the necessary ROS and message types
import rospy
from std_msgs.msg import Float32

# Callback function for raw data
def raw_callback(data):
    # Log the received raw data to the console
    rospy.loginfo("Received raw data: %f", data.data)

# Callback function for filtered data
def filtered_callback(data):
    # Log the received filtered data to the console
    rospy.loginfo("Received filtered data: %f", data.data)

# Function to listen for data on the "signal_raw" and "signal_filtered" topics
def listener():
    # Initialize a ROS node named "present_data"
    rospy.init_node("present_data", anonymous=True)

    # Subscribe to the "signal_raw" and "signal_filtered" topics and call the respective callback functions
    rospy.Subscriber("signal_raw", Float32, raw_callback)
    rospy.Subscriber("signal_filtered", Float32, filtered_callback)

    # Keep the node running and processing incoming data
    rospy.spin()

if __name__ == "__main__":
    # Start the listener node to receive and display data
    listener()

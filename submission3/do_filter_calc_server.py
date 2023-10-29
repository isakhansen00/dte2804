#!/usr/bin/env python

# Import required ROS and message types
from oblig_package.srv import DoFilterCalc, DoFilterCalcResponse
import rospy
from std_msgs.msg import Float32

# Service callback function to perform filtering calculation
def handle_do_filter_calc(req):
    # Calculate the filtered output (yk) using the provided coefficients
    yk = (-req.a1 * req.ykm1) - (req.a2 * req.ykm2) + (req.b1 * req.ukm1) + (req.b2 * req.ukm2)
    # Return the filtered output as a service response
    return DoFilterCalcResponse(yk)

# Function to initialize the server and handle service requests
def do_filter_calc_server():
    # Initialize a ROS node with the name "do_filter_calc_server"
    rospy.init_node("do_filter_calc_server")
    # Create a service that listens on the "do_filter_calc" topic
    s = rospy.Service("do_filter_calc", DoFilterCalc, handle_do_filter_calc)
    # Keep the node running and handle service requests
    rospy.spin()

# Entry point for the script
if __name__ == "__main__":
    # Call the function to start the service server
    do_filter_calc_server()

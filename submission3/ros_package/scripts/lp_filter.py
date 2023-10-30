#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from oblig_package.srv import DoFilterCalc

# Initialize lists for the filter state variables
y = [0.0, 0.0]  # Filter output (y[k] and y[k-1])
u = [0.0, 0.0]  # Input signal (u[k] and u[k-1])

# Define filter coefficients
a1 = -1.0
a2 = 0.881911378298176
b1 = 0.060521644576243
b2 = 0.058037886046515

# Callback function for the /signal_raw topic
def signal_raw_callback(data):
    global y, u, a1, a2, b1, b2

    # Wait for the "do_filter_calc" service to be available
    rospy.wait_for_service("do_filter_calc")

    try:
        # Create a service proxy to call the filter calculation service
        do_filter_calc = rospy.ServiceProxy("do_filter_calc", DoFilterCalc)
        resp = do_filter_calc(a1, a2, b1, b2, y[-1], y[-2], u[-1], u[-2])

        # Update the filter state variables and publish the filtered output
        y.append(resp.yk)
        u.append(data.data)
        pub.publish(resp.yk)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == "__main__":
    # Initialize a ROS node with the name "lp_filter"
    rospy.init_node("lp_filter", anonymous=True)

    # Subscribe to the /signal_raw topic and call signal_raw_callback on new data
    rospy.Subscriber("signal_raw", Float32, signal_raw_callback)

    # Create a publisher for the filtered signal on the /signal_filtered topic
    pub = rospy.Publisher("signal_filtered", Float32, queue_size=10)

    # Run the ROS node continuously until it's stopped
    rospy.spin()

#!/usr/bin/env python

# Import necessary ROS and message types
import rospy
from std_msgs.msg import Float32

# Function to publish raw signals to the "signal_raw" topic
def talker(raw_signal_list):
    # Create a publisher for the "signal_raw" topic
    pub = rospy.Publisher("signal_raw", Float32, queue_size=10)
    # Initialize a ROS node named "signal_source"
    rospy.init_node("signal_source", anonymous=True)

    # Set the publishing rate to 20 Hz
    rate = rospy.Rate(20)

    # Iterate through the raw signal data
    for raw_signal in raw_signal_list:
        try:
            # Attempt to convert the raw signal to a float
            raw_signal_float = float(raw_signal)
            rospy.loginfo(raw_signal_float)  # Log the data
            pub.publish(Float32(raw_signal_float))  # Publish the raw signal as a Float32 message
        except ValueError:
            rospy.logerr(f"Invalid data format: {raw_signal}")

        rate.sleep()  # Control the publishing rate

if __name__ == "__main__":
    try:
        # 1. Specify the path to the .dat file
        file_path = 'src/oblig_package/scripts/signal_raw.dat'

        raw_signal_list = []  # Initialize a list to store data from the .dat file

        try:
            # 2. Open the .dat file for reading
            with open(file_path, 'r') as file:
                # 3. Read the data from the file line by line
                for line in file:
                    # 4. Parse and store the data in the list
                    data = line.strip()  # Remove leading/trailing whitespace
                    raw_signal_list.append(data)

        except FileNotFoundError:
            print(f"The file '{file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        # Call the talker function to publish the raw signal data
        talker(raw_signal_list)

    except rospy.ROSInterruptException:
        pass

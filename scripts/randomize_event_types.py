import argparse

import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Randomize the event types. This is useful to test if the"
                    "performance of joint random event types is the same as not using event types."
                    "This can only be used for equal-statistic partitions of event types."
    )

    args = parser.parse_args()

    layout_desc = "N.D25-4LSTs09MSTs-MSTN"  # LaPalma
    # layout_desc = 'S-M6C8aj-14MSTs37SSTs-MSTF'  # Paranal

    n_types = 3

    particles = ["gamma_cone", "electron", "proton"]

    for particle in particles:
        # It is necessary to start from a real ETs file to be able to take into account the negative
        # values, which should not change and are the same independently of the partition. Any .txt
        # file output from export_event_types.py will be good, even if the number of ETs is different.
    	
        # Read the event types from the file
        event_types = np.loadtxt(f"{particle}.{layout_desc}_ID0.txt")

        # Randomize the positive event types, assigning random values from 1 to n_types
        event_types_pos = np.random.randint(1, n_types + 1, size=len(event_types[event_types > 0]))

        # Fill the original arrays with the randomized event types
        event_types[event_types > 0] = event_types_pos

        # Save the randomized event types
        np.savetxt(f"{particle}.{layout_desc}_ID0_{n_types}types_random.txt", event_types, fmt="%d")

import argparse

import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Change the order of all (positive) event types at random. This is useful to test if the"
                    "performance of joint random event types is the same as not using event types."
    )

    args = parser.parse_args()

    layout_desc = "N.D25-4LSTs09MSTs-MSTN"  # LaPalma
    # layout_desc = 'S-M6C8aj-14MSTs37SSTs-MSTF'  # Paranal

    particles = ["gamma_cone", "electron", "proton"]

    for particle in particles:
        # Read the event types from the file
        event_types = np.loadtxt(f"{particle}.{layout_desc}_ID0.txt")

        # Mask the negative values
        event_types_pos = event_types[event_types > 0]

        # Randomize the event types
        np.random.shuffle(event_types_pos)

        # Fill the original arrays with the randomized event types
        event_types[event_types > 0] = event_types_pos

        # Save the randomized event types
        np.savetxt(f"{particle}.{layout_desc}_ID0_random.txt", event_types, fmt="%d")

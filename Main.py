#!/usr/bin/env python3
import os

ASCII_LOGO = """
▗▄▄▖▗▄▄▖  ▗▄▄▄▖▗▄▄▖ ▗▄▄▄▖▗▄▄▖ ▗▖ ▗▖▗▖  ▗▖     ▗▄▖ ▗▖   ▗▄▄▄▖▗▄▄▖▗▄▄▄▖
▐▌   ▐▌ ▐▌▐▌   ▐▌     █  ▐▌ ▐▌▐▌ ▐▌▐▛▚▞▜▌    ▐▌ ▐▌▐▌   ▐▌   ▐▌ ▐▌ █  
 ▝▀▚▖▐▛▀▘ ▐▛▀▀▘▐▌     █  ▐▛▀▚▖▐▌ ▐▌▐▌  ▐▌    ▐▛▀▜▌▐▌   ▐▛▀▀▘▐▛▀▚▖ █  
▗▄▄▞▘▐▌   ▐▙▄▄▖▝▚▄▄▖  █  ▐▌ ▐▌▝▚▄▞▘▐▌  ▐▌    ▐▌ ▐▌▐▙▄▄▖▐▙▄▄▖▐▌ ▐▌ █                                                            
                                                                  
"""

def main():
    while True:
        print(ASCII_LOGO)
        print("Welcome to Spectrum Alert")
        print("Please choose an option:")
        print("1. Gather Data (DataGathering.py)")
        print("2. Train Model (ModelTrainer.py)")
        print("3. Monitor Spectrum (Monitor.py)")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            duration = input("Enter the duration for data gathering (in minutes): ")
            os.system(f"python3 DataGathering.py {duration}")
        elif choice == "2":
            os.system("python3 ModelTrainer.py")
        elif choice == "3":
            os.system("python3 Monitor.py")
        elif choice == "4":
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.\n")

if __name__ == "__main__":
    main()

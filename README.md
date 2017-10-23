# pulse-programming

This is an interactive progamming language on gate level.

On a white board, you can paint blue NAND gates and black lines.
The NAND gates generate pulses which walk along the lines and
trigger other gates.

## Installation

You can setup this by downloading or cloning the repository.
Then, open a command line in the repository folder.

On Debain/Ubuntu you need to have Python 3 and Pip installed:

    sudo apt-get install python3 pip-python3

Then, we create a new environment for the Python packages:

    pip3 install --user virtualenv
    python3 -m virtualenv ENV -p python3

Every time, you are in the repository, you can run this command to load the
installed environment. Please do this, now.

    source ENV/bin/activate

Now, we install the pakcages required to run the programs.

    pip install -r requirements.txt

Now, all packages are installed.

## Run the Program

Open a Terminal/Console in this folder.
Run this to load the Python environment:

    source ENV/bin/activate

In order to run the program, you can type

    python pulse_programming.py photos/photo4.jpg

`Esc` - the Escape Key closes the program.

## Development

A develepment shortcut is

    while python pulse_programming.py photos/photo4.jpg; do sleep 1; done

to run the program and reload it when `Esc` is pressed.

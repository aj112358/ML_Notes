## Steps for Installing the 'graphviz' library in Anaconda for Windows 10

Written On: Tues, Nov 17, 2020

The installation process of the graphviz software is (more often than not) a very tedious and annoying process. Here are the steps that I took to download it successfully. Please update me if there are new ways of installing.

### Step 1 - Download the 'graphviz' application

First, we download and install the 'graphviz' file onto our Windows 10 machine. Make sure you use the correct installation depending on your version of Windows (32-bit or 64-bit). Here are the download links:

32-bit: https://www2.graphviz.org/Packages/stable/windows/10/cmake/Release/Win32/
64-bit: https://www2.graphviz.org/Packages/stable/windows/10/cmake/Release/x64/


### Step 2 - Install the 'graphviz' application

Double-click to open the .exe file, allow permission for installation if necessary, and simply go through each of the install screens.

**Important**: Make sure that you specify the install option of **"Add Graphviz to the system PATH for current users"** (can also do for all users). This install screen is the third in the installation process, and the screen comes right after the license agreement screen.


### Step 3 - Install 'graphviz' library on Anaconda

From the Start Menu, search for the "Anaconda Prompt" and **open as administrator**. In the prompt type: 'conda install python-graphviz'.

If you get an "InvalidArchiveError", this means that you (most likely) currently have a Jupyter Notebook/IPython environment running. Simply close/shutdown these and try the above command again. It should work this time. (The installer was not able to delete some files it needed to because some IPython environment was still running)


I hope these steps work for you. If they do not and you find another way, please feel free to update this file and send a pull request.

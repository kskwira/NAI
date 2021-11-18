<h3>
Authors: Krzysztof Skwira & Tomasz Lemke
</h3>

App that calculates similarity between users based on the list of movies seen and their individual score

Each user had the task to list at least 10 moves he/she watched and rank it accordingly \
with 10 = Great(Highly Recommended) and 1 = Terrible(Not recommended)

The "similarity" between users is calculated based on the euclidean distance. 

<h3>
Installation: 
</h3>

pip install numpy 


Script should be run in the folder containing scripts files as well as the CSV file with users in the correct format. \
To run the script you need to write in the terminal window: 
<p><b>collaborative_filtering.py --user "name of the user from CSV list to whom we want to calculate the score"</b></p> 

i.e. \
collaborative_filtering.py --user "Tomasz Lemke"


<h3>
Reference:
</h3>


https://en.wikipedia.org/wiki/Euclidean_distance \
https://numpy.org/
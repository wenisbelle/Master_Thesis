OK. Now, another big change. Create a class protocol name recharge_base. This recharge base will be the base we are using till this moment. It will be placed based on the gaussian mentioned early. It will always keep track of the drones that are charging on it, and will broadcast this to the near drones, so they will know which drones are charging when they are withing the range. Something very similar to the charging_notification I was using, with this section you can remove the role of the drone that made this, but keet the same result, the hability to change the behavior and store this data, to pass to others. 

Besides, the base can communicate with any other base, it doesn't matter how far it is in the map. So it can send a notification informing which drones are chargging, when they begin and so on. And also inform when the status change, some drone get in or get out. 


Ok. Now, I want you to create a file named all_optimizations_distance.py

My objective is to optimize the system and analyse the results for different values of TRANSMISSION_RANGE sequentionally. SO I can check the effect of the communication range in the system behavior. So I want that this file calls the main.py in the train mode to run optimizations for these distances: 100, 200, 500, 750, 1000 e 2000. Each run will save inside the swarm or egocentric directory a logbook named ga_logbook_DISTANCE_RANGE.py, so I will have the result of each run in the end. So I will just run this newly created script to run all. 

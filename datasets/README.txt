Metric store the distribution maps that will be upload to gpu server. 

I used to use the maps in folder ./distribution/. But when I generate new distributions map, Sometimes the new map may cover netmap generated before, which had been used to train my model. That's so bad that I have to run my code again to keep all my results are generated based on the same distribution map. 

So I new the 'in_use' folder to store the distribution maps that will not be covered.
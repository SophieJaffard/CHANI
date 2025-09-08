Description
--

This depository contains the code for the article [CHANI: Correlation-based Hawkes Aggregation of Neurons with bio-Inspiration](https://arxiv.org/abs/2405.18828). 

Usage
--
The file 'easy_task.ipynb' contains the code to run to reproduce the numerical results on simulated dataset. To get the results with the two expert aggregations EWA and PWA, the user should 
run the file with expert2 = 'PWA' and expert2 = 'EWA', and to get the results for the two tasks the reader should run the file with task = 'easy1' and task = 'easy2'.

The file 'Exp0hid.ipynb' contains the code to run to reproduce the numerical results on digits dataset without hidden layers. To get the results with the two expert aggregations EWA and PWA, the user should 
run the file with expert2 = 'PWA' and expert2 = 'EWA'.

The file 'Exp1hid.ipynb' contains the code to run to reproduce the numerical results on digits dataset with one hidden layer. To get the results with the two expert aggregations EWA and PWA, the user should 
run the file with expert1 = 'PWA' and expert2 = 'PWA', and expert1 = 'EWA' and expert2 = 'EWA'.

The file 'Exp1hid_allconn.ipynb' contains the code to run to repreoduce the numerical results on digits dataset with one hidden layer and additional connections. To get the results with the two expert aggregations EWA and PWA, the user should 
run the file with expert1 = 'PWA' and expert2 = 'PWA', and expert1 = 'EWA' and expert2 = 'EWA'.

The file 'Exp2hid.ipynb' contains the code to run to reproduce the numerical results on digits dataset with 2 hidden layers. To get the results with the two expert aggregations EWA and PWA, the user should 
run the file with expert1 = 'PWA', expert2 = 'PWA' and expert3 = 'PWA', and expert1 = 'EWA', expert2 = 'EWA' and expert3 = 'EWA'.

The file 'results.ipynb' contains the code to reproduce the figures of the article. It should be run after the other files.

The file 'hansolo.py' contains the functions needed in the other files.







'hansolo.py' contains


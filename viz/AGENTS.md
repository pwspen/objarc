This project is a library and visualization API + webpage for the ARC-AGI datasets (Abstract Reasoning Corpus)

A react project, using typescript and tailwind, is initialized in arc-viz/

Your task is to make a webpage that can display ARC tasks, using the endpoints and data models in /api.py. Replace the data models in Typescript and you should need to use all endpoints, so if you don't, something is wrong. 

There should be a bar along the top, with dropdowns for both the dataset and tasks. Alongside these dropdowns there should also be a 'random task' button to select random task from the current dataset

Every time the task dropdown is changed the main content of the screen should update

This is what the main content should look like. On the left of the screen there should be grids. Arrange input and output vertically with an arrow between them - so whenever you load a new task, the grid just under the top bar should be input1, then an arrow going to output1, then input2, etc. Display the test pair(s) in the same way, but with a clear line separator between the train and test pairs.

On the right half of the screen, lined up with each grid, display the WebGridData, whatever it is. Each string-float name-value pair should get one line. Round each value to 3 sig figs.

Your main goal here should be conciseness, extensibility, reusability etc. Do not worry much about styling yet, just for alignment or whatever.

You'll probably want the following components:
- grid cells
- grid data
- grid (cells + data, takes up full screen width)
- iopair (grid + centered downwards arrow + grid)
- train (list iopair)
- test (list iopair)
- task (train + separator + test)

Please notice my emphasis on conciseness and extensability and all that. Try your absolute hardest to not write spaghetti code. It needs to be modifiable and readable and all that good stuff.

Don't worry about running anything yet (python or web server), the user will take care of that.
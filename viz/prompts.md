Look at the component structure in src/components. To each IOPair, we need to
add some things: three grids of floats, basically heatmaps. These will be
returned as part of WebIOPair. So we need to add a dict of them to WebIOPair.
We may want to add more than 3 later. For now the 3 names (keys) will be
"Input Auto", "Output Auto", and "Cross". Location wise, they should be added
as a third column on the right (the existing two columns are the grid and the
printed data/metrics). If any of the 3 keys are missing from the IOPair throw
an error, but at the place where the grids are extracted to be used, not at
the API level (because, again, we might add or remove grids in the UI later).
Alongside each of these grids we will need a legend or colormap showing the
correspondence between color and number range. Also, the size of these grids
might vary a lot, including in aspect ratio, and we want to normalize it
somehow, but WITHOUT squishing the pixels into rectangles. I think it would be
sensible to pick a maximum dimension in pts or whatever, and for each grid,
set its longest dimension to that, so all the grids are approximately
similarly sized on the screen. Ok, that's all the requirements. Should we use
(modify?) the same Grid component for this? How should we go about setting
this up?

Great, I like the sound of all that. Please go ahead and make these changes. For the python api, please just put the actual grids as blanks in the dictionary for now and I'll handle that. For the legend, I can imagine we want a minimum vertical height so that it will be readable. Pick something reasonable. One thing I forgot to mention: There should be a toggle somewhere to turn the display of these heatmap grids on or off. With it turned off, the current UI should be basically unchanged. With it turned on, "Input Auto" should correspond with the input, then "Cross" should actually line up with the arrow between the input and output, and "Output Auto" should line up with the output. Please keep in mind that we really really care about the readability, maintainability, extensibility, conciseness etcetera of all of this code. I don't want spaghetti for dinner :) Does all of this sound good?
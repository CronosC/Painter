# Painter

Setup:
    - setup the env with conda
    - download a model (e.g. from civitai)
    - put into model folder, adjust path in run.py

Running:
    - fill out prompt with one or more prompts. Each line is interpreted as one prompt. For each line one result .png is created.
    - fill HQ file with modifiers which are used for all prompts. Should only have one line. Can not be empty, but may be only a " "
    - fill negative file with one or more negative prompts. These inform the process on what to avoid. Can have multiple lines, each line is interpreted as one negative prompt. Each combination of prompt x negative prompt is created as one result file.

    - OPTIONAL: Fill out X_axis/Y_axis files. Three lines would make each result .png a 3x3 grid of individual pictures, where the different lines in x_axis are appended to each prompt and vary from left to right. (Same idea for y_axis)
    - The different versions along an axis share the same seed! The only difference is the difference in lines in the axis files.
    - The call to paint_axis_from_file needs to take the file name ("X_axis", "Y_axis") as 3rd/4th argument to work.
    - The other option is to simply input a number as 3rd/4th argument, which does not vary the pictures in the grid, but instead just creates multiple versions with different seeds.
    - bigger grids need more gpu memory

    - e.g: red, green, blue as lines in x_axis, 1 as 4th argument creates a grid like this:
    | picture in red | picture in green | picture in blue |
    - if we add an y axis with two entries e.g. watercolor and digital art the result is:
    | watercolor in red | watercolor in green | watercolor in blue |
    ----------------------------------------------------------------
    | digitalart in red | digitalart in green | digitalart in blue |

Dear reader,


in this folder you can find the PYTHON scripts needed to obtain Fig. 4 in our paper "A Python toolbox for the numerical solution of the Maxey-Riley equation", where we use our implementation of PRasath et al. (2019)'s method to obtain the trajectory and relative velocity of a particle in an experimental field.

In order to generate the figure, you only need to run the file "a01_PLOTTR_DATA1.py" file and the figures will be generated shortly. You may decide to change the parameters within that file. We totally discourage the reader to change anything from the other files, unless she/he understand the code completely.

Once created, the figure is saved within the "VISUAL_OUTPUT" folder.



FIRST WARNING!

The ToolBox presented in this folder (Example_04) requires a matlab dataset with experimental data called (00_2mm_Faraday_50Hz_40ms_1_6g.mat) to be run.

This dataset is not provided for respect with the researchers that obtained the data.

In case of wanting accessibility to the data, please contact the authors.



SECOND WARNING!

Please be aware that changing the parameters related to the time grid, i.e. "tini", "tend", "nt", MAY require a recalculation of the matrix values to ensure convergence of the nonlinear solver.

This is automatically done by the ToolBox by deleting the a00_MAT-F_VALUES.npy, a00_MAT-H_VALUES.npy, a00_MAT-I_VALUES.npy, a00_MAT-Y_VALUES.npy files.

In short: if you encounter problems with the convergence of the solver, try to delete the .npy files and run the script again.


Have fun!

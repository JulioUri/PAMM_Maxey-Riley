Dear reader,


in this folder you can find the PYTHON scripts needed to obtain Fig. 3 in our paper "A Python toolbox for the numerical solution of the Maxey-Riley equation", which is a reproduction of Figure 11 in the paper "Accurate solution method for the Maxey-Riley equation, and the effects of Basset history" by Prasath et al. (2019).

In order to generate the figure, you only need to run the file "a01_PLOTTR_VORTX.py" file and the figures will be generated shortly. You may decided to change the parameters within that file. We totally discourage the reader to change anything from the other files, unless she/he understand the code completely.

Once created, the figure is saved within the "VISUAL_OUTPUT" folder.



WARNING below!

Please be aware that changing the parameters related to the time grid, i.e. "tini", "tend", "nt", MAY require a recalculation of the matrix values to ensure convergence of the nonlinear solver.

This is automatically done by the ToolBox by deleting the a00_MAT-F_VALUES.npy, a00_MAT-H_VALUES.npy, a00_MAT-I_VALUES.npy, a00_MAT-Y_VALUES.npy files.

In short: if you encounter problems with the convergence of the solver, try to delete the .npy files and run the script again.


Have fun!

# rs-fMRI-paper
This repository includes feature preparation part, for ABIDE-I resting state fMRI data.<br>
The repo includes three files:
* 1\. DownloadData.py  :  just an automated script to download ABIDE-I preprocessed, for some strategy and atlases specified.<br>
* 2\.  CalcCorr.ipynb : used to convert raw files to csv files, for easier pandas-based analysis. The file creates:
    * 2.1\. raw time signals per area, similar as input but csv but assigns each brain names to each column instead of numberic labels) <br>
    * 2.2\. functional connectivity matrix, calculated with pearson coefficient, for each couple of brain regions. The correlation is labeled by area names accrdingly <br>
    * 2.3\. Dist feature matrix, calculated as the euclidean distance between each two brain signals. the correlation is labeled by area names accrdingly. <br>
* 3\. CreateDynamicFeature.py dynamic functional connectivity, calculated with for a sliding windowed signals. window type (gaussian), width, and sliding step can be controlled. The output is the percantage of strong correlations, as well as weak/no correlation per pair of areas <br>

This code was used in a recently submitted journal paper, Nov22. <br>
-Please use citation the provided citation if used. <br>
--For any questions, suggestions, feel free to contact me.

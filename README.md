# IACT event types

A set of scripts for testing event types on IACT DL2 data.

The library for parsing data, training machine learning models, testing the performance and for plotting is in the event_types directory.
The scripts for the various event type studies are in the scripts directory. Specifically the scripts are really lacking documentation now. However, they are also fairly simple and self explanatory. No input arguments are taken at the moment, all necessary configuration is hardcoded within the scripts.
The recommended way to study, e.g., regression, is to first run the train_models.py script after selecting the models you would like to train.
Once the models are trained, run the compare_models.py scripts, making sure the models in the list to compare were trained beforehand.

# Setup

 ```
 conda env create -n event_types -f environment.yml
 source activate event_types
 python setup.py install
 ```

Once the package is set up, next time just run

```source activate event_types```

# Requirements

Please note this is still work in progress. Documentation is outdated and clearly lacking.

In order to execute the scripts, you will need some eventDisplay input files. These are root files containing DL2 information from the eventDisplay analysis done by Gernot Maier at DESY.

A description of their content and links to download them can be found here:
* https://forge.in2p3.fr/projects/cta_analysis-and-simulations/wiki/Eventdisplay_Prod3b_DL2_Lists

The easiest way to download them is here:
* https://ccdcacli236.in2p3.fr:2880/vo.cta.in2p3.fr/MC/PROD3/DL2_evndisp/Paranal_20deg/
* https://ccdcacli236.in2p3.fr:2880/vo.cta.in2p3.fr/MC/PROD3/DL2_evndisp/LaPalma_20deg/



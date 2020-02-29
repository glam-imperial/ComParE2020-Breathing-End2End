This repository contains the code to replicate the <b>End-to-End Deep Sequence Modelling</b> baseline for the <b>Breathing Challenge</b> of the Interspeech 2020 Computational Paralinguistics Challenge (ComParE). [<a href="http://www.compare.openaudio.eu/">whitepaper</a>]

# Usage

This code was tested with Python 3.5.

## Install required packages
Execute:

    pip install -r requirements.txt
    
preferably on a separate virtualenv.


## Breathing Challenge package

1. Edit ```end2end/configuration.py``` - specifically the CHALLENGE_FOLDER needs to correspond to the path in your disk the challenge folder exists.

2. Execute ```end2end/data_generator.py``` - this will process the data and extract tf_records, to be used in the experiment by tensorflow.

3. (Optional) Edit ```end2end/experiment/experiment_setup.py``` - this file defines a Python dictionary with configuration values for the experiment.

4. Execute ```end2end/run_experiment.py``` - this will perform one trial of the End2End method for the Breathing Challenge, as described in the whitepaper.

5. Adapt ```end2end/run_experiment.py``` and ```end2end/experiment/core.py``` - make your own adaptation of this method and outperform this baseline!

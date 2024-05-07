# Restaurant Recommendation System
This is a recommendation model for restaurants using the Artificial Bee Colony (ABC) algorithm.

## Building the project
Create a local virtual environment
```
python -m venv venv
```

Activate the virtual environment
```
# Unix/MacOS
source venv/bin/activate

# Windows
venv/Scripts/activate
```

Install dependencies
```
python -m pip install -r requirements.txt
```

## Running the program

### To run the restaurant recommendation system
From the root directory, run the following command:
```
python ./model/abc_model.py ./data/new_restaurants_data.csv
``` 

Then enter your food preferences (eg. = 3,1,1,1,1,1,1) in the command prompt

### To test the curated datasets
From the root directory, run the following command and replace the <test_file> with desired .csv file:
```
python ./model/abc_model.py ./data/tests/<test_file>
``` 

### To run the standard ABC alogithm on 4 benchmark functions
From the root directory, run the following command:
```
python ./model/abc_model_continuous.py
``` 


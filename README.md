# swedish_chef
This project explores the usefulness of several machine learning algorithms applied to a dataset of mostly 19th century American cookbooks.

The data lives in master_data/cookbook_textencoded as 75 or so cookbooks hand-uploaded and tagged with XML.

Before use, you will likely need to set up the directories:
* tmp
* tmp/rnn_data
* tmp/rnn_checkpoints
* tmp/clusters
* data
* data/cookbook_textencoded

The main.py provides an interface (which is pretty clunky, I'm sad to say) to several activities:
It can preprocess the data, which should be done by using the pipeline command like so:
`python3 main.py -p master_data/cookbook_textencoded data/cookbook_textencoded`

This will take the cookbooks that are in the master_data folder, move them into the data/cookbook_textencoded folder, and process them down to several outputs that are then used by the machine learning algorithms.

The code that does the preprocessing lives in the folder called preprocessing.

The next command to issue is:
`python3 main.py --train tmp/ingredient_table tmp/unique.txt tmp/unique_within.txt`

This command will take three files (all of which were generated from the preprocessing pipeline):
* The ingredient table (which holds all of the ingredients in several forms)
* The unique.txt, which holds all of the ingredients in order, with no duplicates
* The unique_within.txt, which holds all of the ingredients in order, grouped into recipes, with duplicates inside of recipes removed.

It takes these three files and it generates some more files which are of use to it while it runs several models on the data:
* Word2Vec is used to calculate similarities between ingredients
* KMeans is used to cluster the ingredients into clusters of ingredients that often go together
* A recurrent neural network is created and trained on the recipes to create a model which can then generate brand new recipe text.

The RNN bears a little more explanation.

It trains on a random subset of the data by taking each recipe along with its ingredients and learns at a word-encoding level the sequences of words that belong in a recipe, given the ingredients used in that recipe. This allows you to querry it after it has been trained by giving it a list of ingredients and it will use those to come up with what text. In the future, it would be good to get recipes that have steps tagged so that it could be trained on steps - first step given these ingredients, second step given the first step and these ingredients, etc.

Finally, to use the now trained models:
`python3 main.py -s NUMBER Y/N OPTIONAL_LIST`
Where:
* NUMBER is the number of ingredients you want the swedish chef to generate
* y or n: y to have it generate a recipe along with ingredients, n for no
* OPTIONAL LIST is an optional list of ingredients: "apple" "pear" "bread" - if given, this will constrain the ingredients generator to generate ingredients that it believes goes well with the given list of ingredients.




                                       Proactive  Chat
----------------------------------------------------------------------------------------------------------------
This is a pytorch port of generative model for proactive chat


Requirements
-----------------------------------------------
python>=3.6
pytorch>=1.0
tqdm
numpy
nltk
scikit-learn


Quickstart
-----------------------------------------------
Step 1: Preprocess
preprocess all the data using in model training and testing stage with the following commands

    python ./tools/convert_conversation_corpus_to_model_text.py corpus_file text_file topic_file index_file 1 1

Step 2: Train
train model with the following commands. make sure all the config in network.py is ready before model training
    python ./network.py --gpu 0

Step 3: Test
test model with the following commands.
    python ./network.py --test --ckpt models/best.model --gen_file ./output/test.result --gold_score_file ./output/gold.scores --use_posterior False --gpu 0 > log.txt

actually, you can run the script run_train.sh/run_test.sh training/testing model directly



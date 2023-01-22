"""
Follow these steps:
● Create a file called semantic.py and run all the code extracts above.
● Write a note about what you found interesting about the similarities
between cat, monkey and banana and think of an example of your own.
● Run the example file with the simpler language model ‘en_core_web_sm’ and write a note on what you notice is different from the model
'en_core_web_md'.
● Host your solution on a Git host such as GitLab or GitHub.
    ○ Remember to exclude any venv or virtualenv files from your repo.
● Add the link for your remote Git repo to a text file named semantic_similarity.txt


"""

import spacy
nlp = spacy.load('en_core_web_md')

# Example in task pdf - comparing similarity of four words
print("\n==== Similarity Example In PDF - using en_core_web_md ====\n")
tokens = nlp('cat apple monkey banana')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Interesting to see that can that 'animals' and 'fruits' are reasonably similar to each other in this model.
# Monkey and banana are higher than monkey and apple, which suggests it's taking into account what monkeys are popularised eating.
# Monkey and banana are also the same length, which could also be playing into their similarities.

# My example - comparing the similarity of four new words
print("\n==== My Similarity Example - using en_core_web_md ====\n")
my_words = nlp('doctor footballer woman man')
for word1 in my_words:
    for word2 in my_words:
        print(word1.text, word2.text, word1.similarity(word2))

# In my example, it's interesting to see that man and woman are similar - perhaps as genders of humans.
# There doesn't seem to be much gender bias in the model with associating doctor or football to either man or woman more.


nlp = spacy.load('en_core_web_sm')

# Example in task pdf - comparing similarity of four words
print("\n==== Similarity Example In PDF - using en_core_web_sm ====\n")
tokens = nlp('cat apple monkey banana')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# My example - comparing the similarity of four new words
print("\n==== My Similarity Example - using en_core_web_sm ====\n")
my_words = nlp('doctor footballer hairdresser woman man')
for word1 in my_words:
    for word2 in my_words:
        print(word1.text, word2.text, word1.similarity(word2))


# When using the simpler en_core_web_sm model the similarity scores are quite different.
# This is because it's only based on tagger, parser and NER. It doesn't ship with word vectors and only uses
# context-sensitive tensors. This is potentially why cat and apple are more similar in this model - they're both nouns.
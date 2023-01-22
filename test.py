import spacy

nlp = spacy.load('en_core_web_md')

# Tokenisation of each word?
word1 = nlp("cry")
word2 = nlp("sob")
word3 = nlp("banana")

print(f"Similarity of '{word1}' to '{word2}': {word1.similarity(word2)}")
print(f"Similarity of {word3}' to '{word2}: {word3.similarity(word2)}")
print(f"Similarity of {word3}' to '{word1}: {word3.similarity(word1)}")

print(type(word1))

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)

print(f"\nSentence to Compare: {sentence_to_compare}\n")
print(f"Sentence Similarities: \n")

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

doc_sentence = nlp(sentence_to_compare)

for word in doc_sentence:
    print(word.text, word.pos_)

pos_list = ['AUX', 'SCONJ', 'PRON', 'NOUN', 'ADP', 'DET']

print("\n==== EXPLANATIONS ====\n")
for word in pos_list:
      explanation = spacy.explain(word)
      print(f"{word} means: {explanation}")
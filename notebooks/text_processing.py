import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import RegexpParser

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# The sentences
prompt_response_2 = "The image features a group of penguins standing on rocks near the water."

# Tokenize and part-of-speech tag the sentences
tagged2 = pos_tag(word_tokenize(prompt_response_2))

# Define the chunk grammar to identify noun phrases
grammar = "NP: {<DT>?<JJ>*<NN.*>+}"

# Parse the sentences
cp = RegexpParser(grammar)
tree2 = cp.parse(tagged2)


noun_phrases2 = [' '.join(leaf[0] for leaf in subtree.leaves())
                for subtree in tree2.subtrees()
                if subtree.label() == 'NP']

# Print the noun phrases
print(noun_phrases2)
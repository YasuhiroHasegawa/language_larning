import gradio as gr
import nltk
from nltk import pos_tag, word_tokenize
from pattern.en import comparative, superlative

# NLTK's resources download (only needed the first time)
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def transform_and_compare(original_text, comparison_target, superlative_group):
    tokens = word_tokenize(original_text)
    tagged = pos_tag(tokens)
    adjectives = [(pos, word) for pos, (word, tag) in enumerate(tagged) if tag == 'JJ']
    tokens_comparative = tokens[:]
    tokens_superlative = tokens[:]
    for pos, adj in adjectives:
        tokens_comparative[pos] = comparative(adj)
        tokens_superlative[pos] = "the " + superlative(adj) if superlative(adj)[0] in 'aeiou' else superlative(adj)

    # Correctly handle the articles "a" and "an" for superlative transformations
    if tokens_superlative[0] in ['a', 'an']:
        tokens_superlative[0] = 'the'

    # Removing any end period from the original sentence
    if tokens_comparative[-1] == '.':
        tokens_comparative.pop()
    if tokens_superlative[-1] == '.':
        tokens_superlative.pop()

    comparative_sentence = ' '.join(tokens_comparative) + f' than {comparison_target}.'
    superlative_sentence = ' '.join(tokens_superlative) + f' among {superlative_group}.'
    return comparative_sentence, superlative_sentence

iface = gr.Interface(
    fn=transform_and_compare,
    inputs=["text", "text", "text"],
    outputs=["text", "text"],
    title="Adjective Transformer",
    description="Enter a sentence to transform its adjectives into comparative and superlative forms, correcting articles where necessary."
)

iface.launch()

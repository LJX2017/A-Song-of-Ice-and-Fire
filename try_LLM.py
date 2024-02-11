from transformers import pipeline
generator = pipeline('text-generation', model='KoboldAI/OPT-13B-Erebus')
print(generator("Welcome Captain Janeway, I apologize for the delay.", do_sample=True, min_length=50))
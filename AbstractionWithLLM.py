import transformers
import torch

class ReadAbstractClusters: #make sure this is read by label
    pass




class AbstractionWithLLM:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def create_abstractions_with_texts(self, texts):
        prompts  =  [self.transform_text_to_prompt(text) for text in texts]
        print(self.pipeline(prompts))


    def transform_text_to_prompt(self, text):
        return f"Turn this statement: I have an instrument that is able to {text}. Turn this statement into a " \
               f"more abstract statement."


if __name__ == "__main__":
    #trying to load llama-3
    # Use a pipeline as a high-level helper
    device = torch.device("cuda")
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from transformers import pipeline

    pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    texts = ["to provide protection from sun burn"]
    abs = AbstractionWithLLM(pipe)
    abs.create_abstractions_with_texts(texts)

